import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import logging
from time import perf_counter

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .downloading import download_image, validate_url
from .errors import ServiceError
from .inference import decode_output
from .model import ModelRuntime
from .preprocessing import decode_image, preprocess
from .schemas import ImageInfo, LandmarkRequest, LandmarkResponse, ModelInfo, Timings

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("cephalometry")
# HTTP clients must never emit URLs (signed query strings are credentials).
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
security = HTTPBearer(auto_error=False)


class BodyLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        events, size = [], 0
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            size += len(event.get("body", b""))
            if size > 16384:
                return await JSONResponse({"errorCode": "REQUEST_TOO_LARGE"}, status_code=413)(scope, receive, send)
            events.append(event)
            if not event.get("more_body", False):
                break
        async def replay():
            return events.pop(0) if events else await receive()
        await self.app(scope, replay, send)


def create_app(settings: Settings | None = None, runtime: ModelRuntime | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    model = runtime or ModelRuntime(settings)
    gate = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        await run_in_threadpool(model.load)
        app.state.model = model
        yield

    app = FastAPI(title="DentalFlow Cephalometric AI", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(BodyLimitMiddleware)

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, error: ServiceError):
        return JSONResponse({"errorCode": error.code}, status_code=error.status, headers={"Cache-Control": "no-store"})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        # Default Pydantic errors echo inputs, including signed URLs. Do not use them.
        return JSONResponse({"errorCode": "INVALID_REQUEST"}, status_code=422)

    async def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
        if not credentials or not settings.internal_api_key or not hmac.compare_digest(credentials.credentials.encode(), settings.internal_api_key.encode()):
            raise ServiceError("UNAUTHORIZED", 401)

    @app.get("/health")
    async def health():
        return {"status": "ok" if model.loaded else "degraded", "modelLoaded": model.loaded,
                "modelName": model.manifest.name if model.manifest else None,
                "modelVersion": model.manifest.version if model.manifest else None,
                "errorCode": None if model.loaded else "MODEL_NOT_CONFIGURED"}

    @app.post("/v1/cephalometric/landmarks", response_model=LandmarkResponse, dependencies=[Depends(authorize)])
    async def landmarks(payload: LandmarkRequest):
        validate_url(payload.imageUrl, settings)
        if not model.loaded:
            raise ServiceError("MODEL_NOT_CONFIGURED", 503)
        if payload.modelVersion and payload.modelVersion != model.manifest.version:
            raise ServiceError("MODEL_NOT_CONFIGURED", 503)
        if gate.locked():
            raise ServiceError("AI_UNAVAILABLE", 429)
        start = perf_counter()
        async with gate:
            try:
                data = await download_image(payload.imageUrl, settings, payload.imageSha256)
                download_done = perf_counter()
                def prepare():
                    image = decode_image(data, settings.max_image_pixels)
                    return preprocess(image, model.manifest)
                tensor, transform = await run_in_threadpool(prepare)
                preprocess_done = perf_counter()
                output = await run_in_threadpool(model.run, tensor)
                points = decode_output(output, model.manifest, transform)
                inference_done = perf_counter()
                timings = Timings(downloadMs=(download_done-start)*1000, preprocessMs=(preprocess_done-download_done)*1000,
                                  inferenceMs=(inference_done-preprocess_done)*1000, totalMs=(inference_done-start)*1000)
                response = LandmarkResponse(model=ModelInfo(name=model.manifest.name, version=model.manifest.version),
                                            image=ImageInfo(width=transform.width, height=transform.height), landmarks=points, timings=timings)
                logger.info(json.dumps({"event": "inference", "analysisId": str(payload.analysisId) if payload.analysisId else None,
                                        "modelVersion": model.manifest.version, "status": "completed", "landmarkCount": len(points), **timings.model_dump()}))
                return response
            except ServiceError as error:
                logger.warning(json.dumps({"event": "inference", "analysisId": str(payload.analysisId) if payload.analysisId else None,
                                           "status": "failed", "errorCode": error.code, "durationMs": (perf_counter()-start)*1000}))
                raise
            except Exception:
                logger.error("inference_failed errorCode=INVALID_AI_RESPONSE")
                raise ServiceError("INVALID_AI_RESPONSE", 503) from None

    return app


app = create_app()
