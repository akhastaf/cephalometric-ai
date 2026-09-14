import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import hashlib
import threading

from fastapi.testclient import TestClient
import httpx
import numpy as np
from PIL import Image
from pydantic import ValidationError
import pytest

from app.config import Settings
from app.downloading import download_image, validate_url
from app.errors import ServiceError
from app.inference import decode_output
from app.main import create_app
from app.model import ModelRuntime
from app.preprocessing import decode_image, preprocess
from app.schemas import Landmark, LandmarkResponse
from tests.onnx_fixture import export_fixture, synthetic_manifest

AUTH = {"Authorization": "Bearer test-secret"}
URL = "https://storage.example/xray?X-Amz-Signature=private"


def png(w=200, h=100):
    data = BytesIO()
    Image.new("RGB", (w, h), (128, 64, 32)).save(data, "PNG")
    return data.getvalue()


@pytest.fixture
def settings():
    return Settings(internal_api_key="test-secret", allowed_image_origins=("https://storage.example",))


@pytest.fixture
def manifest():
    return synthetic_manifest()


def runtime_fixture(tmp_path, settings, manifest):
    path = export_fixture(tmp_path, manifest)
    settings = settings.model_copy(update={"model_path": str(path), "model_name": manifest.name, "model_version": manifest.version})
    runtime = ModelRuntime(settings)
    return settings, runtime


def mock_download(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(handler)))


def test_health_and_authentication(settings):
    with TestClient(create_app(settings)) as client:
        health = client.get('/health').json()
        assert health["modelLoaded"] is False and health["errorCode"] == "MODEL_NOT_CONFIGURED"
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            assert client.post('/v1/cephalometric/landmarks', json={"imageUrl": URL}, headers=headers).status_code == 401
        response = client.post('/v1/cephalometric/landmarks', json={"imageUrl": URL}, headers=AUTH)
        assert response.status_code == 503 and response.json()["errorCode"] == "MODEL_NOT_CONFIGURED"


@pytest.mark.parametrize('url', ['file:///etc/passwd', 'http://169.254.169.254/latest/meta-data', 'https://storage.example.evil.test/a', 'https://storage.example@evil.test/a', 'https://evil.test/a', 'https://storage.example/a#fragment', 'https://storage.example:443/a', 'not-a-url'])
def test_untrusted_url(settings, url):
    with pytest.raises(ServiceError, match="INVALID_IMAGE_URL"):
        validate_url(url, settings)


def test_invalid_request_does_not_echo_signed_url(settings):
    with TestClient(create_app(settings)) as client:
        response = client.post('/v1/cephalometric/landmarks', json={"imageUrl": URL, "extra": URL}, headers=AUTH)
        assert response.status_code == 422
        assert 'private' not in response.text and 'storage.example' not in response.text
        assert client.post('/v1/cephalometric/landmarks', content=b'x'*17000, headers=AUTH).status_code == 413


def test_no_credentials_fail_closed():
    with TestClient(create_app(Settings())) as client:
        assert client.post('/v1/cephalometric/landmarks', json={"imageUrl": URL}, headers=AUTH).status_code == 401


def test_invalid_and_oversized_image():
    with pytest.raises(ServiceError, match="UNSUPPORTED_IMAGE"):
        decode_image(b'not an image', 100000)
    with pytest.raises(ServiceError, match="IMAGE_TOO_LARGE"):
        decode_image(png(), 10)


def test_letterbox_and_reverse_coordinates(manifest):
    tensor, transform = preprocess(decode_image(png(), 100000), manifest)
    assert tensor.shape == (1,3,64,64) and tensor.dtype == np.float32
    assert transform.top == 16 and transform.resized_height == 32
    assert transform.normalized_original(0.5,0.5) == (0.5,0.5)
    assert transform.normalized_original(0,0.25) == (0,0)
    assert transform.normalized_original(1,0.75) == (1,1)
    np.testing.assert_allclose(tensor[0,:,32,32], np.array([128,64,32])/255, atol=1e-6)


def test_exif_orientation():
    image = Image.new('RGB', (200,100))
    exif = Image.Exif()
    exif[274] = 6
    data = BytesIO()
    image.save(data, 'JPEG', exif=exif)
    assert decode_image(data.getvalue(), 100000).size == (100,200)


@pytest.mark.parametrize('field,value', [('x',-0.1),('y',1.01),('confidence',float('nan')),('confidence',float('inf')),('code','unknown')])
def test_output_schema(field,value):
    values = dict(code='N',x=0.5,y=0.5,confidence=0.9)
    values[field] = value
    with pytest.raises(ValidationError):
        Landmark(**values)


def test_invalid_adapter_outputs(manifest):
    _, transform = preprocess(decode_image(png(),100000),manifest)
    for output in [np.zeros((1,3,3)), np.full((1,4,3), np.nan), np.zeros((1,4,3))]:
        with pytest.raises(ServiceError, match="INVALID_AI_RESPONSE"):
            decode_output(output, manifest, transform)


def test_download_redirect_expiry_limit_and_checksum(settings, monkeypatch):
    for status, code in [(302,'STORAGE_ERROR'),(403,'SIGNED_URL_EXPIRED'),(404,'STORAGE_ERROR')]:
        mock_download(monkeypatch, lambda req: httpx.Response(status, headers={'Location':'https://evil.test'}))
        with pytest.raises(ServiceError, match=code):
            asyncio.run(download_image(URL,settings))
        monkeypatch.undo()
    mock_download(monkeypatch, lambda req: httpx.Response(200, headers={'Content-Length':'999999999'},content=b''))
    with pytest.raises(ServiceError,match='IMAGE_TOO_LARGE'):
        asyncio.run(download_image(URL,settings))
    monkeypatch.undo()
    mock_download(monkeypatch, lambda req: httpx.Response(200,content=png()))
    with pytest.raises(ServiceError,match='IMAGE_CHANGED'):
        asyncio.run(download_image(URL,settings,'0'*64))


def test_real_onnx_runtime_and_api(tmp_path, settings, manifest, monkeypatch):
    settings, runtime = runtime_fixture(tmp_path,settings,manifest)
    mock_download(monkeypatch, lambda req: httpx.Response(200,content=png()))
    with TestClient(create_app(settings,runtime)) as client:
        assert runtime.session.get_providers() == ['CPUExecutionProvider']
        session = runtime.session
        for _ in range(2):
            response = client.post('/v1/cephalometric/landmarks',json={'imageUrl':URL,'imageSha256':hashlib.sha256(png()).hexdigest()},headers=AUTH)
            assert response.status_code == 200, response.text
            data = LandmarkResponse.model_validate(response.json())
            assert data.image.width == 200 and data.image.height == 100
            assert all(0<=p.x<=1 and 0<=p.y<=1 for p in data.landmarks)
        assert runtime.session is session  # Loaded exactly once at startup.


def test_checksum_mismatch_fails_closed(tmp_path,settings,manifest):
    settings,runtime = runtime_fixture(tmp_path,settings,manifest)
    Path = type(tmp_path)
    Path(settings.model_path).write_bytes(b'changed')
    runtime.load()
    assert not runtime.loaded and runtime.error_code == 'MODEL_NOT_CONFIGURED'


def test_single_inference_concurrency(tmp_path,settings,manifest,monkeypatch):
    settings,runtime = runtime_fixture(tmp_path,settings,manifest)
    mock_download(monkeypatch, lambda req: httpx.Response(200,content=png()))
    started, release = threading.Event(), threading.Event()
    original = runtime.run
    def slow_run(tensor):
        started.set()
        release.wait(5)
        return original(tensor)
    runtime.run = slow_run
    with TestClient(create_app(settings,runtime)) as client, ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(client.post,'/v1/cephalometric/landmarks',json={'imageUrl':URL},headers=AUTH)
        assert started.wait(5)
        try:
            response = client.post('/v1/cephalometric/landmarks',json={'imageUrl':URL},headers=AUTH)
            assert response.status_code == 429
        finally:
            release.set()
        assert first.result().status_code == 200
