# DentalFlow Cephalometric AI

Stateless FastAPI service, Python 3.12, ONNX Runtime **CPU**. NestJS owns patient/tenant authorization, storage, BullMQ, persistence and all clinical calculations. This service receives a temporary signed GET URL, detects landmarks and returns normalized coordinates. It has no PostgreSQL or Redis connection and stores no images.

Repository: [akhastaf/cephalometric-ai](https://github.com/akhastaf/cephalometric-ai). New work starts on a separate branch from `dev` and opens a PR to `dev`. After owner approval and merge, a separate `dev` → `main` PR prepares a release. **Deployment is currently paused.** See [CONTRIBUTING.md](CONTRIBUTING.md) for branch policy, local hooks and the current GitHub plan limitation on private-repository protection.

## Run with DentalFlow locally

This repository belongs alongside `app-back` and `app-front` inside the [DentalFlow deployment workspace](https://github.com/akhastaf/dentalflow-deployment). The parent's `docker-compose.dev.yml` now starts the CPU AI service by default; no extra Compose overlay is needed. For a fresh workspace, clone this repository as `cephalometric-ai` and check out the development feature branch being tested (or `dev` after its PR is merged).

Set a strong `CEPH_AI_INTERNAL_API_KEY` in the parent's ignored `.env.dev`. The current local workspace already has a generated secret shared by Nest and Python. Never put it in Nuxt public configuration or commit it. Then run from the workspace root:

```sh
docker compose -f docker-compose.dev.yml --env-file=.env.dev up --build
```

After the initial build, the original `up` command works unchanged. The AI container is named `dentalflow-cephalometric-ai`, with the Compose service name `cephalometric-ai`. Nest calls `http://cephalometric-ai:8000` on the existing Docker network. Its port is not published on the host. The Nest worker consumes `ai-cephalometric` from the existing Redis service; Python has no queue listener. The existing MinIO image remains in the same bucket and is read through a temporary signed URL.

Check process health and logs:

```sh
docker compose -f docker-compose.dev.yml --env-file=.env.dev ps
docker compose -f docker-compose.dev.yml --env-file=.env.dev exec -T cephalometric-ai python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
docker compose -f docker-compose.dev.yml --env-file=.env.dev logs --tail=50 cephalometric-ai
```

Without approved weights, health is `degraded` with `modelLoaded:false` and `MODEL_NOT_CONFIGURED`; the container remains healthy because it can accept authenticated requests. That is an intentional unavailable-model state, not working landmark detection.

To test approved weights locally, put the artifact and manifest in the ignored `models/` directory and add to the parent's `.env.dev`:

```dotenv
CEPH_LOCAL_MODEL_PATH=/app/models/model.onnx
CEPH_LOCAL_MODEL_MANIFEST_PATH=/app/models/model.onnx.json
CEPH_LOCAL_MODEL_NAME=approved-model-name
CEPH_LOCAL_MODEL_VERSION=approved-immutable-version
```

The folder is mounted read-only. Recreate the service with `docker compose -f docker-compose.dev.yml --env-file=.env.dev up -d --build cephalometric-ai` to load a model/config change. Application code changes also require rebuilding; this runtime deliberately uses one production-style Uvicorn worker. The old `docker-compose.cephalometry.yml` overlay remains an empty compatibility file in the deployment workspace.

## Model availability

**No clinical model weights are included.** `/health` returns HTTP 200 with `status: degraded`, `modelLoaded: false`, `errorCode: MODEL_NOT_CONFIGURED` until an approved model and manifest load successfully. Authenticated inference returns HTTP 503 `MODEL_NOT_CONFIGURED`. Health is a process/liveness check; deployment readiness for clinical use requires inspecting `modelLoaded`, not only the HTTP code. There is no fake prediction fallback or automatic weights download.

Candidates inspected on 2026-09-13:

- [CeLDA, ShanghaiTech IMPACT](https://github.com/ShanghaiTech-IMPACT/CeLDA): research resources, without an immediately verified deployable ONNX artifact and complete commercial/redistribution rights for this integration.
- [CLdetection2023](https://github.com/5k5000/CLdetection2023): Apache-2.0 repository code and externally hosted PyTorch checkpoints. A code license alone does not establish rights to redistribute checkpoints or underlying training data. No weights were downloaded or integrated.

This is a decision to leave the adapter unconfigured, not a legal conclusion about every available model. Obtain a publisher-supported model, explicit weights license and data-rights review before enabling clinical use. Do not infer a license from GitHub's code-license badge.

## Configure an approved model

1. Obtain `model.onnx` through your approved artifact supply chain. Never use patient data or credentials in the artifact repository.
2. Verify source, immutable version, checksum, commercial use, redistribution rights, training data rights and landmark definitions. Store the actual license with the artifact.
3. Copy `models/manifest.example.json` to an approved manifest. Every `REPLACE` value and every input/preprocessing value is illustrative; replace them from the model's documented contract. The all-zero checksum intentionally cannot match a real model.
4. Set `MODEL_PATH=/app/models/model.onnx`, `MODEL_MANIFEST_PATH=/app/models/model.onnx.json`, `MODEL_NAME` and `MODEL_VERSION`. If manifest path is omitted it defaults to `MODEL_PATH` plus `.json`.
5. Supply artifacts through a private build context (Docker copies `models/`) only after redistribution approval, or a read-only runtime mount populated by your deployment process. The Git ignore excludes weights by default. No volume is needed for patient images.
6. Rebuild/restart to load a new model. Model startup validates SHA-256, identity, graph input/output contract and CPU provider. Run parity and clinical acceptance tests before changing the approved version.

### Adapter contract

One static float32 input `[1,C,H,W]`, C=1 or 3, named by the manifest. One float32 output. Preprocessing: decode JPEG/PNG; reject animation, malformed images and resource-limit violations; apply EXIF orientation; RGB or grayscale; proportional bilinear letterbox (center padding, nearest integer resized dimensions); divide by 255, then per-channel `(value-mean)/std`; contiguous NCHW. Exact padding/mean/std and input dimensions come from the manifest. If the model requires different preprocessing, add a reviewed versioned adapter with parity tests; do not alter preprocessing to make an unknown model fit.

Output adapters:

- `coordinates-xy-confidence-v1`: `[1,K,3]`, x/y normalized to the letterboxed input's continuous edges and genuine model confidence in `[0,1]`.
- `probability-heatmaps-v1`: `[1,K,Hh,Wh]`, probabilities in `[0,1]`. Argmax cell centers become coordinates; peak probability is confidence. This is **not** a confidence calibration guarantee. Raw logits, negative scores or uncalibrated heatmaps need a different reviewed adapter.

`codes` specifies exact output order. Letterbox transforms are inverted, returning x/y normalized against the EXIF-oriented original image, x right, y down, range `[0,1]`. Out-of-image/padding predictions fail validation rather than being clamped. Supported codes: S, N, A, B, Pog, Gn, Me, Go, ANS, PNS, Or, Po, U1, L1. S/N/A/B are required for the initial measurements; U1/L1 mean **incisal tips**, not roots or axes. The backend documentation defines every point.

## API and security

- `GET /health`: public process health and model identity; no patient data.
- `POST /v1/cephalometric/landmarks`: `Authorization: Bearer <CEPH_AI_INTERNAL_API_KEY>`; JSON `{analysisId?, imageUrl, imageSha256?, modelVersion?}`. Nest sends the checksum of the selected attachment to detect object replacement. The response contains model identity, oriented image dimensions, landmarks and `downloadMs`, `preprocessMs`, `inferenceMs`, `totalMs`.

Set a strong shared internal secret in Railway variables, never Nuxt public runtime config. `ALLOWED_IMAGE_ORIGINS` is a comma-separated exact storage-origin allowlist with no trailing slash, path or wildcard. Only your owned bucket endpoints belong here; redirects, URL credentials, fragments, arbitrary hosts and environment HTTP proxies are rejected. Production requires HTTPS storage. `ALLOW_INSECURE_IMAGE_HTTP=true` exists only for local MinIO. Do not allow attacker-controlled DNS origins. Nest uses 600-second signed GET URLs; no permanent storage credentials enter this service.

Requests are bounded to 16 KiB JSON and streamed images to `MAX_IMAGE_SIZE_MB` (default 20), `MAX_IMAGE_PIXELS` (40 million), maximum dimension 16000. The default download budget is 90 seconds and inference budget 60 seconds. Keep their sum plus preprocessing/cold start below Nest's `CEPH_AI_REQUEST_TIMEOUT_MS` (default 180000). A single async gate covers download and CPU inference; concurrent extra requests return retryable 429. ONNX runs one inter/intra-op thread, with an inference termination timer. Use exactly one Uvicorn worker initially.

Logs contain analysis ID, version, timings and status; HTTP access logging is disabled and signed URLs/image data are never logged. Python receives no patient name, tenant details or S3 secrets.

## Railway deployment

These are future deployment instructions, not authorization to deploy. No Railway service or auto-deployment is configured by the GitHub/Compose setup.

1. When deployment is explicitly approved, add an independent service to the existing Railway project/environment from `akhastaf/cephalometric-ai`, branch `main`. This repository already has its `Dockerfile` at the root, so use `/` as the service root directory. Do not give it database/Redis variables.
2. Configure `CEPH_AI_INTERNAL_API_KEY`, exact `ALLOWED_IMAGE_ORIGINS`, approved model settings, and optional size/time limits from `.env.example`. Railway provides `PORT`; the Docker command binds `0.0.0.0:$PORT`. CPU only, one replica/worker initially, enough memory for the selected model plus a 40M-pixel decode (benchmark actual usage).
3. Set Railway healthcheck path `/health`. Configure the Nest service with `CEPH_AI_URL` pointing to this service (private service hostname + port where available, or Railway HTTPS domain), the same secret, and `CEPH_AI_REQUEST_TIMEOUT_MS=180000`. The Python allowlist must match the **internal signed URL origin generated by Nest's existing S3 client**, which may differ from browser `S3_PUBLIC_ENDPOINT`.
4. Enable Serverless/App Sleeping if desired. The service has no background queue listener or periodic outbound polling; incoming HTTP wakes it. Do not configure an external keep-alive monitor. Nest remains running with BullMQ and Redis, using bounded retries/backoff to tolerate cold starts. See [Railway Serverless](https://docs.railway.com/deployments/serverless).
5. Deploy first without weights to verify authenticated `MODEL_NOT_CONFIGURED`, then with approved artifacts. Verify `modelLoaded:true`, rejection of missing/invalid tokens, and an approved non-patient validation image. Exercise concurrent tenant requests via Nest, not direct unauthenticated Python calls.
6. Apply the Nest migration and grant existing CASL permissions as described in `app-back/docs/cephalometry.md`. Do not use TypeORM synchronization in production.

## Local checks

```sh
docker build -t dentalflow-ceph-ai:test .
docker build -f Dockerfile.test -t dentalflow-ceph-ai-tests:test .
docker run --rm dentalflow-ceph-ai-tests:test python -m pytest -q
docker run --rm dentalflow-ceph-ai-tests:test ruff check app tests
```

`requirements-dev.txt` installs ONNX solely to create a tiny constant-output fixture **inside tests**. It is synthetic, clinically meaningless, never included in the production Docker context and never used as a production fallback. Production installs ONNX Runtime, not PyTorch or ONNX training libraries.

For local inference, use the parent `docker-compose.dev.yml` as described above. Missing model configuration is expected until approved weights are supplied.

## Clinical release gates

The implementation verifies software contracts, not clinical accuracy. Clinicians must validate landmark definitions (including bilateral-point conventions), EXIF/orientation, image aspect ratio/square pixels, preprocessing/export parity, confidence calibration, local population reference applicability and intra/interobserver accuracy. The backend requires explicit clinician validation for REVIEWED; AI completion never means clinically validated. No diagnosis, treatment plan, calibration-dependent millimetre measurement, CBCT, training or superimposition is implemented.
