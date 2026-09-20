# HRNet-W32 engineering pilot

Runnable research tooling, separate from the CPU-only production worker. **No real pilot, full training, Colab charge or clinical validation has been performed.** Start with [the Colab notebook](Aariz_HRNet_W32_pilot.ipynb). Both preparation and pilot execution default to disabled. Full training is not implemented.

## What this implementation fixes

- All 29 outputs use the audited stable landmark IDs and unrounded annotator mean.
- Original split assignments remain unchanged. Preparation applies the reviewed 18-image training-only exclusion proposal, giving **682 train / 150 valid**. No test members are extracted and the dataset class rejects `test` before filesystem access. Near-duplicate candidates and patient independence remain unresolved.
- Whole-archive SHA-256, selected file hashes, manifest IDs and exclusions are checked. Training verifies the materialized data again. Use immutable CPU-prepared data locally on the Colab runtime.
- The source is the exact audited MIT HRNet implementation (vendored source hash checked before import). Its separate license is in `vendor/LICENSE`. No pretrained weights or upstream checkpoint loader is used. The untouched upstream file produces a Python syntax warning in an unused pretrained-loading branch; the wrapper does not invoke it.
- HRNet-W32 uses standard 1/4/3 fusion-module stages, widths 32/64/128/256 and a 29-channel linear head. Input 768², target 192², sigma 2, masked MSE, microbatch 2, accumulation 4, Adam 0.001, AMP, gradient clipping 1. Source code and pilot config/schema/exclusion hashes are saved.
- Affine preprocessing retains image aspect ratio. Pillow edge/center conversion is explicit and tested against an image impulse. Targets use fractional centers and the inverse transform; screen coordinates are not involved. No flipping or prediction clamping.
- Validation reports MRE, SDR at 2/2.5/3/4 mm, per-landmark/device results, SNA/SNB/signed ANB errors, invalid predictions and worst cases. Millimetre values using ProMax 2D's CSV spacing are explicitly provisional, alongside results excluding that device. Failed points remain in SDR denominators. Incomplete validation is labeled and must never support model selection.

## Colab procedure

1. Review this branch and use its **exact 40-character commit SHA** in the notebook. Use Python 3.12. Start on CPU, mount your Drive, enable only `RUN_CPU_PREPARATION`, install pinned research dependencies and download/verify Aariz. The test set is never materialized.
2. Run the synthetic checks in the notebook on CPU. Inspect training-only coordinate overlays before any clinically meaningful experiment; automated image/point alignment is tested, but clinical annotation sign-off remains outstanding.
3. Disable CPU preparation. Switch to one GPU, immediately record UTC activation time, actual balance and displayed CCU/hour. Close other paid runtimes. Enable `RUN_PILOT`; repeat source/environment setup after the runtime switch. There is no automatic GPU selection or account purchase.
4. Run the pilot cell once with a new run directory. Setup, staging and validation count against the wall allowance. Limits: 3 epochs / 250 optimizer attempts / at most 30 session minutes. The wall allowance is `min(1800, 3600 * 0.8 * 10 / observed_rate)` seconds. Reserve 60 seconds for persistence and up to five minutes for validation. Rate changes require manual stop/review. This is not a provider-enforced billing cap.
5. The subprocess is bounded, reports progress every 50 steps, saves a final checkpoint/report, and the notebook disconnects the GPU in `finally`. If you stop before reaching that cell or installation fails, disconnect manually. A very slow Drive checkpoint can overrun the reserve; monitor Colab's balance.
6. On CPU, enter the settled ending CCU balance, any grants and an evidence reference. Concurrent/uncertain billing leaves actual usage null. Share the report before considering a full run. No GPU-hour estimate is reported as actual CCUs.

`report.json`, `validation.json`, `pip-freeze.txt`, and `pilot-checkpoint.pt` are written to Drive under a new run name. Model parameters, optimizer/scaler and RNG states are preserved for analysis; **resume is deliberately unsupported** in this first bounded pilot. A later resume implementation must verify all provenance and restore sampler/partial-epoch and budget state. No checkpoint is loaded here, avoiding unsafe untrusted checkpoint deserialization.

## Scope changes from the original review outline

The initial 10+30-step synthetic GPU microbenchmark is not a separate paid phase: the real pilot records early training step timings inside its single capped session. CPU synthetic tests verify a full W32 forward/backward pass at 64²; this tests topology/gradients, not 768² GPU memory or clinical quality. Full-resolution CPU inference is checked separately. All measured GPU performance and CCU fields remain absent until Colab execution. No full-run scheduler, test-set evaluation, confidence calibration, ONNX export or application changes are included.

## Local verification

```sh
docker build -f research/aariz/pilot/Dockerfile.test -t dentalflow-aariz-pilot-tests .
docker run --rm --memory=3g --cpus=2 dentalflow-aariz-pilot-tests
```

This research image contains CPU PyTorch. Production `Dockerfile` and requirements remain unchanged and contain no PyTorch or research files. The ordinary service test suite can skip optional pilot tests when PyTorch is absent; run the command above to exercise them.
