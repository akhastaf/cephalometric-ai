# Colab notebook structure — review only

The companion `.ipynb` contains section descriptions and safe configuration/ledger cells only. It installs nothing, downloads nothing, allocates no GPU and contains no training call. Implementing/running the pilot follows review of this package. Full training is a separate approval after measured pilot results.

| Cell group | Purpose | Data access / stop condition |
|---|---|---|
| 1. Scope and approval | Show HRNet-W32, 29 outputs, 200 monthly CCUs, 10-CCU pilot budget | Default `RUN_PILOT = False`; never infer approval from a previous experiment |
| 2. Runtime ledger | Record UTC time, remaining CCUs, displayed CCU/hour, GPU model/VRAM, Colab plan | Confirm balance in Colab UI; do not infer CCUs from GPU seconds |
| 3. Reproducible environment | Pin Python/PyTorch/CUDA-compatible versions and exact source commits; save pip freeze and code diff | Implement lock after a CPU import/export smoke check; no `pip install latest` or unpinned remote execution |
| 4. Dataset acquisition | Download pinned file 51041642, resume, verify MD5 and saved SHA-256; validate paths/CRC | CPU runtime first; fail on checksum mismatch; no automatic alternate dataset/mirror |
| 5. Audit evidence | Verify official split manifest, stable IDs, labels, devices, calibration issues and duplicate quarantine decisions | Reuse read-only audit; integrity-only test access, no test labels parsed or model run |
| 6. Training workspace | Materialize only train/valid plus metadata to runtime local disk | Test images/labels excluded from training directory and dataset constructors; read locally, checkpoint to Drive |
| 7. Targets and transforms | Map all 29 stable IDs; preserve both annotator groups; float mean; fractional heatmap centers | Train-only overlays for image/coordinate alignment; no test visualization |
| 8. Architecture and weights | Instantiate pinned HRNet-W32 definition with 29 heatmaps | `pretrained=None` unless separate weights audit is explicitly approved; code MIT is not weights approval |
| 9. CPU smoke checks | Tensor shapes, normalized inverse mapping, loss/gradient finiteness, one tiny batch | No clinical claim; verify affine/padding and Gaussian convention |
| 10. GPU microbenchmark | 10 warmup + 30 timed steps at microbatch 2, AMP; record peak VRAM | Counts toward pilot cap; no GPU allocation in this review notebook |
| 11. Pilot | At most 3 epochs / 250 optimizer steps / 30 minutes / planned 10 CCUs, whichever first | Seed fixed; train only; checkpoints every epoch and 50 optimizer steps; no automatic full run |
| 12. Validation | Validation MRE/SDR and angle errors, timing and failure counts | No tuning with test; calibration conflict reported, not silently corrected |
| 13. Pilot report | Save actual before/after CCUs, measured wall/GPU time, device, step rate, loss curves, validation metrics | Stop if actual CCUs unavailable; label as not measured instead of estimating actual consumption |
| 14. Disconnect and review | Persist report/checkpoint and release runtime | Explicit stop; full run requires a new reviewed budget/configuration |
| 15. Later full experiment | Reuse approved configuration, seed, budgets and checkpoint hashes | Disabled/not implemented here |
| 16. Later locked test/export | Record selection decision before unsealing test; evaluate once; ONNX parity and CPU benchmarks | Separate final evaluation notebook; no test loader in pilot notebook |

## CCU ledger requirements

Record balance immediately before GPU activation and immediately after disconnect/usage settlement. Record screenshot/resource-panel evidence references locally without patient images or credentials. Actual consumption is `start balance + grants/top-ups - end balance`, provided no other paid runtimes consumed the same account balance. Record any concurrent sessions, billing updates or rollover; otherwise mark the value inconclusive. The notebook cannot claim to meter Colab billing through an undocumented API.

At start, estimate a conservative runtime ceiling from the **displayed** hourly rate: `min(30 minutes, 60 × 0.8 × remaining pilot budget / observed CCUs per hour)`. Stop on a changed rate or approaching balance limit; use a wall-clock training callback and manually monitor billing. This is an operational budget, not a guaranteed provider-enforced 10-CCU cap. Environment installation and downloads occur on CPU before GPU activation. No top-up is authorized.

Persist checkpoints containing epoch/step, model and optimizer state, AMP scaler, RNG states, dataset/schema/config/source hashes, preprocessing version and cumulative wall time. Resume only when all identities match. Gradient accumulation does not increase the BatchNorm microbatch size.

Sources: [Google AI Pro: 200 CCUs](https://support.google.com/googleone/answer/14534406?hl=en), [Colab FAQ](https://research.google.com/colaboratory/faq.html). Training time and CCU usage are **not yet measured**.
