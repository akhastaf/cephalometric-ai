# Proposed experiment matrix — no runs authorized by this file

Monthly planning budget: **200 Colab CCUs**. Use measured hourly consumption and throughput to revise allocations after the pilot. These are ceilings/reservations, not predictions of achievable accuracy or proof that every run fits. No simultaneous GPU runs or automatic top-ups.

| ID | Experiment | Candidate / settings | Data used | Proposed ceiling | Decision |
|---|---|---|---|---|---|
| A0 | Audit, transform and environment checks | No model training; CPU | Train/valid; test integrity/isolation only | 0 GPU CCUs (local CPU; Colab CPU consumption checked if used) | Complete/review first |
| P0 | Short primary pilot | HRNet-W32, 29 channels, 768² input, 192² heatmaps, random init unless independently cleared weights; microbatch 2, accumulation 4 | Train + validation | 10 CCUs; 30 min; 3 epochs; 250 optimizer steps, first limit wins | **Not started.** Report actual CCUs/time before any full run |
| H1 | Primary production candidate | W32 configuration revised once from pilot; at most 80 epochs, patience 12 | Train + validation only | Reserve 100 CCUs | Requires approved pilot report and predicted complete-run cost with headroom |
| H2 | One targeted W32 confirmation | Second seed or one clearly justified change; not a broad search | Train + validation only | Reserve 30 CCUs | Optional; choose hypothesis before running |
| U1 | Comparison baseline | Lightweight U-Net with 29 channels, same split/preprocessing/evaluation; CPU size/latency comparison | Train + validation only | Reserve 20 CCUs | Optional after H1; not primary model |
| E1 | Frozen final evaluation | Selected checkpoint + fixed decoder/calibration; all requested metrics | Sealed 150 test cases; optional external clinic set under separate authorization | Reserve 10 CCUs; prefer CPU | No selection/tuning after unsealing |
| X1 | ONNX parity / CPU benchmark | Selected W32 and, if trained, U-Net | Frozen validation/evaluation fixtures | Local CPU; reserve 0 GPU CCUs | After selection; do not quantize without repeat accuracy evaluation |
| R | Contingency | Interrupted runs, export corrections, measurement rerun | As appropriate without reusing test for selection | Hold 30 CCUs | Remains unspent unless needed |
| W48 | Deferred architecture escalation | HRNet-W48 | None initially | **0 CCUs this plan** | Only if W32 fails predefined validation accuracy/failure criteria and a new budget is approved |

Total reserved: 200 CCUs. If the measured H1 projection exceeds its allocation, propose a revised schedule/month instead of silently lowering image resolution, truncating evaluation or consuming contingency. A short randomly initialized pilot is a pipeline/budget check, not a meaningful clinical accuracy verdict.

## Required pilot report

- Hardware, GPU memory, actual Colab CCU/hour, starting/ending balances and elapsed-session evidence.
- Total session time, training/validation time, median step time, examples/second, peak allocated/reserved GPU RAM, OOM/AMP overflow events.
- Number of epochs, optimizer steps, train/validation images actually seen, transforms and all artifact hashes.
- Train loss and validation metrics, failures and per-device sample counts; no test score.
- Estimated full-run time/CCUs using measured throughput plus validation/checkpoint overhead and 20% headroom, explicitly labeled estimate.
- Recommendation: proceed, revise, or stop. Full experiment remains disabled until the report is reviewed.

## Final evaluation protocol

Freeze code, checkpoint SHA-256, preprocessing/decoder, calibration, selection criterion, random seed and schema before accessing test labels. Keep the official split; never move difficult examples out of it or merge validation/test into training. Hash/integrity checks are not a test performance evaluation.

For each image and landmark calculate radial error in original calibrated coordinates: `sqrt((dx*sx)^2 + (dy*sy)^2)`. Aariz supplies one pixel-size scalar per image, so `sx=sy=CSV pixel_size`, explicitly an isotropic assumption. Preserve the ProMax 2D CSV-versus-paper discrepancy in the report and include a clearly separated sensitivity result if still unresolved; never replace 0.139 with 1.139 silently.

Report overall **MRE**, **SDR at 2, 2.5, 3 and 4 mm**, all 29 **per-landmark MRE/SDR**, and **per-device** results with denominators. Add median, 95th percentile, maximum errors and patient-level bootstrap confidence intervals. SDR uses every expected prediction in its denominator; missing/invalid predictions count as failures. If nonfinite predictions prevent a finite overall MRE, report the all-case result as undefined/infinite plus conditional finite-case MRE and the explicit failure rate, never quietly omit failures.

Calculate **SNA**, **SNB**, and signed **ANB = SNA − SNB** using the same definitions as Nest and original geometry. Compare prediction-derived angles to annotation-derived angles: signed bias, MAE, RMSE, median/p95 absolute error, invalid-geometry counts. These targets are benchmark annotations, not independently established clinical truth. Report the unrounded-consensus protocol as primary and a secondary `ceil(mean)` calculation only for comparison with the author's reference loader.

Failure review includes worst cases, lost/padded points, mirrored/oriented input, annotation disagreement, device-specific problems and clinically sensitive points S/N/A/B. Test-case visual review occurs only **after** the selected model is frozen; subsequent changes require a new independent holdout. Clinical acceptability thresholds must be signed off before selection; SDR@2 mm is a benchmark metric, not universal clinical clearance.

After selection, export all 29 channels to ONNX; preserve a versioned raw-heatmap decoder and separately calibrated confidence. Compare PyTorch FP32 with ONNX Runtime CPU FP32 for heatmaps, decoded landmarks and angle results, including ties/flat peaks and padding. Proposed numerical starting tolerances: `atol=1e-4, rtol=1e-3` for heatmaps and maximum decoded difference 0.1 original pixel; establish and record justified exceptions before acceptance. Report errors rather than changing tolerances after seeing failures.

CPU benchmark: one inference at a time, fixed thread count, batch 1, documented CPU model/vCPU/RAM, input resolution, model size, startup/model-load time, 10 warmups and 100 timed requests. Report median/p95 inference and end-to-end latency, process peak RSS, and compare W32/U-Net only under identical conditions. No Railway deployment is authorized. Doctor corrections and explicit review remain required before clinically finalized reports; automatic measurements may be shown as unreviewed previews.
