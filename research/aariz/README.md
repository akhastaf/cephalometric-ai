# Aariz / HRNet-W32 review package

**Audit reviewed; pilot tooling added on 2026-09-20. No pilot/full run has started.** Plan corrected to **Google AI Pro, 200 Colab CCUs/month**. HRNet-W32 is the primary candidate with **all 29 Aariz landmarks**; U-Net is an optional comparison, and W48 remains deferred.

## Audit outcome

The full official archive is downloaded and verified: publisher MD5 matches; SHA-256 is recorded; all 4,002 files passed CRC; official image counts are 700 train / 150 validation / 150 test. All 1,700 train/validation landmark files passed structural/coordinate checks. Test labels remain unparsed, with only integrity/metadata/blind duplicate checks performed on test bytes.

**Two material findings remain explicit in the pilot:**

1. There are **six exact byte-duplicate cross-split pairs**: two train–test and four train–validation. There are also 12 pixel-identical pairs within training. A proposed 18-image **training-only exclusion mask** would leave 682 effective training images while retaining the original archive, split assignments, and all 150/150 validation/test cases. The original audit mask is preserved; the new pilot preparation applies it only to its derived training workspace. No real training workspace has been generated in this phase. Another 413 pHash candidate pairs require cautious adjudication; they are not confirmed duplicates.
2. ProMax 2D spacing is **0.139 mm/pixel in the actual CSV**, but **1.139 in the paper**. This affects 41 images. The CSV value is preserved and the conflict remains flagged for clarification.

These findings are supported by local file hashes and direct equality checks. An audit completing successfully does not mean the dataset is free of leakage or clinically validated.

## Deliverables

| Requested item | Review artifact |
|---|---|
| Dataset integrity, metadata and split audit | [dataset-audit.md](dataset-audit.md), [machine-readable results](evidence/audit-results.json) |
| Dataset/code/checkpoint licensing | [license-and-weights-audit.md](license-and-weights-audit.md), [source provenance](evidence/source-provenance.json) |
| Exact 29-landmark codes, definitions, IDs and ordering | [readable table](landmarks-29.md), [versioned JSON schema](landmarks-29.json) |
| HRNet-W32 topology, transforms, targets and training settings | [hrnet-w32.proposed.json](hrnet-w32.proposed.json) |
| Colab structure and CCU accounting | [notebook structure](colab-notebook-structure.md), [unexecuted review notebook](Aariz_HRNet_W32_review_outline.ipynb) |
| Runnable bounded pilot (2026-09-20) | [pilot guide](pilot/README.md), [Colab notebook](pilot/Aariz_HRNet_W32_pilot.ipynb) |
| Expected experiments and full evaluation | [experiment-matrix.md](experiment-matrix.md) |
| Actual time/CCU report fields | [pilot-report-template.json](pilot-report-template.json) — not run, values null |
| Proposed duplicate handling | [proposed-training-exclusions.json](proposed-training-exclusions.json) — review only, not applied |
| Later 29-point runtime/schema work and ONNX/CPU integration | [integration-after-selection.md](integration-after-selection.md) |
| Reproducible audit tool and synthetic safeguards | [audit_archive.py](audit_archive.py), [test_review_package.py](test_review_package.py) |

## Proposed pilot

HRNet-W32, 29 linear heatmaps, 768×768 letterboxed input, 192×192 targets, Gaussian sigma 2, fractional target centers, Adam at 0.001, microbatch 2 with accumulation 4, and AMP. No pretrained checkpoint is cleared; proposed random initialization is for pipeline/cost verification. A short pilot cannot establish clinical accuracy or convergence.

Stop at the earliest of **3 epochs / 250 optimizer steps / 30 minutes / planned 10 CCUs**, with runtime limits based on the actual displayed Colab rate and 20% headroom. Record actual balances and session evidence; do not substitute an estimated GPU-hour conversion for measured CCUs. Full training remains disabled until the pilot report and revised budget are reviewed.

The [pilot implementation and notebook](pilot/README.md) now implement this bounded workflow following the user’s instruction to continue. The original review notebook remains unchanged for provenance. The pilot uses pinned PyTorch 2.8.0 and separate research dependencies; actual Colab CUDA compatibility, cost and clinical accuracy await execution. No account connection or GPU allocation was performed.

## Verification and boundaries

- Full-file checksum verification and all ZIP CRC reads passed.
- All 1,700 train/validation annotations passed ID/symbol/count/bounds checks; all declared `open_issues=0`.
- Six cross-split duplicate pairs independently confirmed with byte-for-byte equality.
- **17 research tests passed** (6 audit/schema + 11 pilot safeguards); scoped Ruff check passed. The existing worker suite also passed **24 tests**. Production Docker build and a synthetic full-resolution HRNet forward pass passed; no clinical or Colab evaluation was performed.
- No production service, database, frontend or Docker configuration changed. The production Dockerfile only copies `app/` and `models/`; this research material and the raw dataset are not runtime contents.
- No Colab training time, CCU usage or model performance is available. No dataset image, annotation coordinates or checkpoint is committed in this package; committed evidence contains metadata/hashes only.

Dataset permission and source-code notices are preserved. Pretrained weights require their own documented rights/provenance review. Clinical release additionally needs validated measurement/landmark protocols and independent testing on the intended clinic population. Doctor correction and explicit validation remain mandatory before clinically finalized reports.
