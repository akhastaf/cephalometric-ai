# Clinical model readiness — 2026-09-14

The software can load a supplied, verified ONNX artifact. No clinical weights have been downloaded or integrated. The normal development service still reports `MODEL_NOT_CONFIGURED`.

Update 2026-09-15: [Aariz/HRNet-W32 review package](../research/aariz/README.md) prepares an all-29-landmark training proposal under the corrected 200-CCU Google AI Pro budget. The complete archive audit confirmed six exact cross-split duplicate pairs and a CSV-versus-paper spacing discrepancy. No exclusion mask, training run or production schema change has been applied; model readiness remains unchanged.

Update 2026-09-20: [bounded HRNet-W32 pilot tooling](../research/aariz/pilot/README.md) is implemented with all 29 landmarks, no pretrained checkpoint and no full-training mode. It applies the training-only exclusion mask to a derived workspace when preparation is explicitly run. No real pilot has run; production model readiness is unchanged.

## Candidates examined

| Candidate | Published evidence | Integration decision |
|---|---|---|
| [cwlachap HRNet](https://huggingface.co/cwlachap/hrnet-cephalometric-landmark-detection) | The model card explicitly states MIT for the model. Its file list contains a PyTorch checkpoint and YAML configuration, not ONNX or the model constructor. | Promising candidate for publisher clarification, not currently reproducible. Do not guess the network or claim export parity. |
| [CephaloHRNet](https://github.com/Cestovatels/CephaloHRNet) | MIT training/inference code; the inspected README describes training and does not link a ready clinical checkpoint for this integration. | Does not supply an immediately usable model artifact. Training a new model remains out of scope. |
| [NLM CephViT](https://huggingface.co/nlm-dir/CephViT) | The card lists OpenMDW 1.1, an encrypted PyTorch checkpoint and a study on CBCT-derived digitally reconstructed radiographs. | This does not establish validation on the conventional lateral X-rays targeted by V1. No export or clinical integration attempted. |
| [CeLDA](https://github.com/ShanghaiTech-IMPACT/CeLDA) | Research implementation; the dataset is offered for research purposes. | No immediately verified commercial/redistributable ONNX package. |

For HRNet, the [published configuration](https://huggingface.co/cwlachap/hrnet-cephalometric-landmark-detection/raw/main/config.yaml) gives 768×768 input, 19 outputs and channel normalization, but does not establish all inference preprocessing/postprocessing details. Its model card calls `get_hrnet_w32(config)` without providing the implementation; the repository's [open discussion](https://huggingface.co/cwlachap/hrnet-cephalometric-landmark-detection/discussions/1) also reports that missing function. The precise output channel-to-landmark order, confidence semantics and export parity must be verified with the publisher. A model-card license statement is useful evidence, not a substitute for documenting the complete artifact and applicable rights.

## Required model handoff

Obtain the original architecture and inference code, immutable checkpoint/ONNX version and checksum, weights license and redistribution terms, data-use provenance, exact landmark order/definitions, complete preprocessing and output/confidence definitions. An ONNX export must reproduce the original model's output on an approved validation set. Then complete `models/manifest.example.json` from that evidence and configure the existing local model variables.

Clinical acceptance separately requires representative lateral X-rays, clinician annotations, accuracy and confidence evaluation, orientation/aspect-ratio checks and approval of landmark/reference protocols. Software transport tests cannot substitute for these checks.

## Synthetic transport fixture

`tests/onnx_fixture.py` creates a tiny constant-output graph only inside the disposable transport test volume. Its output positions are arbitrary and have no anatomical meaning. Production Docker excludes `tests/`; no fixture is copied to the normal `models/` folder or enabled in the development Compose stack. The test graph validates networking, object signing, ONNX execution, normalized-coordinate transforms, database persistence and review/report plumbing only.
