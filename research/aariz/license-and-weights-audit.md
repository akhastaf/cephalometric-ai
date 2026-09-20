# Dataset, code and weights audit — 2026-09-15

These are separate permission/provenance records. A code license is not assumed to license separately hosted weights or training data. No pretrained checkpoint has been downloaded, loaded, converted or approved.

| Component | Evidence | Finding / disposition |
|---|---|---|
| Aariz dataset | Figshare article 27986417, version 1; file 51041642; DOI `10.6084/m9.figshare.27986417.v1`; saved release metadata and verified archive | Published **CC BY 4.0**. Suitable copyright-license basis for an attributed development dataset; clinical/commercial release still requires privacy/intended-use review. |
| Aariz archive terms | Only `Readme.txt` and device CSV in addition to images/annotations; saved README | No additional restriction found in the archive README. Dataset license comes from the dataset deposit, not the source-code license. |
| Aariz loader/mapping source | `manwaarkhd/aariz` commit `0634b8b4e6783b13fb6383fac694d61123904d3c` | **MIT**, copyright Muhammad Anwaar Khalid (2023). Notice retained. Mapping parsed as literals; source not executed. |
| HRNet architecture source | `leoxiaobin/deep-high-resolution-net.pytorch` commit `6f69e4676ad8d43d0d61b64b1b9726f0c369e7b1` | **MIT**, copyright Leo Xiao (2019). Notice retained. Proposed W32 topology inspected against source/config; no model instantiated or trained. |
| Official HRNet ImageNet checkpoint | Upstream README names `hrnet_w32-36af842e.pth` through its model-zoo links | **NOT CLEARED.** This review has not established an independent artifact-level weights license, full checksum or complete training-data provenance. Do not infer permission from the MIT repository or abbreviated filename. |
| Official HRNet COCO/MPII pose checkpoints | Listed separately in upstream model zoo | **NOT CLEARED / NOT SELECTED.** No downloaded weights or assumed transfer-learning permission. |
| Initial pilot initialization | No external checkpoint | Proposed random initialization for an engineering/cost pilot; no claim that a short pilot establishes W32 clinical accuracy or adequate convergence. |

## Obligations and remaining review

Retain dataset citation, CC BY link, pinned release identifier and record of preprocessing/annotation changes in dataset/model cards and notices. Do not imply publisher endorsement. Preserve MIT notices for copied source/mapping material. The publication's own CC BY-NC-ND article license is distinct from the dataset deposit's CC BY license; do not copy publication figures into the commercial app on the strength of the dataset license.

The authors report ethics approval for open CC BY publication of anonymized data. Their acquisition section also describes consent for educational/research use. Retain this evidence and resolve any applicable privacy/consent obligations in commercial-release review; do not claim that CC BY itself grants personality/privacy rights or medical-device clearance. No attempt at re-identification is permitted. No clinic patient data was uploaded to a cloud notebook.

For any future checkpoint require: exact artifact URL/revision; full SHA-256; explicit weights license/notice; licensor and artifact scope; original architecture/training recipe and data provenance; downstream commercial/redistribution conditions; preprocessing and output semantics; local approval decision/date. If evidence is missing, keep `pretrained_checkpoint = null`. Fine-tuning does not remove upstream obligations.

## Attribution to retain

Khalid, Muhammad Anwaar; Zulfiqar, Kanwal; Bashir, Ulfat; Shaheen, Areeba; Iqbal, Rida; Rizwan, Zarnab; Rizwan, Ghina; Fraz, Muhammad Moazam (2025). *Aariz Cephalometric Dataset*. Figshare, version 1. https://doi.org/10.6084/m9.figshare.27986417.v1. CC BY 4.0. DentalFlow's proposed preprocessing, ID mapping and unrounded consensus targets are adaptations, not author-endorsed changes.

Sources: [dataset deposit](https://figshare.com/articles/dataset/Aariz_Cephalometric_Dataset/27986417), [dataset API](https://api.figshare.com/v2/articles/27986417), [CC BY legal terms](https://creativecommons.org/licenses/by/4.0/legalcode.en), [authors' paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12313948/), [Aariz pinned source](https://github.com/manwaarkhd/aariz/tree/0634b8b4e6783b13fb6383fac694d61123904d3c), [HRNet pinned source](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/tree/6f69e4676ad8d43d0d61b64b1b9726f0c369e7b1). Source hashes are in `evidence/source-provenance.json`.
