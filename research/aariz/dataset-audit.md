# Aariz release audit — 2026-09-15

**Audit completed with findings. Do not start the pilot until the split-leakage policy has been reviewed.** The official archive and its 700/150/150 assignment remain unchanged. No training or Colab runtime was started.

## Verified release and integrity

| Check | Observed result |
|---|---|
| Deposit | Figshare article 27986417, version 1; file 51041642 |
| Dataset DOI | `10.6084/m9.figshare.27986417.v1` |
| Archive size | 2,098,209,792 bytes |
| Published and computed MD5 | `e0bd645bca6759abdae4f199d841bda6` — match |
| Locally computed SHA-256 | `d9fa872b36065dac9615cfcad0c7512c450fe2d86a1839cdec4cbe001def33ea` |
| ZIP directory | 4,023 entries, including 21 directories |
| ZIP integrity | All 4,002 files fully read with ZIP CRC verification; no failure |
| Uncompressed content | 4,239,068,622 bytes; audited in memory per file, not fully extracted |
| Unsafe paths, duplicate member paths, encrypted members, symlinks | None found |
| Image formats | 370 PNG, 236 JPEG (.jpeg), 235 JPEG (.jpg), 159 BMP |
| Official image assignment | 700 train, 150 valid, 150 test; 1,000 unique anonymous image IDs |
| CSV association | 1,000 rows; image ID, extension and split associations matched |
| Annotations per image | Junior, senior and CVM files present; CVM not analyzed |
| Train/validation landmark annotations | 1,700 files parsed; every file had all 29 unique known IDs, matching symbols, finite in-bounds coordinates and matching image ID |
| Annotation issue marker | All 1,700 parsed train/valid files declared `open_issues=0` |
| Source array ordering | Same pinned 29-ID order in all 1,700 parsed files |
| Test labels parsed / model evaluations | **0 / 0** |

The ZIP was downloaded with bounded range requests after the sequential transfer stalled. The final full-file MD5 and SHA-256 cover the complete archive; there is no inference of integrity from partial downloads. The raw archive stays outside Git at `/private/tmp/dentalflow-aariz-audit-20260915/Aariz.zip`. Temporary storage is not permanent archival storage; Colab should redownload from the pinned source and verify both hashes.

## Confirmed duplicate images — split isolation fails

There are **17 identical-byte pairs** and **18 identical-decoded-RGB-pixel pairs**. The additional pixel-identical pair is inside training and differs only in file representation. All these groups map to ProMax with ProTouch in the CSV.

| Relationship | Identical bytes | Identical decoded pixels |
|---|---:|---:|
| Train ↔ test | 2 pairs | 2 pairs |
| Train ↔ validation | 4 pairs | 4 pairs |
| Within train | 11 pairs | 12 pairs |
| Total | 17 pairs | 18 pairs |

The six cross-split pairs were independently re-read and compared with direct byte equality, not only hash equality. The full evidence is in `evidence/cross-split-byte-verification.json`. Distinct anonymous IDs do not establish distinct patients.

| First ID | Split | Matching ID | Split | Shared SHA-256 |
|---|---|---|---|---|
| `cl5lg05uf019m074k8q27h7oy` | test | `cl5lg05uk01eq074k5kou14gg` | train | `85567b38ca0de4f4995f9e74ddccf15a99b205ca731458642bf6b2125abb1ea2` |
| `cl5lg05ul01g6074k007y5xhr` | test | `cl5lg05uq01km074kcn02bn6n` | train | `2aed7ea933afc18f13fef409fcd106f35882752d1c9c012f034d80b7ff24b3fc` |
| `cl5lg05uk01ei074k6cjy5n5s` | train | `cl5lg05ug01be074ke6h6g4zi` | valid | `ec7b0439b26994402084e19cca19d056d5c3b6c88e7febd4172b8850e82b5ffa` |
| `cl5lg05uk01eu074k43xm28on` | train | `cl5lg05ue0196074k9b619xdr` | valid | `f0d3dcd1860e5a6e40be1ad9fe2e966adaaf39069377029497fef145a6d0b54d` |
| `cl5lg05uk01fe074kdu5640v5` | train | `cl5lg05ug01ba074k7hvh8wpn` | valid | `769ce29a05cbda6dfe765ab4a911d7b34e48410d3bdf8b07bbb3e1074c459ed6` |
| `cl5lg05ul01ge074k302594d5` | train | `cl5lg05uq01ki074ket8j8p6r` | valid | `67903bf316183409d6db8b64658589ce0d204e3c8abf3f07fce0dcca68b1aba4` |

Automated DCT pHash screening found **413 cross-split candidate pairs** at Hamming distance ≤4, including the six exact overlaps. These are screening candidates, **not 413 proven duplicates**. Similar anatomy/backgrounds can collide. No test image was visually reviewed; further independent, blinded duplicate adjudication or creator clarification is needed before claiming near-duplicate/patient isolation.

**Proposed, not applied:** retain the official archive and every validation/test case. Apply a versioned training-only exclusion mask for the six held-out overlaps and one redundant copy from each of the 12 train-only pixel-identical pairs. This would use **682 of the original 700 training images**, with validation/test still 150/150 and no reassignment. The exact 18 IDs and deterministic tie-break rule are in `proposed-training-exclusions.json`. This is a disclosed filtered training subset, not an unchanged full-training benchmark. It needs your review and does not resolve all pHash candidates or prove patient identity isolation. Do not silently resplit, combine duplicate annotations, delete original data, or fit on leaked images.

## Device and calibration metadata

The archive CSV contains `cephalogram_id,machine,pixel_size,image_format,mode`. It gives a single positive scalar spacing per image, interpreted as isotropic mm/pixel. All 1,000 images have a matched row. No independently verifiable patient identifier or physical calibration acquisition record is supplied in that CSV.

| Device | CSV mm/pixel | Train | Valid | Test | Total |
|---|---:|---:|---:|---:|---:|
| ART Plus | 0.1 | 256 | 55 | 55 | 366 |
| Hyperion X5 | 0.089 | 101 | 21 | 21 | 143 |
| ProMax 2D | 0.139 | 29 | 6 | 6 | 41 |
| ProMax with ProTouch | 0.139 | 95 | 20 | 20 | 135 |
| Rotograph EVO | 0.135 | 54 | 12 | 13 | 79 |
| Smart3D | 0.1 | 41 | 9 | 9 | 59 |
| Veraviewepocs 2D | 0.144 | 124 | 27 | 26 | 177 |

**Calibration conflict:** the CSV consistently gives **0.139 mm/pixel** for 41 ProMax 2D images, while Table 2 of the paper gives **1.139 mm/pixel**. Preserve the CSV value as the declared release value, flag it as unresolved, and request creator clarification before treating mm-based scores as clinically established. Do not silently alter either source. A separate sensitivity analysis may show both assumptions without selecting whichever makes accuracy look better. The metadata check establishes consistency of the archive, not independently measured physical pixel calibration or a verified absence of earlier image resampling.

## Landmark protocol and loader discrepancies

The exact schema is in [landmarks-29.md](landmarks-29.md) and [landmarks-29.json](landmarks-29.json). Output order follows the pinned source config/observed train-valid JSON order: **Pn is index 8**. The paper lists Pronasale last, so copying paper row order would misassign channels. Inputs must join by stable `landmark_id`; the source loader instead relies on list order.

The source loader uses `ceil((junior+senior)/2)` and the deprecated `np.float` alias. The proposal retains fractional consensus coordinates and records the protocol difference. The source loader also turns the split into `Train/Valid/Test`, but the actual ZIP uses lowercase `train/valid/test`; that loader fails on a case-sensitive filesystem unless adapted. No downloaded source code was executed.

Clinical landmark definitions remain subject to clinician sign-off, particularly constructed Gn/Go, Ar and anatomical Po. Metadata/annotation bounds checks do not prove anatomical accuracy. The paper's inclusion/exclusion criteria also limit generalization to all dental patients; independent validation on the intended clinic population is still required.

## License disposition

The dataset deposit is CC BY 4.0; both Aariz mapping code and HRNet architecture code are MIT. Archive README contains no conflicting additional data restriction. Full source/weights separation, attribution, consent/privacy considerations and the unapproved pretrained-checkpoint register are in [license-and-weights-audit.md](license-and-weights-audit.md). No weights were downloaded or integrated.

## Reproduction and scope

Run on the complete pinned archive, using Python with NumPy 2.3.5 and Pillow 12.3.0 (the audit environment used here):

```sh
python research/aariz/audit_archive.py \
  --archive /path/to/Aariz.zip \
  --config research/aariz/evidence/aariz-source-config.py \
  --output /path/to/audit-output
python -m unittest discover -s research/aariz -p 'test_review_package.py' -v
```

Outputs include file-level hashes, image fingerprints/metadata, annotation checks, device counts and duplicate groups. Test files are read only for CRC/hash and blind duplicate screening/metadata association; test annotation content is not parsed and no test performance/selection analysis occurs. This narrowly scoped integrity access is necessary to check the requested split isolation. The pilot workspace must subsequently exclude all test content.

No training time, clinical metrics or actual Colab CCU consumption can be reported yet because **no run occurred**. `pilot-report-template.json` uses null fields rather than invented numbers. No Google account, GPU session, production model, database migration or deployment was used.

Sources: [pinned dataset deposit](https://figshare.com/articles/dataset/Aariz_Cephalometric_Dataset/27986417), [release API](https://api.figshare.com/v2/articles/27986417), [author code](https://github.com/manwaarkhd/aariz/tree/0634b8b4e6783b13fb6383fac694d61123904d3c), [paper Table 2 and definitions](https://pmc.ncbi.nlm.nih.gov/articles/PMC12313948/). Duplicate findings above come from this local audit of the verified archive, not from a published author claim.
