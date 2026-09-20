"""Read-only release audit. Never trains; test labels are hashed, never parsed.

Requires Pillow and NumPy. Outputs metadata/hashes only, never source images.
Test image decoding is confined to automated integrity/duplicate checks.
"""

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
from io import BytesIO, StringIO
import json
import math
from pathlib import Path, PurePosixPath
import stat
import struct
import zipfile

import numpy as np
from PIL import Image, ImageOps


def landmark_config(path):
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "ANATOMICAL_LANDMARKS"
            for t in node.targets
        ):
            result = ast.literal_eval(node.value)  # Never execute downloaded code.
            if len(result) != 29:
                raise ValueError("Expected 29 source landmark IDs")
            return result
    raise ValueError("Missing source landmark mapping")


def parse_landmarks(raw, expected_ids, width, height):
    records = json.loads(raw)["landmarks"]
    ids = [p["landmark_id"] for p in records]
    if len(ids) != 29 or set(ids) != set(expected_ids):
        raise ValueError("Missing, duplicate or unknown landmark ID")
    points = {p["landmark_id"]: p["value"] for p in records}
    for value in points.values():
        x, y = value["x"], value["y"]
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in (x, y)
        ):
            raise ValueError("Nonfinite/non-numeric coordinate")
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Coordinate outside pixel-center image domain")
    return ids, points


def phash(gray):
    """64-bit DCT fingerprint for candidate screening, not duplicate proof."""
    small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    x = np.arange(32)
    basis = np.cos(np.pi * np.outer(np.arange(8), 2 * x + 1) / 64)
    coeff = basis @ small @ basis.T
    median = np.median(coeff.flatten()[1:])
    bits = (coeff > median).flatten()
    return sum(int(bit) << i for i, bit in enumerate(bits))


def duplicate_groups(records, key):
    groups = defaultdict(list)
    for row in records:
        groups[row[key]].append({"id": row["id"], "split": row["split"]})
    return [v for v in groups.values() if len(v) > 1]


def audit(archive, config, output):
    expected = landmark_config(config)
    output.mkdir(parents=True, exist_ok=True)
    md5, sha = hashlib.md5(), hashlib.sha256()
    with archive.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
    if md5.hexdigest() != "e0bd645bca6759abdae4f199d841bda6":
        raise ValueError("Archive does not match the pinned Figshare release")
    report = {
        "release": "10.6084/m9.figshare.27986417.v1",
        "archive_bytes": archive.stat().st_size,
        "archive_md5": md5.hexdigest(),
        "archive_sha256": sha.hexdigest(),
        "test_labels_parsed": 0,
        "test_model_evaluations": 0,
        "policy": "test bytes used only for CRC/hash, IDs and blind duplicate screening",
    }
    manifest, images, issues = [], [], []
    shapes, labels_count, orders = defaultdict(Counter), Counter(), Counter()
    label_issue_counts = Counter()
    device_counts = defaultdict(Counter)
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member paths")
        for i in infos:
            path = PurePosixPath(i.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in i.filename:
                raise ValueError("Unsafe archive path")
            if stat.S_ISLNK(i.external_attr >> 16) or i.flag_bits & 1:
                raise ValueError("Symlink/encrypted archive member")
            if i.file_size > 128 * 1024 * 1024:
                raise ValueError("Unexpectedly large archive member")
        rows = list(
            csv.DictReader(
                StringIO(
                    z.read("Aariz/cephalogram_machine_mappings.csv").decode("utf-8-sig")
                )
            )
        )
        metadata = {r["cephalogram_id"]: r for r in rows}
        if len(metadata) != len(rows):
            raise ValueError("Duplicate IDs in device CSV")
        report["metadata_rows"] = len(rows)
        report["metadata_columns"] = list(rows[0])
        report["csv_sha256"] = hashlib.sha256(
            z.read("Aariz/cephalogram_machine_mappings.csv")
        ).hexdigest()
        # Every file is read fully to validate ZIP CRC, including opaque test labels.
        for idx, info in enumerate(infos):
            if info.is_dir():
                continue
            raw = z.read(info)
            manifest.append(
                {
                    "path": info.filename,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "crc32": f"{info.CRC:08x}",
                }
            )
            parts = PurePosixPath(info.filename).parts
            if len(parts) < 4 or parts[2] != "Cephalograms":
                continue
            split, identifier = parts[1], Path(parts[-1]).stem
            if split not in ("train", "valid", "test"):
                raise ValueError("Unexpected split")
            row = metadata.get(identifier)
            if not row or row["mode"].lower() != split:
                raise ValueError("Image/CSV split disagreement")
            if row["image_format"].lower() != Path(parts[-1]).suffix[1:].lower():
                issues.append(
                    {
                        "id": identifier,
                        "split": split,
                        "issue": "Image extension/CSV disagreement",
                    }
                )
            spacing = float(row["pixel_size"])
            if not math.isfinite(spacing) or spacing <= 0:
                raise ValueError("Invalid pixel spacing")
            with Image.open(BytesIO(raw)) as original:
                original.load()
                orientation = original.getexif().get(274, 1)
                image = ImageOps.exif_transpose(original).convert("RGB")
                width, height = image.size
                if orientation != 1:
                    issues.append(
                        {
                            "id": identifier,
                            "split": split,
                            "issue": "EXIF orientation",
                            "value": orientation,
                        }
                    )
                gray = image.convert("L")
                fingerprint = phash(gray)
                pixels_hash = hashlib.sha256(
                    struct.pack("<II", width, height) + image.tobytes()
                ).hexdigest()
                shapes[split][f"{width}x{height}"] += 1
            record = {
                "id": identifier,
                "split": split,
                "path": info.filename,
                "width": width,
                "height": height,
                "device": row["machine"],
                "mm_per_pixel": spacing,
                "byte_sha256": manifest[-1]["sha256"],
                "pixel_sha256": pixels_hash,
                "phash": f"{fingerprint:016x}",
            }
            images.append(record)
            device_counts[(row["machine"], row["pixel_size"])][split] += 1
            for annotator in ("Junior Orthodontists", "Senior Orthodontists"):
                label_path = f"Aariz/{split}/Annotations/Cephalometric Landmarks/{annotator}/{identifier}.json"
                if label_path not in names:
                    raise ValueError("Missing paired annotation")
                if split == "test":
                    continue
                try:
                    raw_label = z.read(label_path)
                    label = json.loads(raw_label)
                    if label.get("ceph_id") != identifier:
                        raise ValueError("Annotation/image ID disagreement")
                    for point in label["landmarks"]:
                        if (
                            point["landmark_id"] in expected
                            and point["symbol"]
                            != expected[point["landmark_id"]]["symbol"]
                        ):
                            raise ValueError("Source landmark ID/symbol disagreement")
                    label_issue_counts[str(label.get("open_issues"))] += 1
                    ids, _ = parse_landmarks(raw_label, expected, width, height)
                    orders[tuple(ids)] += 1
                    labels_count[split] += 1
                except (ValueError, KeyError, TypeError) as exc:
                    issues.append(
                        {
                            "id": identifier,
                            "split": split,
                            "annotator": annotator,
                            "issue": str(exc),
                        }
                    )
            if len(images) % 100 == 0:
                print(
                    f"Audited {len(images)} images; no test labels parsed", flush=True
                )
        report["crc_verified_files"] = len(manifest)
        report["archive_entries"] = len(infos)
        report["uncompressed_bytes"] = sum(i.file_size for i in infos)
    ids = [r["id"] for r in images]
    if set(ids) != set(metadata) or len(ids) != len(set(ids)):
        raise ValueError("Duplicate image IDs or unmatched metadata rows")
    counts = Counter(r["split"] for r in images)
    if counts != Counter(train=700, valid=150, test=150):
        raise ValueError("Official split counts differ")
    report["splits"] = dict(counts)
    report["paired_annotations_parsed"] = dict(labels_count)
    report["train_valid_annotation_open_issues_values"] = dict(label_issue_counts)
    report["annotation_order_variants"] = [
        {"count": count, "symbols": [expected[i]["symbol"] for i in order]}
        for order, count in orders.items()
    ]
    report["devices"] = [
        dict(device=device, mm_per_pixel=float(spacing), splits=dict(c))
        for (device, spacing), c in sorted(device_counts.items())
    ]
    report["image_shapes"] = {s: dict(c) for s, c in shapes.items()}
    report["issues"] = issues
    report["byte_duplicates"] = duplicate_groups(images, "byte_sha256")
    report["decoded_pixel_duplicates"] = duplicate_groups(images, "pixel_sha256")
    candidates = []
    for i, left in enumerate(images):
        for right in images[i + 1 :]:
            if left["split"] == right["split"]:
                continue
            distance = bin(int(left["phash"], 16) ^ int(right["phash"], 16)).count("1")
            if distance <= 4:
                candidates.append(
                    {
                        "left": left["id"],
                        "left_split": left["split"],
                        "right": right["id"],
                        "right_split": right["split"],
                        "phash_hamming": distance,
                    }
                )
    report["cross_split_phash_candidates_hamming_le4"] = candidates
    report["limitations"] = [
        "Image identifiers are not independently verified patient identities.",
        "pHash candidate screening does not prove absence of cropped/edited or same-patient images.",
        "Test label contents intentionally remain unvalidated until model selection is locked.",
        "Device calibration is declared metadata, not independently measured physical calibration.",
    ]
    for filename, value in [
        ("audit-results.json", report),
        ("file-manifest.json", manifest),
        ("image-manifest.json", images),
    ]:
        (output / filename).write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n"
        )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "splits",
                    "paired_annotations_parsed",
                    "crc_verified_files",
                    "issues",
                    "byte_duplicates",
                    "decoded_pixel_duplicates",
                    "cross_split_phash_candidates_hamming_le4",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.archive, args.config, args.output)
