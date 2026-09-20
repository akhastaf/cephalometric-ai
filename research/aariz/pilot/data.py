"""Hash-verified train/valid-only loader. Test annotations are never opened."""

import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset

from research.aariz.audit_archive import parse_landmarks

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA256 = "d9fa872b36065dac9615cfcad0c7512c450fe2d86a1839cdec4cbe001def33ea"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(archive, destination):
    """CPU-stage materialization, never extract test or arbitrary ZIP paths."""
    if digest(archive) != ARCHIVE_SHA256:
        raise ValueError("Archive SHA-256 mismatch")
    destination = Path(destination)
    if destination.exists():
        raise ValueError(
            "Use a new empty destination; existing data is never overwritten"
        )
    rows = json.loads((ROOT / "evidence/image-manifest.json").read_text())
    manifest = {
        r["path"]: r
        for r in json.loads((ROOT / "evidence/file-manifest.json").read_text())
    }
    exclusions = {
        r["id"]
        for r in json.loads((ROOT / "proposed-training-exclusions.json").read_text())[
            "exclusions"
        ]
    }
    selected = [
        r
        for r in rows
        if r["split"] == "valid"
        or (r["split"] == "train" and r["id"] not in exclusions)
    ]
    if (
        sum(r["split"] == "train" for r in selected) != 682
        or sum(r["split"] == "valid" for r in selected) != 150
    ):
        raise ValueError("Unexpected reviewed split counts")
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise ValueError("Duplicate ZIP member names")
        for row in selected:
            row["labels"] = [
                f"Aariz/{row['split']}/Annotations/Cephalometric Landmarks/{a}/{row['id']}.json"
                for a in ("Junior Orthodontists", "Senior Orthodontists")
            ]
            for name in [row["path"], *row["labels"]]:
                parts = Path(name).parts
                if (
                    len(parts) < 3
                    or parts[0] != "Aariz"
                    or parts[1] not in ("train", "valid")
                    or ".." in parts
                ):
                    raise ValueError("Unsafe or held-out member")
                raw = z.read(name)
                if hashlib.sha256(raw).hexdigest() != manifest[name]["sha256"]:
                    raise ValueError("Member hash mismatch")
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
    (destination / "pilot-manifest.json").write_text(json.dumps(selected, indent=2))
    return {"train": 682, "valid": 150, "test_materialized": 0, "excluded_train": 18}


def geometry(width, height, size, rng=None):
    # Pillow affine maps pixel edges; applying the same matrix to annotation
    # coordinates follows the reviewed x/W convention, without integer rounding.
    s = min(size / width, size / height)
    affine = np.array(
        [[s, 0, (size - width * s) / 2], [0, s, (size - height * s) / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    if rng is not None:
        angle, scale = np.deg2rad(rng.uniform(-7, 7)), rng.uniform(0.9, 1.1)
        c, sn = scale * np.cos(angle), scale * np.sin(angle)
        center = size / 2
        dx, dy = rng.uniform(-0.02, 0.02, 2) * size
        aug = np.array(
            [
                [c, -sn, center - c * center + sn * center + dx],
                [sn, c, center - sn * center - c * center + dy],
                [0, 0, 1],
            ]
        )
        affine = aug @ affine
    return affine


def pillow_inverse(inverse):
    # Annotation coordinates refer to pixel centers (integer x/y). Pillow's
    # affine sampling coordinates refer to edges: T_edge = +.5 @ T @ -.5.
    matrix = inverse.copy()
    matrix[:2, 2] += 0.5 - matrix[:2, :2].sum(axis=1) * 0.5
    return tuple(matrix[:2].flat)


def transform_points(points, matrix):
    return (np.c_[points, np.ones(len(points))] @ matrix.T)[:, :2]


def heatmaps(points, size=768, sigma=2):
    side = size // 4
    center = points / 4 - 0.5
    grid_y, grid_x = np.mgrid[:side, :side]
    values = np.exp(
        -(
            (grid_x[None] - center[:, 0, None, None]) ** 2
            + (grid_y[None] - center[:, 1, None, None]) ** 2
        )
        / (2 * sigma**2)
    )
    mask = (
        np.isfinite(points).all(axis=1)
        & (points >= 0).all(axis=1)
        & (points < size).all(axis=1)
    )
    return np.nan_to_num(values).astype(np.float32), mask.astype(np.float32)


def decode(heatmap, inverse):
    n, h, w = heatmap.shape
    index = heatmap.reshape(n, -1).argmax(axis=1)
    points = np.c_[(index % w + 0.5) * 4, (index // w + 0.5) * 4]
    return transform_points(points, inverse)


class AarizDataset(Dataset):
    def __init__(self, root, split, size=768, seed=20260915):
        if split not in ("train", "valid"):
            raise ValueError("Pilot loader forbids test access")
        self.root, self.split, self.size, self.seed, self.epoch = (
            Path(root),
            split,
            size,
            seed,
            0,
        )
        rows = json.loads((self.root / "pilot-manifest.json").read_text())
        self.rows = [r for r in rows if r["split"] == split]
        expected_count = 682 if split == "train" else 150
        if len(self.rows) != expected_count:
            raise ValueError("Unexpected split size")
        evidence = {
            r["id"]: r
            for r in json.loads((ROOT / "evidence/image-manifest.json").read_text())
        }
        exclusions = {
            r["id"]
            for r in json.loads(
                (ROOT / "proposed-training-exclusions.json").read_text()
            )["exclusions"]
        }
        expected_ids = {
            i
            for i, r in evidence.items()
            if r["split"] == split and (split != "train" or i not in exclusions)
        }
        if {r["id"] for r in self.rows} != expected_ids:
            raise ValueError("Dataset identity/exclusion mismatch")
        files = {
            r["path"]: r["sha256"]
            for r in json.loads((ROOT / "evidence/file-manifest.json").read_text())
        }
        self.ids = [
            r["source_landmark_id"]
            for r in json.loads((ROOT / "landmarks-29.json").read_text())["landmarks"]
        ]
        for row in self.rows:
            original = evidence[row["id"]]
            if any(row[k] != original[k] for k in original):
                raise ValueError("Manifest changed from audit")
            labels = [
                f"Aariz/{split}/Annotations/Cephalometric Landmarks/{a}/{row['id']}.json"
                for a in ("Junior Orthodontists", "Senior Orthodontists")
            ]
            if row["labels"] != labels:
                raise ValueError("Label paths do not match split")
            for name in [row["path"], *labels]:
                path = (self.root / name).resolve()
                if (
                    not path.is_relative_to(self.root.resolve())
                    or digest(path) != files[name]
                ):
                    raise ValueError("Data hash/path mismatch")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        coords = []
        for name in row["labels"]:
            raw = (self.root / name).read_bytes()
            _, points = parse_landmarks(raw, self.ids, row["width"], row["height"])
            coords.append([[points[i]["x"], points[i]["y"]] for i in self.ids])
        points = np.mean(coords, axis=0)
        rng = (
            np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch, index])
            )
            if self.split == "train"
            else None
        )
        affine = geometry(row["width"], row["height"], self.size, rng)
        inverse = np.linalg.inv(affine)
        with Image.open(BytesIO((self.root / row["path"]).read_bytes())) as source:
            if (
                source.size != (row["width"], row["height"])
                or source.getexif().get(274, 1) != 1
            ):
                raise ValueError("Image geometry changed; re-audit required")
            image = source.convert("RGB").transform(
                (self.size, self.size),
                Image.Transform.AFFINE,
                pillow_inverse(inverse),
                Image.Resampling.BILINEAR,
                fillcolor=0,
            )
        if rng is not None:
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.1))
        values, mask = heatmaps(transform_points(points, affine), self.size)
        return {
            "image": torch.from_numpy(
                (np.asarray(image).astype(np.float32).transpose(2, 0, 1) / 255 - 0.5)
                / 0.5
            ),
            "target": torch.from_numpy(values),
            "mask": torch.from_numpy(mask),
            "points": points,
            "inverse": inverse,
            "id": row["id"],
            "device": row["device"],
            "spacing": row["mm_per_pixel"],
            "width": row["width"],
            "height": row["height"],
        }
