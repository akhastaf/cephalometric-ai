"""Synthetic CPU checks only: no Aariz data, downloads, GPU or paid runtime."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip(
    "torch", reason="Run research pilot Docker tests for optional PyTorch checks"
)

from .data import AarizDataset, decode, geometry, heatmaps, transform_points  # noqa: E402 -- optional PyTorch guard must run before research imports
from .metrics import clinical, evaluate, summary  # noqa: E402 -- optional PyTorch guard must run before research imports
from .model import SOURCE_SHA256, build_model  # noqa: E402 -- optional PyTorch guard must run before research imports
from .run import actual_ccus, budget_seconds, masked_loss  # noqa: E402 -- optional PyTorch guard must run before research imports


def test_test_loader_rejected_before_filesystem_access(tmp_path):
    with pytest.raises(ValueError, match="forbids test"):
        AarizDataset(tmp_path / "does-not-exist", "test")


def test_non_square_geometry_roundtrip_and_heatmap_decode():
    points = np.array([[17.2, 41.3], [151, 220], [71.5, 190.25]])
    forward = geometry(173, 251, 768)
    transformed = transform_points(points, forward)
    np.testing.assert_allclose(
        transform_points(transformed, np.linalg.inv(forward)), points, atol=1e-10
    )
    maps, valid = heatmaps(transformed)
    assert valid.tolist() == [1, 1, 1]
    decoded = decode(maps, np.linalg.inv(forward))
    assert np.abs(decoded - points).max() <= 2 / forward[0, 0]


def test_affine_image_and_point_alignment():
    from .data import pillow_inverse

    image = np.zeros((80, 160, 3), np.uint8)
    image[30, 70] = 255
    forward = geometry(160, 80, 640)
    rendered = Image.fromarray(image).transform(
        (640, 640),
        Image.Transform.AFFINE,
        pillow_inverse(np.linalg.inv(forward)),
        Image.Resampling.BILINEAR,
    )
    values = np.asarray(rendered)[:, :, 0].astype(float)
    y, x = np.mgrid[:640, :640]
    centroid = np.array([(values * x).sum(), (values * y).sum()]) / values.sum()
    np.testing.assert_allclose(
        centroid, transform_points(np.array([[70, 30]]), forward)[0], atol=0.01
    )


def test_train_transform_reproducibility_and_invalid_target_mask():
    a = geometry(100, 200, 768, np.random.default_rng(9))
    b = geometry(100, 200, 768, np.random.default_rng(9))
    np.testing.assert_array_equal(a, b)
    _, mask = heatmaps(np.array([[-1, 3], [100, 768], [40, 40]]))
    assert mask.tolist() == [0, 0, 1]


def test_budget_and_actual_billing_not_estimates():
    assert budget_seconds(4, 200) == 1800
    assert budget_seconds(32, 200) == 900
    for rate, balance in [(0, 200), (float("nan"), 200), (4, 9), (4, float("inf"))]:
        with pytest.raises(ValueError):
            budget_seconds(rate, balance)
    assert actual_ccus(200, 198.5) == 1.5
    assert actual_ccus(200, 201, grants=2) == 1
    assert actual_ccus(200, 198, concurrent=True) is None
    assert actual_ccus(200, 201) is None


def test_mre_sdr_failures_and_calibration_groups():
    codes = ["S", "N", "A", "B"]
    truth = np.array([[[20, 10], [10, 10], [10, 20], [20, 20]]] * 2, dtype=float)
    predicted = truth.copy()
    predicted[:, :, 0] += 2
    predicted[0, 0] = [-1, 10]
    rows = [
        {"id": "a", "device": "ART Plus", "spacing": 0.1, "width": 100, "height": 100},
        {
            "id": "b",
            "device": "ProMax 2D",
            "spacing": 0.139,
            "width": 100,
            "height": 100,
        },
    ]
    report = evaluate(predicted, truth, rows, codes)
    assert report["overall_mm_using_declared_csv_PROVISIONAL"]["failed"] == 1
    assert (
        report["overall_mm_using_declared_csv_PROVISIONAL"]["SDR_percent"]["2"] == 87.5
    )
    assert report["mm_excluding_spacing_conflict"]["count"] == 4
    assert report["clinical_degrees"]["SNA"]["invalid"] == 1
    assert summary([np.inf])["MRE"] is None
    json.dumps(report, allow_nan=False)


def test_angle_geometry_and_signed_anb():
    values = clinical(
        np.array([[20, 10], [10, 10], [10, 20], [20, 20]]), ["S", "N", "A", "B"]
    )
    assert values == pytest.approx({"SNA": 90, "SNB": 45, "ANB": 45})


def test_full_w32_forward_backward_and_masked_loss():
    torch.set_num_threads(1)
    model = build_model().train()
    images = torch.randn(2, 3, 64, 64)
    result = model(images)
    assert result.shape == (2, 29, 16, 16)
    assert 28_000_000 < sum(p.numel() for p in model.parameters()) < 30_000_000
    mask = torch.ones(2, 29)
    mask[:, 0] = 0
    target = torch.zeros_like(result)
    loss = masked_loss(result, target, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.final_layer.weight.grad is not None
    assert torch.isfinite(model.final_layer.weight.grad).all()
    with pytest.raises(ValueError, match="No valid"):
        masked_loss(result, target, torch.zeros_like(mask))


def test_vendor_source_is_exact_audited_file():
    from .data import digest

    assert digest(Path(__file__).parent / "vendor/pose_hrnet.py") == SOURCE_SHA256


def test_notebook_gates_and_no_saved_outputs():
    notebook = json.loads(
        (Path(__file__).parent / "Aariz_HRNet_W32_pilot.ipynb").read_text()
    )
    code = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    )
    assert "RUN_PILOT = False" in code
    assert "RUN_CPU_PREPARATION = False" in code
    assert "--authorize-pilot" in code
    assert "unassign" in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), "notebook", "exec")


def test_prepare_and_loader_verify_hashes_exclusions_and_never_read_test(
    tmp_path, monkeypatch
):
    import hashlib
    from io import BytesIO
    import zipfile
    from . import data

    root = tmp_path / "schema"
    (root / "evidence").mkdir(parents=True)
    ids = [f"landmark-{i}" for i in range(29)]
    (root / "landmarks-29.json").write_text(
        json.dumps({"landmarks": [{"source_landmark_id": i} for i in ids]})
    )
    image_buffer = BytesIO()
    Image.new("RGB", (16, 16), "white").save(image_buffer, format="PNG")
    raw_image = image_buffer.getvalue()
    raw_label = json.dumps(
        {"landmarks": [{"landmark_id": i, "value": {"x": 8, "y": 8}} for i in ids]}
    ).encode()
    rows, files = [], []
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for split, count in (("train", 700), ("valid", 150), ("test", 1)):
            for i in range(count):
                identifier = f"{split}-{i}"
                name = f"Aariz/{split}/Cephalograms/{identifier}.png"
                rows.append(
                    {
                        "id": identifier,
                        "split": split,
                        "path": name,
                        "width": 16,
                        "height": 16,
                        "device": "ART Plus",
                        "mm_per_pixel": 0.1,
                    }
                )
                for path, raw in [
                    (name, raw_image),
                    *[
                        (
                            f"Aariz/{split}/Annotations/Cephalometric Landmarks/{a}/{identifier}.json",
                            raw_label,
                        )
                        for a in ("Junior Orthodontists", "Senior Orthodontists")
                    ],
                ]:
                    z.writestr(path, raw)
                    files.append(
                        {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}
                    )
    (root / "evidence/image-manifest.json").write_text(json.dumps(rows))
    (root / "evidence/file-manifest.json").write_text(json.dumps(files))
    (root / "proposed-training-exclusions.json").write_text(
        json.dumps({"exclusions": [{"id": f"train-{i}"} for i in range(18)]})
    )
    monkeypatch.setattr(data, "ROOT", root)
    monkeypatch.setattr(data, "ARCHIVE_SHA256", data.digest(archive))
    original_read = zipfile.ZipFile.read

    def checked_read(self, name, *args, **kwargs):
        assert "/test/" not in name
        return original_read(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", checked_read)
    destination = tmp_path / "prepared"
    assert data.prepare(archive, destination)["train"] == 682
    assert not (destination / "Aariz/test").exists()
    dataset = AarizDataset(destination, "train", size=64)
    assert len(dataset) == 682
    assert all(r["id"] != "train-0" for r in dataset.rows)
    sample = dataset[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["target"].shape == (29, 16, 16)
    with pytest.raises(ValueError, match="never overwritten"):
        data.prepare(archive, destination)
    (destination / dataset.rows[0]["path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash/path"):
        AarizDataset(destination, "train")
