"""Deterministic NONCLINICAL graph for isolated transport tests, never a runtime fallback."""
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from app.schemas import ModelManifest


def synthetic_manifest() -> ModelManifest:
    return ModelManifest(
        name="test-fixture", version="test-v1", sha256="0" * 64,
        source="generated-test-graph", license="test-fixture-only",
        rights_review="synthetic software test only", landmark_definitions="arbitrary tensor positions, not anatomy",
        validation_report="NOT A CLINICAL MODEL", input_name="image", output_name="points",
        input_width=64, input_height=64, channels=3, resize="letterbox-bilinear-v1",
        scale="divide-by-255", mean=[0, 0, 0], std=[1, 1, 1], pad_value=0,
        output_adapter="coordinates-xy-confidence-v1", confidence_definition="synthetic test tensor",
        codes=["S", "N", "A", "B"],
    )


def export_fixture(directory: Path, manifest: ModelManifest | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = (manifest or synthetic_manifest()).model_copy(deep=True)
    values = np.array([[[0.35, 0.4, 0.9], [0.6, 0.4, 0.9], [0.7, 0.6, 0.8], [0.6, 0.65, 0.7]]], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("Constant", inputs=[], outputs=["points"], value=helper.make_tensor("synthetic_tensor", TensorProto.FLOAT, values.shape, values.flatten()))],
        "TEST_ONLY_NOT_CLINICAL", [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("points", TensorProto.FLOAT, [1, 4, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    path = directory / "test-only.onnx"
    onnx.save(model, path)
    manifest.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".onnx.json").write_text(manifest.model_dump_json())
    return path


if __name__ == "__main__":
    # This path is a disposable volume in the isolated test Compose project.
    # Deliberately do not accept the normal /app/models production mount.
    export_fixture(Path("/transport-fixture"))
    print("Exported NONCLINICAL transport fixture to isolated test volume.")
