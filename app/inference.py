import numpy as np

from .errors import ServiceError
from .preprocessing import Transform
from .schemas import Landmark, ModelManifest


def decode_output(output: np.ndarray, manifest: ModelManifest, transform: Transform) -> list[Landmark]:
    count = len(manifest.codes)
    if not np.isfinite(output).all():
        raise ServiceError("INVALID_AI_RESPONSE", 503)
    coordinates = []
    if manifest.output_adapter == "coordinates-xy-confidence-v1":
        if output.shape != (1, count, 3):
            raise ServiceError("INVALID_AI_RESPONSE", 503)
        coordinates = output[0].tolist()
    elif manifest.output_adapter == "probability-heatmaps-v1":
        if output.ndim != 4 or output.shape[:2] != (1, count) or min(output.shape[2:]) < 2 or output.min() < 0 or output.max() > 1:
            raise ServiceError("INVALID_AI_RESPONSE", 503)
        for heatmap in output[0]:
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            # Explicit adapter semantics: heatmap cell center and peak probability.
            coordinates.append(((x + 0.5) / heatmap.shape[1], (y + 0.5) / heatmap.shape[0], float(heatmap[y, x])))
    result = []
    for code, (x, y, confidence) in zip(manifest.codes, coordinates, strict=True):
        x, y = transform.normalized_original(float(x), float(y))
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= confidence <= 1):
            # No clamping, fake confidence, or predictions placed at the edge of padding.
            raise ServiceError("INVALID_AI_RESPONSE", 503)
        result.append(Landmark(code=code, x=x, y=y, confidence=float(confidence)))
    return result
