"""Validation-only metrics. No diagnosis or invented probability confidence."""

import numpy as np

THRESHOLDS = (2, 2.5, 3, 4)


def angle(a, b, c):
    u, v = a - b, c - b
    denominator = np.linalg.norm(u) * np.linalg.norm(v)
    if denominator < 1e-10 or not np.isfinite(denominator):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denominator, -1, 1))))


def clinical(points, codes):
    p = dict(zip(codes, points, strict=True))
    sna, snb = angle(p["S"], p["N"], p["A"]), angle(p["S"], p["N"], p["B"])
    return {"SNA": sna, "SNB": snb, "ANB": sna - snb}


def summary(errors):
    values = np.asarray(errors).ravel()
    finite = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "failed": int(values.size - finite.size),
        "MRE": float(finite.mean()) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "p95": float(np.percentile(finite, 95)) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        # Failed predictions stay in the denominator and never count as success.
        "SDR_percent": {
            str(t): float(100 * np.count_nonzero(values <= t) / values.size)
            if values.size
            else None
            for t in THRESHOLDS
        },
    }


def evaluate(predictions, targets, rows, codes):
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    pixel = np.linalg.norm(predictions - targets, axis=2)
    for index, row in enumerate(rows):
        valid = (
            np.isfinite(predictions[index]).all(axis=1)
            & (predictions[index] >= 0).all(axis=1)
            & (predictions[index] < [row["width"], row["height"]]).all(axis=1)
        )
        pixel[index, ~valid] = np.inf
    mm = pixel * np.array([r["spacing"] for r in rows])[:, None]
    calibrated = np.array([r["device"] != "ProMax 2D" for r in rows])
    by_device = {}
    for device in sorted({r["device"] for r in rows}):
        values = mm[[r["device"] == device for r in rows]]
        by_device[device] = {
            "spacing_status": "UNRESOLVED_CSV_PAPER_CONFLICT"
            if device == "ProMax 2D"
            else "dataset_declared_not_independently_calibrated",
            **summary(values),
        }
    angle_errors = {name: [] for name in ("SNA", "SNB", "ANB")}
    for index, (pred, target) in enumerate(zip(predictions, targets, strict=True)):
        predicted, reference = clinical(pred, codes), clinical(target, codes)
        needed_valid = all(
            np.isfinite(pixel[index, codes.index(code)])
            for code in ("S", "N", "A", "B")
        )
        for name in angle_errors:
            angle_errors[name].append(
                predicted[name] - reference[name] if needed_valid else float("nan")
            )
    angles = {}
    for name, values in angle_errors.items():
        values = np.array(values)
        finite = values[np.isfinite(values)]
        angles[name] = {
            "count": len(values),
            "invalid": int(len(values) - len(finite)),
            "bias_degrees": float(finite.mean()) if len(finite) else None,
            "MAE_degrees": float(abs(finite).mean()) if len(finite) else None,
            "RMSE_degrees": float(np.sqrt((finite**2).mean())) if len(finite) else None,
            "p95_absolute_degrees": float(np.percentile(abs(finite), 95))
            if len(finite)
            else None,
        }
    worst = sorted(range(len(rows)), key=lambda i: float(mm[i].max()), reverse=True)[
        :20
    ]
    return {
        "split": "valid",
        "images": len(rows),
        "calibration_warning": "ProMax 2D CSV 0.139 vs paper 1.139 mm/pixel; overall CSV metrics are provisional",
        "overall_mm_using_declared_csv_PROVISIONAL": summary(mm),
        "mm_excluding_spacing_conflict": summary(mm[calibrated]),
        "mean_pixel_error": float(pixel[np.isfinite(pixel)].mean())
        if np.isfinite(pixel).any()
        else None,
        "per_landmark_mm_PROVISIONAL": {
            c: summary(mm[:, i]) for i, c in enumerate(codes)
        },
        "by_device_mm": by_device,
        "clinical_degrees": angles,
        "outliers": [
            {
                "id": rows[i]["id"],
                "device": rows[i]["device"],
                "failed_landmarks": int((~np.isfinite(mm[i])).sum()),
                "max_finite_mm": float(mm[i][np.isfinite(mm[i])].max())
                if np.isfinite(mm[i]).any()
                else None,
            }
            for i in worst
        ],
        "test_evaluations": 0,
    }
