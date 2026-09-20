"""Explicit opt-in, CUDA-only engineering pilot. No full-run or resume mode."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import AarizDataset, ROOT, decode, digest
from .metrics import evaluate
from .model import build_model


def budget_seconds(rate, balance):
    if not all(math.isfinite(v) and v > 0 for v in (rate, balance)) or balance < 10:
        raise ValueError(
            "Record a positive actual hourly rate and at least 10 available CCUs"
        )
    return min(1800, 3600 * 0.8 * 10 / rate)


def actual_ccus(start, end, grants=0, concurrent=False):
    if concurrent or not all(math.isfinite(v) and v >= 0 for v in (start, end, grants)):
        return None
    consumed = start + grants - end
    return consumed if consumed >= 0 else None


def masked_loss(pred, target, mask):
    count = mask.sum()
    if count.item() == 0:
        raise ValueError("No valid heatmap targets")
    return (((pred.float() - target) ** 2).mean(dim=(-2, -1)) * mask).sum() / count


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def source_hash():
    h = hashlib.sha256()
    for path in sorted((ROOT / "pilot").rglob("*.py")):
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def run(args):
    if not args.authorize_pilot:
        raise ValueError("Pilot requires explicit --authorize-pilot")
    allowance = budget_seconds(args.ccu_per_hour, args.ccu_start)
    activated = datetime.fromisoformat(args.gpu_activated_at.replace("Z", "+00:00"))
    if activated.tzinfo is None:
        raise ValueError("GPU activation timestamp needs timezone")
    elapsed = (datetime.now(timezone.utc) - activated).total_seconds()
    if elapsed < 0 or elapsed >= allowance - 60:
        raise ValueError("GPU session budget already exhausted or timestamp in future")
    if not torch.cuda.is_available():
        raise ValueError(
            "Pilot requires a Colab CUDA GPU; use synthetic tests for CPU smoke checks"
        )
    output = Path(args.output)
    if output.exists():
        raise ValueError("Use a new output directory; no automatic resume or overwrite")
    output.mkdir(parents=True)
    deadline = (
        time.monotonic() + allowance - elapsed - 60
    )  # reserve teardown/checkpoint time
    report = {
        "status": "STARTING",
        "experiment": "aariz29-w32-engineering-pilot-v1",
        "full_training": False,
        "test_access": False,
        "ccu_start": args.ccu_start,
        "observed_ccu_per_hour": args.ccu_per_hour,
        "ccu_end": None,
        "actual_ccu_consumed": None,
        "gpu_activated_at": activated.isoformat(),
        "max_session_seconds": allowance,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source_sha256": source_hash(),
        "schema_sha256": digest(ROOT / "landmarks-29.json"),
        "config_sha256": digest(ROOT / "hrnet-w32.proposed.json"),
        "exclusions_sha256": digest(ROOT / "proposed-training-exclusions.json"),
        "steps": 0,
        "masked_landmarks": 0,
        "amp_skips": 0,
        "loss_history": [],
        "seed": 20260915,
        "dataset_policy": "18 reviewed train exclusions; 682 train / 150 valid; unresolved near-duplicates; provisional calibration",
        "pretrained": None,
    }
    save_json(output / "report.json", report)
    (output / "pip-freeze.txt").write_text(
        subprocess.check_output(
            [__import__("sys").executable, "-m", "pip", "freeze"], text=True
        )
    )
    model = optimizer = scaler = None
    started = time.monotonic()
    epoch = -1

    def checkpoint():
        if model is None or optimizer is None or scaler is None:
            return
        temporary = output / "pilot-checkpoint.pt.tmp"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "step": report["steps"],
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "provenance": report,
                "resume_supported": False,
            },
            temporary,
        )
        temporary.replace(output / "pilot-checkpoint.pt")

    try:
        random.seed(report["seed"])
        np.random.seed(report["seed"])
        torch.manual_seed(report["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Fail loudly if an operation has no deterministic implementation.
        torch.use_deterministic_algorithms(True)
        train = AarizDataset(args.data, "train")
        valid = AarizDataset(args.data, "valid")
        if time.monotonic() >= deadline:
            raise TimeoutError("Budget exhausted during dataset verification")
        generator = torch.Generator().manual_seed(report["seed"])
        loader = DataLoader(
            train,
            batch_size=2,
            shuffle=True,
            num_workers=2,
            generator=generator,
            pin_memory=True,
        )
        report["dataset_setup_seconds"] = time.monotonic() - started
        model_started = time.monotonic()
        model = build_model().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        scaler = torch.amp.GradScaler("cuda")
        torch.cuda.reset_peak_memory_stats()
        report["parameters"] = sum(p.numel() for p in model.parameters())
        report["model_setup_seconds"] = time.monotonic() - model_started
        training_started = time.monotonic()
        report["status"] = "TRAINING"
        training_deadline = deadline - min(
            300, max(0, deadline - time.monotonic()) * 0.25
        )
        stop = "max_epochs"
        for epoch in range(3):
            train.epoch = epoch
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            for batch_index, batch in enumerate(loader):
                if time.monotonic() >= training_deadline or report["steps"] >= 250:
                    stop = (
                        "wall_budget"
                        if time.monotonic() >= training_deadline
                        else "max_optimizer_steps"
                    )
                    break
                group_size = min(4, len(loader) - (batch_index // 4) * 4)
                step_started = time.monotonic()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    prediction = model(batch["image"].cuda(non_blocking=True))
                    loss = masked_loss(
                        prediction,
                        batch["target"].cuda(non_blocking=True),
                        batch["mask"].cuda(non_blocking=True),
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                scaler.scale(loss / group_size).backward()
                accumulated += 1
                report["masked_landmarks"] += int((batch["mask"] == 0).sum())
                if accumulated == group_size:
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(norm):
                        raise FloatingPointError("Nonfinite gradient")
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    report["amp_skips"] += int(scaler.get_scale() < old_scale)
                    optimizer.zero_grad(set_to_none=True)
                    accumulated = 0
                    report["steps"] += 1
                    torch.cuda.synchronize()
                    report["loss_history"].append(
                        {
                            "epoch": epoch + 1,
                            "step": report["steps"],
                            "loss_last_microbatch": float(loss.detach()),
                            "last_microbatch_seconds": time.monotonic() - step_started,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    )
                    if report["steps"] % 50 == 0:
                        checkpoint()
                        save_json(output / "report.json", report)
            checkpoint()
            if stop != "max_epochs":
                break
        report["training_seconds"] = time.monotonic() - training_started
        report["seconds_per_optimizer_attempt"] = (
            report["training_seconds"] / report["steps"] if report["steps"] else None
        )
        report["stop_reason"] = stop
        validation_started = time.monotonic()
        model.eval()
        predictions, targets, rows = [], [], []
        with torch.inference_mode():
            for batch in DataLoader(valid, batch_size=2, shuffle=False, num_workers=2):
                if time.monotonic() >= deadline:
                    break
                maps = model(batch["image"].cuda()).float().cpu().numpy()
                for i, heatmap in enumerate(maps):
                    if not np.isfinite(heatmap).all():
                        point = np.full((29, 2), np.nan)
                    else:
                        point = decode(heatmap, batch["inverse"][i].numpy())
                    predictions.append(point)
                    targets.append(batch["points"][i].numpy())
                    rows.append(
                        {
                            "id": batch["id"][i],
                            "device": batch["device"][i],
                            "spacing": float(batch["spacing"][i]),
                            "width": int(batch["width"][i]),
                            "height": int(batch["height"][i]),
                        }
                    )
        report["validation_seconds"] = time.monotonic() - validation_started
        report["validation_images"] = len(rows)
        report["validation_complete"] = len(rows) == 150
        if rows:
            codes = [
                p["code"]
                for p in json.loads((ROOT / "landmarks-29.json").read_text())[
                    "landmarks"
                ]
            ]
            save_json(
                output / "validation.json", evaluate(predictions, targets, rows, codes)
            )
        report["status"] = "PILOT_FINISHED_AWAITING_BILLING_AND_REVIEW"
    except BaseException as exc:
        report["status"] = "STOPPED"
        report["failure_type"] = type(exc).__name__
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
        save_json(output / "report.json", report)
        checkpoint()
        if (output / "pilot-checkpoint.pt").exists():
            report["checkpoint_sha256"] = digest(output / "pilot-checkpoint.pt")
            save_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-pilot", action="store_true")
    parser.add_argument("--ccu-start", type=float, required=True)
    parser.add_argument("--ccu-per-hour", type=float, required=True)
    parser.add_argument("--gpu-activated-at", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
