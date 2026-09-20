"""Pinned upstream W32 topology; random initialization, never fetch weights."""

import hashlib
from pathlib import Path

import torch.nn as nn

SOURCE_SHA256 = "c0f2f8a137836c6e8c489e4a1c46010bfce5308584e13d231565f7f90a2636f8"


def build_model():
    source = Path(__file__).parent / "vendor/pose_hrnet.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Pinned HRNet source checksum changed")
    from .vendor.pose_hrnet import PoseHighResolutionNet

    extra = {"FINAL_CONV_KERNEL": 1, "PRETRAINED_LAYERS": []}
    for stage, modules, channels in [
        (2, 1, [32, 64]),
        (3, 4, [32, 64, 128]),
        (4, 3, [32, 64, 128, 256]),
    ]:
        extra[f"STAGE{stage}"] = {
            "NUM_MODULES": modules,
            "NUM_BRANCHES": len(channels),
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4] * len(channels),
            "NUM_CHANNELS": channels,
            "FUSE_METHOD": "SUM",
        }
    model = PoseHighResolutionNet({"MODEL": {"NUM_JOINTS": 29, "EXTRA": extra}})
    # Same initialization as upstream, without invoking its checkpoint-loading path.
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=0.001)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    return model
