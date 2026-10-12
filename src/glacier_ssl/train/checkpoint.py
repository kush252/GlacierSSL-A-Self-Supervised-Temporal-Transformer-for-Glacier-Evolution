"""Phase 12 — checkpointing for reliable resumption.

Saves the student backbone, EMA teacher backbone, all heads, optimizer state, config, and
run metadata (build plan Phase 12).
"""
from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path, step, student_backbone, teacher_backbone, heads, optimizer,
                    config, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "student_backbone": student_backbone.state_dict(),
        "teacher_backbone": teacher_backbone.state_dict(),
        "heads": {name: h.state_dict() for name, h in heads.items()},
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    return path


def load_checkpoint(path, student_backbone, teacher_backbone, heads, optimizer=None,
                    map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    student_backbone.load_state_dict(ckpt["student_backbone"])
    teacher_backbone.load_state_dict(ckpt["teacher_backbone"])
    for name, h in heads.items():
        if name in ckpt["heads"]:
            h.load_state_dict(ckpt["heads"][name])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0)
