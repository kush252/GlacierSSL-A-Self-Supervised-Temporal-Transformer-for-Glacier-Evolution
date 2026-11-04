"""GPU acceleration helpers: device resolution, AMP, and DataParallel wrapping.

Keeps the acceleration policy in one place so the trainer stays readable. Everything here
is a no-op on CPU, so the CPU code path (and the test suite) is unaffected.

Multi-GPU uses ``nn.DataParallel`` (single-process, splits the batch across visible GPUs).
It is the pragmatic choice for a Kaggle notebook with 2x T4 and this custom multi-call
trainer. DistributedDataParallel is more efficient but needs a multi-process launcher
(torchrun / mp.spawn); see the README note.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn


def autocast_ctx(device_type: str, dtype, enabled: bool):
    """autocast context when enabled, else a no-op (avoids CPU+float16 validation errors)."""
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=True)


def resolve_device(requested) -> torch.device:
    """Accepts a string ("cpu"/"cuda"/"auto"/None) or an already-resolved torch.device."""
    if requested in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(requested)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


def use_data_parallel(policy, device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    n = torch.cuda.device_count()
    if isinstance(policy, bool):
        return policy and n > 1
    if str(policy).lower() == "true":
        return n > 1
    if str(policy).lower() == "false":
        return False
    return n > 1  # "auto"


def amp_settings(cfg_amp, cfg_dtype, device: torch.device):
    """Return (enabled, dtype). AMP is only enabled on CUDA."""
    enabled = bool(cfg_amp) and device.type == "cuda"
    dtype = torch.bfloat16 if str(cfg_dtype).lower() in ("bf16", "bfloat16") else torch.float16
    return enabled, dtype


class AutocastRunner(nn.Module):
    """Wraps a module so autocast is applied INSIDE forward.

    This is the correct pattern for DataParallel + AMP: autocast is thread-local and DP
    runs each replica's forward on its own thread, so enabling autocast only in the main
    thread would miss the replicas. Applying it inside forward covers every replica.
    """

    def __init__(self, module: nn.Module, amp_enabled: bool, amp_dtype, device_type: str):
        super().__init__()
        self.module = module
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype
        self.device_type = device_type

    def forward(self, *args, **kwargs):
        with autocast_ctx(self.device_type, self.amp_dtype, self.amp_enabled):
            return self.module(*args, **kwargs)


def enable_tf32(flag: bool) -> None:
    if flag and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
