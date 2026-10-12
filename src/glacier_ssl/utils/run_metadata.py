"""Recorded run-metadata block (Phase 1): config snapshot + git/env state + seeds.

Satisfies Cross-cutting rule #8 (reproducibility) by persisting enough to reconstruct
a run. Git state is captured if the project happens to be a repo; absence is recorded,
not fatal (the project root is not a git repo in the current environment).
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _git_state() -> dict[str, Any]:
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
        if root.returncode != 0:
            return {"available": False}
        sha = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return {"available": True, "commit": sha, "dirty": bool(dirty)}
    except Exception:
        return {"available": False}


def write_run_metadata(output_dir: str | Path, config: dict[str, Any],
                       seed: int, extra: dict | None = None) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": seed,
        "python": sys.version,
        "platform": platform.platform(),
        "git": _git_state(),
        "config": config,
    }
    if extra:
        meta["extra"] = extra
    try:
        import torch
        meta["torch"] = torch.__version__
        meta["cuda_available"] = torch.cuda.is_available()
    except Exception:
        pass
    path = out / "run_metadata.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return path
