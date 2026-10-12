"""Central logging (Phase 1).

Provides a stdlib logger plus a lightweight JSONL metric logger for per-loss values,
gradient norms, valid-sample counts, and collapse diagnostics demanded by the head
files' evaluation sections.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any


def get_logger(name: str = "glacier_ssl", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(level)
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                          datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


class MetricLogger:
    """Append-only JSONL metric sink. One record per ``log`` call."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, step: int, **metrics: Any) -> None:
        record = {"step": int(step), "t": time.time(), **metrics}
        self._fh.write(json.dumps(record, default=float) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
