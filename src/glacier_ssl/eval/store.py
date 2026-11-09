"""Organised, append-only storage for evaluation results.

Every evaluation run gets ONE directory, ``<out_root>/<ckpt_run>__<ckpt_epoch>__<time>/``::

    run_manifest.json    what was evaluated and how (checkpoint, config, seed, args, git)
    summary.json         every headline number, one block per test
    REPORT.md            human-readable tables + figure links, paper-ready
    eval.log             full log of the run
    <test>/              one folder per test: metrics.json, tables (.csv + .md), figures
    cache/               extracted embeddings (large, safe to delete, never committed)

Plus ``<out_root>/results_index.jsonl`` -- one line per (run, test) with the headline
metrics, so different checkpoints / ablations can be compared later without opening each
run folder.
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


def _clean(obj: Any) -> Any:
    """JSON-safe: numpy -> python, NaN/inf -> None, Path -> str."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _fmt(v: Any, digits: int) -> str:
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return "nan" if not math.isfinite(float(v)) else f"{float(v):.{digits}f}"
    return str(v)


class ResultsStore:
    def __init__(self, root: str | Path, run_name: str, manifest: dict | None = None):
        self.root = Path(root)
        self.dir = self.root / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self._sections: list[tuple[str, str]] = []
        self._summary: dict[str, Any] = {"run_name": run_name, "tests": {}}
        self._manifest = manifest or {}
        self._t0 = time.time()
        self.write_json("run_manifest.json", self._manifest)

    # -- paths -------------------------------------------------------------------
    def sub(self, name: str) -> Path:
        p = self.dir / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cache_dir(self) -> Path:
        return self.sub("cache")

    # -- writers -----------------------------------------------------------------
    def write_json(self, rel: str, obj: Any) -> Path:
        path = self.dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_clean(obj), fh, indent=2)
        return path

    def write_table(self, rel_stem: str, rows: list[dict], digits: int = 4) -> str:
        """Write ``<stem>.csv`` and ``<stem>.md``; return the markdown table."""
        path = self.dir / rel_stem
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            return "_(no rows)_"
        cols: list[str] = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with open(path.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for r in rows:
                w.writerow([_clean(r.get(c)) for c in cols])
        md = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        for r in rows:
            md.append("| " + " | ".join(_fmt(r.get(c), digits) for c in cols) + " |")
        text = "\n".join(md)
        path.with_suffix(".md").write_text(text + "\n", encoding="utf-8")
        return text

    def save_fig(self, fig, rel: str, dpi: int = 200) -> str:
        """Save PNG (for the report) and PDF (vector, for the paper). Returns the PNG path
        relative to the run folder."""
        path = self.dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return str(Path(rel).with_suffix(".png")).replace("\\", "/")

    # -- report ------------------------------------------------------------------
    def add_section(self, title: str, body_md: str) -> None:
        self._sections.append((title, body_md))

    def record(self, test: str, headline: dict, notes: str = "") -> None:
        """Register a test's headline numbers in summary.json and the cross-run index."""
        self._summary["tests"][test] = {"headline": _clean(headline), "notes": notes}
        ck = self._manifest.get("checkpoint", {})
        args = self._manifest.get("args", {})
        line = {
            "run_name": self.run_name,
            "test": test,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "checkpoint": ck.get("path"),
            "step": ck.get("step"),
            "backbone": args.get("backbone"),
            "split": args.get("split"),
            "headline": _clean(headline),
        }
        with open(self.root / "results_index.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")

    def finalize(self) -> Path:
        self._summary["elapsed_seconds"] = round(time.time() - self._t0, 1)
        self.write_json("summary.json", self._summary)
        ck = self._manifest.get("checkpoint", {})
        args = self._manifest.get("args", {})
        head = [
            f"# Evaluation report - `{self.run_name}`", "",
            f"- checkpoint: `{ck.get('path')}` (step {ck.get('step')}, "
            f"epoch {ck.get('epoch')})",
            f"- backbone evaluated: `{args.get('backbone')}` | "
            f"split: `{args.get('split')}` | "
            f"glaciers: {self._manifest.get('n_glaciers')} | "
            f"timestamps: {self._manifest.get('n_timestamps')}",
            "- config + seed + git state: see `run_manifest.json`",
            "- every number below is also in `summary.json`; per-test details are in the "
            "sub-folders", "",
        ]
        body = []
        for title, md in self._sections:
            body += [f"## {title}", "", md, ""]
        path = self.dir / "REPORT.md"
        path.write_text("\n".join(head + body), encoding="utf-8")
        return path
