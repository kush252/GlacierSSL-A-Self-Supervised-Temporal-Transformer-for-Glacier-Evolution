"""Sanitize dataset file/folder names for a Kaggle Dataset upload — reversibly.

Kaggle Datasets reject names containing characters outside ``[A-Za-z0-9._-]`` (apostrophes,
spaces, parentheses, accents, ...). The glacier dataset's region folders are named after
GLIMS glacier names, e.g. ``DES_BRAMES_DE_L'OURS_1`` (apostrophe) or
``BREITHORNGLETSCHER_(ZERMATT)`` -> rejected. This tool renames every offending path
component to a safe equivalent and rewrites the manifest's stored paths with the SAME
deterministic rule, so training on Kaggle still resolves images.

REVERSIBILITY (the important part):
* Every on-disk rename is appended, in order, to a log at
  ``src/artifacts/kaggle_sanitize/renames.json`` ({old, new} relative to the dataset root).
* Name collisions (two different names mapping to the same safe name) are disambiguated with
  a numeric suffix and recorded explicitly in that log under "collisions".
* The manifest is backed up to ``manifest.json.bak`` before rewriting.
* ``--revert`` replays the log in reverse to restore every original name and manifest.

The local SQLite DB is intentionally left untouched (it is not used on Kaggle; its paths
can be regenerated later and re-sanitized, or reverted via the log).

    # 1. preview (writes a PLAN file you can inspect in full)
    python src/scripts/sanitize_for_kaggle.py
    # 2. execute: rename on disk + rewrite manifest, saving the reversible log
    python src/scripts/sanitize_for_kaggle.py --apply --manifest
    # 3. undo everything if needed
    python src/scripts/sanitize_for_kaggle.py --revert
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glacier_ssl.config import load_config

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_LOG_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "kaggle_sanitize"
_LOG = _LOG_DIR / "renames.json"
_PLAN = _LOG_DIR / "plan.json"


def sanitize_component(name: str) -> str:
    cleaned = _SAFE.sub("_", name)
    cleaned = cleaned.strip("._-") or "x"
    return cleaned


def sanitize_relpath(relpath: str) -> str:
    parts = re.split(r"[\\/]", relpath)
    return "/".join(sanitize_component(p) if i else p for i, p in enumerate(parts))


def _plan_renames(root: Path):
    """Walk deepest-first; return ordered list of (abs_src, abs_dst) + collision records.

    Collisions (distinct names -> same safe name within one directory) get a numeric
    suffix; both the chosen target and the fact of collision are recorded.
    """
    ops = []
    collisions = []
    # os.walk topdown=False yields children before parents, so renames never invalidate a
    # path we still need to visit.
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        taken = set(os.listdir(dirpath))          # names currently present in this dir
        for name in sorted(filenames) + sorted(dirnames):
            safe = sanitize_component(name)
            if safe == name:
                continue
            final = safe
            if final in taken and final != name:
                n = 1
                while f"{safe}_{n}" in taken:
                    n += 1
                final = f"{safe}_{n}"
                collisions.append({"dir": str(Path(dirpath).relative_to(root)),
                                   "original": name, "naive": safe, "assigned": final})
            taken.discard(name)
            taken.add(final)
            ops.append((os.path.join(dirpath, name), os.path.join(dirpath, final)))
    return ops, collisions


def _rel(root: Path, p: str) -> str:
    return str(Path(p).relative_to(root)).replace("\\", "/")


def run(root: Path, apply: bool):
    if not root.exists():
        print(f"[skip] dataset root not found: {root}")
        return [], []
    ops, collisions = _plan_renames(root)
    for src, dst in ops[:25]:
        print(f"  {_rel(root, src)}  ->  {Path(dst).name}")
    if len(ops) > 25:
        print(f"  ... and {len(ops) - 25} more")
    print(f"[disk] {'renaming' if apply else 'would rename'} {len(ops)} component(s); "
          f"{len(collisions)} collision(s)")

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_root": str(root),
        "count": len(ops),
        "renames": [{"old": _rel(root, s), "new": _rel(root, d)} for s, d in ops],
        "collisions": collisions,
    }
    if not apply:
        _PLAN.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"[plan] full mapping written to {_PLAN}")
        return ops, collisions

    done = []
    for src, dst in ops:
        os.rename(src, dst)
        done.append({"old": _rel(root, src), "new": _rel(root, dst)})
    record["renames"] = done
    _LOG.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"[disk] done. reversible log written to {_LOG}")
    return ops, collisions


def rewrite_manifest(manifest_path: Path, apply: bool):
    if not manifest_path.exists():
        print(f"[skip] manifest not found: {manifest_path}")
        return
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    n = 0
    for g in data.get("glaciers", {}).values():
        if g.get("region"):
            g["region"] = sanitize_component(str(g["region"]))
        for obs in g.get("observations", []):
            for band, pth in list(obs.get("band_paths", {}).items()):
                new = sanitize_relpath(pth)
                if new != pth:
                    obs["band_paths"][band] = new
                    n += 1
            if obs.get("scl_path"):
                obs["scl_path"] = sanitize_relpath(obs["scl_path"])
    print(f"[manifest] {'rewriting' if apply else 'would rewrite'} {n} band path(s)")
    if apply:
        shutil.copy2(manifest_path, manifest_path.with_suffix(".json.bak"))
        manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[manifest] backup kept at {manifest_path.with_suffix('.json.bak')}")


def revert(root: Path, manifest_path: Path):
    if not _LOG.exists():
        print(f"[revert] no log at {_LOG}; nothing to undo.")
        return
    record = json.loads(_LOG.read_text(encoding="utf-8"))
    renames = record.get("renames", [])
    # replay in reverse: new -> old
    restored = 0
    for item in reversed(renames):
        new_abs = root / item["new"]
        old_abs = root / item["old"]
        if new_abs.exists():
            old_abs.parent.mkdir(parents=True, exist_ok=True)
            os.rename(new_abs, old_abs)
            restored += 1
    print(f"[revert] restored {restored}/{len(renames)} name(s)")
    bak = manifest_path.with_suffix(".json.bak")
    if bak.exists():
        shutil.copy2(bak, manifest_path)
        print(f"[revert] manifest restored from {bak}")
    _LOG.rename(_LOG.with_suffix(".reverted.json"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--apply", action="store_true", help="actually rename on disk")
    ap.add_argument("--manifest", action="store_true", help="also rewrite manifest paths")
    ap.add_argument("--revert", action="store_true", help="undo using the saved log")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(__file__).resolve().parents[2]
    dataset_root = root / str(cfg.data.dataset_root)
    manifest_path = root / str(cfg.data.manifest_path)

    print(f"Dataset root: {dataset_root}")
    if args.revert:
        revert(dataset_root, manifest_path)
        return

    print("DRY RUN (no changes)\n" if not args.apply else "APPLYING CHANGES\n")
    run(dataset_root, args.apply)
    if args.manifest:
        rewrite_manifest(manifest_path, args.apply)
    if not args.apply:
        print("\nRe-run with --apply --manifest to execute (a reversible log is saved).")


if __name__ == "__main__":
    main()
