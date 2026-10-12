"""Configuration system (Phase 1).

A single source of truth loaded from YAML into a dot-accessible, nested namespace.
Every value the architecture files tag "initial/baseline/experiment" lives here, never
as a literal inside module logic (build plan, Cross-cutting rule #7).
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


class Config:
    """Dot-accessible recursive view over a plain dict.

    Supports ``cfg.a.b.c``, ``cfg["a"]["b"]``, ``cfg.get("a.b", default)`` and
    ``in`` checks. Backed by the underlying dict so it round-trips to YAML/JSON.
    """

    def __init__(self, data: dict[str, Any] | None = None):
        object.__setattr__(self, "_data", data if data is not None else {})

    # -- attribute / item access -------------------------------------------------
    def __getattr__(self, key: str) -> Any:
        try:
            value = self._data[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value) if isinstance(value, dict) else value

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"

    # -- helpers -----------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return Config(node) if isinstance(node, dict) else node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def keys(self):
        return self._data.keys()

    def items(self):
        for k, v in self._data.items():
            yield k, (Config(v) if isinstance(v, dict) else v)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path: str | os.PathLike | None = None,
                overrides: dict | None = None) -> Config:
    """Load default config, optionally merge a user YAML and a dict of overrides."""
    with open(_DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            data = _deep_merge(data, yaml.safe_load(fh) or {})
    if overrides:
        data = _deep_merge(data, overrides)
    return Config(data)
