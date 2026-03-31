"""Resolve model_path strings in YAML to absolute local snapshot dirs (R2R-root-relative)."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

_MODEL_BLOCKS = (
    "quick",
    "reference",
    "continuation_main",
    "continuation_reference",
    "verify",
)


def _r2r_root() -> str:
    # r2r/utils/model_paths.py -> R2R/
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def resolve_repo_model_path(p: str, *, r2r_root: Optional[str] = None) -> str:
    """If p points at an existing local dir (absolute, cwd-relative, or under R2R root), return abspath; else return p (e.g. Hub id)."""
    if not p or not isinstance(p, str):
        return p
    expanded = os.path.expanduser(p)
    if os.path.isabs(expanded) and os.path.isdir(expanded):
        return os.path.abspath(expanded)
    if os.path.isdir(expanded):
        return os.path.abspath(expanded)
    root = r2r_root or _r2r_root()
    cand = os.path.join(root, expanded)
    if os.path.isdir(cand):
        return os.path.abspath(cand)
    return p


def normalize_model_paths_in_config(cfg: Optional[Dict[str, Any]]) -> None:
    """In-place: rewrite model_path under known blocks when a matching local directory exists."""
    if not cfg:
        return
    root = _r2r_root()
    for key in _MODEL_BLOCKS:
        block = cfg.get(key)
        if isinstance(block, dict) and "model_path" in block:
            block["model_path"] = resolve_repo_model_path(block["model_path"], r2r_root=root)


def tokenizer_local_files_only(model_path: str) -> bool:
    """True when model_path is an existing directory — use with AutoTokenizer.from_pretrained(..., local_files_only=...)."""
    return os.path.isdir(os.path.expanduser(model_path))
