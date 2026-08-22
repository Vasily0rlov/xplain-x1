"""Provenance: enough recorded that any run is reproducible from its artefacts (S-#1)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(arr).tobytes())


def sha256_config(cfg: dict[str, Any]) -> str:
    return sha256_bytes(json.dumps(cfg, sort_keys=True, default=str).encode())


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).resolve().parents[3],
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def write_provenance(run_dir: Path, *, config: dict[str, Any], data_hash: str,
                     seeds: dict[str, int], extra: dict[str, Any] | None = None) -> Path:
    record = {
        "config": config,
        "config_hash": sha256_config(config),
        "data_hash": data_hash,
        "seeds": seeds,
        "git_commit": git_commit(),
        **(extra or {}),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "provenance.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    return path
