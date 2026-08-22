"""The dataset object every pipeline stage consumes (S-#4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..util.provenance import sha256_array


@dataclass
class Dataset:
    name: str
    X: np.ndarray                      # float32 [n, d], encoded but NOT standardised
    y: np.ndarray                      # int64 (classification) or float32 (regression)
    feature_names: list[str]
    task: str                          # "classification" | "regression"
    n_classes: int | None = None
    continuous_mask: np.ndarray | None = None   # bool [d]: standardise these columns
    ground_truth: dict[str, Any] | None = None  # synthetics only: planted structure
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float32)
        if self.task == "classification":
            self.y = np.asarray(self.y, dtype=np.int64)
            if self.n_classes is None:
                self.n_classes = int(self.y.max()) + 1
        else:
            self.y = np.asarray(self.y, dtype=np.float32)
        if self.continuous_mask is None:
            self.continuous_mask = np.ones(self.X.shape[1], dtype=bool)

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def d(self) -> int:
        return self.X.shape[1]

    def data_hash(self) -> str:
        return sha256_array(self.X)[:16] + "-" + sha256_array(
            self.y.astype(np.float64))[:16]
