"""Public dataset loaders + encodings (S-#4).  P0 scope: zoo, wine."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import Dataset

DATA_CACHE = Path(__file__).resolve().parents[3] / "data_cache"


def load_wine() -> Dataset:
    """Wine (sklearn-bundled): 178 rows, 13 continuous, 3 classes. Near-additive."""
    from sklearn.datasets import load_wine as _lw

    raw = _lw()
    return Dataset(
        name="wine", X=raw.data, y=raw.target,
        feature_names=list(raw.feature_names), task="classification", n_classes=3,
        continuous_mask=np.ones(raw.data.shape[1], dtype=bool),
        meta={"source": "sklearn.load_wine", "expected_depth": 1})


def load_zoo() -> Dataset:
    """Zoo (OpenML 'zoo' v1): 101 rows, 15 boolean predicates + legs, 7 classes."""
    from sklearn.datasets import fetch_openml

    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    raw = fetch_openml("zoo", version=1, as_frame=True, data_home=str(DATA_CACHE),
                       parser="auto")
    df: pd.DataFrame = raw.frame.copy()
    y = raw.target.astype("category").cat.codes.to_numpy()
    df = df.drop(columns=[raw.target_names[0]] if raw.target_names else ["type"],
                 errors="ignore")
    df = df.drop(columns=[c for c in df.columns if c.lower() == "animal"],
                 errors="ignore")
    cols, arrs, cont = [], [], []
    for c in df.columns:
        s = df[c]
        if str(s.dtype) in ("category", "object", "bool"):
            codes = (s.astype("category").cat.codes if str(s.dtype) != "bool"
                     else s.astype(int))
            vals = codes.to_numpy().astype(np.float32)
            uniq = np.unique(vals)
            if len(uniq) <= 2:                     # boolean predicate -> {0,1}
                vals = (vals == uniq.max()).astype(np.float32)
                cont.append(False)
            else:                                   # small ordinal (legs)
                cont.append(True)
        else:
            vals = s.to_numpy().astype(np.float32)
            cont.append(True)                       # legs: meaningful order
        cols.append(str(c))
        arrs.append(vals)
    X = np.stack(arrs, axis=1)
    return Dataset(name="zoo", X=X, y=y, feature_names=cols,
                   task="classification", n_classes=7,
                   continuous_mask=np.array(cont),
                   meta={"source": "openml:zoo:v1", "expected_depth": 2})
