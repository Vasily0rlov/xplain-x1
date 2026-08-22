"""Public dataset loaders + encodings (S-#4): the MVL public datasets.

Encoding rules (DATASETS.md, mandatory): continuous -> standardise; low-card
ordinal -> integer with meaningful order; binary -> {0,1}; low-cardinality
nominals -> named binary VALUE PREDICATES ("odor=foul") for values with
prevalence in [5%, 95%] (semantically distinct low-card values are legitimate
monosemantic predicates; the D-32 instability concerns interchangeable
high-cardinality indicators, which we collapse instead — e.g. country -> US?).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import Dataset

DATA_CACHE = Path(__file__).resolve().parents[3] / "data_cache"


def _fetch(name: str, version: int | str = "active"):
    from sklearn.datasets import fetch_openml

    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    return fetch_openml(name, version=version, as_frame=True,
                        data_home=str(DATA_CACHE), parser="auto")


def _predicates(s: pd.Series, col: str, lo: float = 0.05, hi: float = 0.95
                ) -> list[tuple[str, np.ndarray, bool]]:
    """Binary value-predicates for a nominal column: (name, values, continuous?)."""
    out = []
    freq = s.value_counts(normalize=True)
    for val, f in freq.items():
        if lo <= f <= hi:
            out.append((f"{col}={val}", (s == val).to_numpy().astype(np.float32),
                        False))
    return out


def _encode_frame(df: pd.DataFrame, ordinal: dict[str, dict] | None = None,
                  drop: list[str] | None = None) -> tuple[np.ndarray, list, np.ndarray]:
    """Generic encoder: continuous kept, ordinal mapped, nominal -> predicates."""
    ordinal = ordinal or {}
    drop = drop or []
    cols, arrs, cont = [], [], []
    for c in df.columns:
        if c in drop:
            continue
        s = df[c]
        if c in ordinal:
            arrs.append(s.map(ordinal[c]).to_numpy().astype(np.float32))
            cols.append(c)
            cont.append(True)
        elif str(s.dtype) in ("category", "object", "bool"):
            uniq = s.astype(str).nunique()
            if uniq <= 2:
                codes = s.astype("category").cat.codes.to_numpy().astype(np.float32)
                arrs.append((codes == codes.max()).astype(np.float32))
                cols.append(c)
                cont.append(False)
            else:
                for name, vals, is_cont in _predicates(s.astype(str), c):
                    arrs.append(vals)
                    cols.append(name)
                    cont.append(is_cont)
        else:
            arrs.append(pd.to_numeric(s, errors="coerce")
                        .fillna(s.median() if s.dtype.kind in "if" else 0)
                        .to_numpy().astype(np.float32))
            cols.append(c)
            cont.append(True)
    return np.stack(arrs, axis=1), cols, np.array(cont)


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


def load_tictactoe() -> Dataset:
    """Tic-Tac-Toe endgame: 9 cells, ternary with meaningful order x=1/b=0/o=-1
    (DATASETS: single-column ordinal), y = x-wins.  Pure composition."""
    raw = _fetch("tic-tac-toe", version=1)
    df = raw.frame.drop(columns=[raw.target.name])
    mapping = {"x": 1.0, "b": 0.0, "o": -1.0}
    X, cols, cont = _encode_frame(df, ordinal={c: mapping for c in df.columns})
    y = (raw.target.astype(str) == "positive").to_numpy().astype(np.int64)
    return Dataset(name="tictactoe", X=X, y=y, feature_names=cols,
                   task="classification", n_classes=2, continuous_mask=cont,
                   meta={"source": "openml:tic-tac-toe:v1", "expected_depth": 2,
                         "known_structure": "8 winning 3-cell lines"})


def load_mushroom() -> Dataset:
    """Mushroom: 22 low-card nominals -> value predicates; y = poisonous."""
    raw = _fetch("mushroom", version=1)
    df = raw.frame.drop(columns=[raw.target.name])
    X, cols, cont = _encode_frame(df)
    y = (raw.target.astype(str).isin(["p", "poisonous"])).to_numpy().astype(np.int64)
    return Dataset(name="mushroom", X=X, y=y, feature_names=cols,
                   task="classification", n_classes=2, continuous_mask=cont,
                   meta={"source": "openml:mushroom:v1", "expected_depth": 2,
                         "known_shortcut": "odor"})


def load_adult() -> Dataset:
    """Adult census income: mixed types; education string dropped (education-num
    keeps the ordinal); native-country collapsed to US?; y = income>50K."""
    raw = _fetch("adult", version=2)
    df = raw.frame.drop(columns=[raw.target.name])
    df = df.drop(columns=[c for c in ("education", "fnlwgt") if c in df.columns])
    if "native-country" in df.columns:
        df["native-country=US"] = (
            df["native-country"].astype(str) == "United-States")
        df = df.drop(columns=["native-country"])
    X, cols, cont = _encode_frame(df)
    y = raw.target.astype(str).str.contains(">50K").to_numpy().astype(np.int64)
    return Dataset(name="adult", X=X, y=y, feature_names=cols,
                   task="classification", n_classes=2, continuous_mask=cont,
                   meta={"source": "openml:adult:v2", "expected_depth": 1,
                         "note": "honest-shallow expected (DATASETS)"})


def load_bike() -> Dataset:
    """Bike sharing (hourly): regression on count; hour/temp are the known
    certified order-2 rung (DATASETS)."""
    raw = _fetch("Bike_Sharing_Demand", version=2)
    df = raw.frame.drop(columns=[raw.target.name])
    ordmaps = {}
    for c in df.columns:
        if str(df[c].dtype) in ("category", "object"):
            vals = df[c].astype(str)
            seasons = {"winter": 0.0, "spring": 1.0, "summer": 2.0, "fall": 3.0}
            if set(vals.unique()) <= set(seasons):
                ordmaps[c] = seasons
            elif vals.str.match(r"^-?\d+(\.\d+)?$").all():
                ordmaps[c] = {v: float(v) for v in vals.unique()}
    X, cols, cont = _encode_frame(df, ordinal=ordmaps)
    y = raw.target.to_numpy().astype(np.float32)
    return Dataset(name="bike", X=X, y=y, feature_names=cols,
                   task="regression", continuous_mask=cont,
                   meta={"source": "openml:Bike_Sharing_Demand:v2",
                         "expected_depth": 2, "known_rung": "hour x temp"})


def load_drybean() -> Dataset:
    """Dry Bean (UCI 602): 16 continuous morphological descriptors, 7 classes.
    Distributed as a zip containing an ARFF; cached locally."""
    import io
    import urllib.request
    import zipfile

    from scipy.io import arff

    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    arff_path = DATA_CACHE / "Dry_Bean_Dataset.arff"
    if not arff_path.exists():
        url = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
        blob = urllib.request.urlopen(url, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            member = next(m for m in z.namelist() if m.endswith(".arff"))
            arff_path.write_bytes(z.read(member))
    data, _meta = arff.loadarff(str(arff_path))
    df = pd.DataFrame(data)
    target = df["Class"].str.decode("utf-8")
    df = df.drop(columns=["Class"])
    X, cols, cont = _encode_frame(df)
    y = target.astype("category").cat.codes.to_numpy().astype(np.int64)
    return Dataset(name="drybean", X=X, y=y, feature_names=cols,
                   task="classification", n_classes=int(y.max()) + 1,
                   continuous_mask=cont,
                   meta={"source": "uci:602:dry-bean", "expected_depth": 2,
                         "known_rung": "Compactness x ShapeFactor1"})
