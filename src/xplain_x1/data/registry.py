"""Dataset registry: one entry per dataset (S-#4).  P0 scope: synthetics + zoo + wine."""
from __future__ import annotations

from typing import Callable

from .dataset import Dataset
from .loaders import load_wine, load_zoo
from .synthetic import CONFIGS, make_synthetic

_LOADERS: dict[str, Callable[[], Dataset]] = {
    "zoo": load_zoo,
    "wine": load_wine,
}
for _name in CONFIGS:
    _LOADERS[_name] = (lambda n=_name: make_synthetic(n))

# MVL build order (S-#4); extended tier appended after MVL bars are met
MVL = list(CONFIGS) + ["zoo", "wine"]  # P0 subset; grows in P5 with the rest of the MVL


def available() -> list[str]:
    return sorted(_LOADERS)


def get_dataset(name: str) -> Dataset:
    if name not in _LOADERS:
        raise KeyError(f"unknown dataset {name!r}; available: {available()}")
    return _LOADERS[name]()
