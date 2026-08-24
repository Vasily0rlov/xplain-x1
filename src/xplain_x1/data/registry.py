"""Dataset registry: one entry per dataset (S-#4).  P0 scope: synthetics + zoo + wine."""
from __future__ import annotations

from typing import Callable

from .dataset import Dataset
from .loaders import (load_adult, load_bike, load_drybean, load_morpher,
                      load_mushroom, load_tictactoe, load_wine, load_zoo)
from .synthetic import CONFIGS, EXTRA_CONFIGS, make_synthetic

_LOADERS: dict[str, Callable[[], Dataset]] = {
    "zoo": load_zoo,
    "wine": load_wine,
    "tictactoe": load_tictactoe,
    "mushroom": load_mushroom,
    "adult": load_adult,
    "bike": load_bike,
    "drybean": load_drybean,
    "morpher": load_morpher,
}
for _name in {**CONFIGS, **EXTRA_CONFIGS}:
    _LOADERS[_name] = (lambda n=_name: make_synthetic(n))

# MVL (S-#4): synthetics + seven public datasets, small -> large
MVL_PUBLIC = ["zoo", "tictactoe", "wine", "mushroom", "drybean", "bike", "adult"]
MVL = list(CONFIGS) + MVL_PUBLIC


def available() -> list[str]:
    return sorted(_LOADERS)


def get_dataset(name: str) -> Dataset:
    if name not in _LOADERS:
        raise KeyError(f"unknown dataset {name!r}; available: {available()}")
    return _LOADERS[name]()
