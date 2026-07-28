from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


RadioMLKey = Tuple[str, int]


def load_radioml2016a(path: str | Path) -> Dict[RadioMLKey, np.ndarray]:
    path = Path(path)
    with path.open("rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


def radioml_classes(data: Dict[RadioMLKey, np.ndarray]) -> List[str]:
    return sorted({key[0] for key in data.keys()})


def radioml_snrs(data: Dict[RadioMLKey, np.ndarray]) -> List[int]:
    return sorted({int(key[1]) for key in data.keys()})


@dataclass(frozen=True)
class RadioMLArrays:
    x_train: np.ndarray
    y_train: np.ndarray
    snr_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    snr_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    snr_test: np.ndarray
    classes: List[str]


def make_radioml_split(
    data: Dict[RadioMLKey, np.ndarray],
    train_per_group: int = 200,
    val_per_group: int = 50,
    test_per_group: int = 50,
    min_snr: int | None = None,
    seed: int = 0,
) -> RadioMLArrays:
    """Create a stratified split by modulation and SNR group."""

    rng = np.random.default_rng(seed)
    classes = radioml_classes(data)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    x_train, y_train, snr_train = [], [], []
    x_val, y_val, snr_val = [], [], []
    x_test, y_test, snr_test = [], [], []

    for mod, snr in sorted(data.keys(), key=lambda item: (item[0], item[1])):
        if min_snr is not None and int(snr) < min_snr:
            continue
        arr = np.asarray(data[(mod, snr)], dtype=np.float32)
        n = arr.shape[0]
        required = train_per_group + val_per_group + test_per_group
        if required > n:
            raise ValueError(f"Requested {required} samples from group {(mod, snr)} with only {n}")
        indices = rng.permutation(n)
        train_idx = indices[:train_per_group]
        val_idx = indices[train_per_group : train_per_group + val_per_group]
        test_idx = indices[train_per_group + val_per_group : required]
        label = class_to_idx[mod]

        x_train.append(arr[train_idx])
        y_train.append(np.full(train_per_group, label, dtype=np.int64))
        snr_train.append(np.full(train_per_group, int(snr), dtype=np.int64))

        x_val.append(arr[val_idx])
        y_val.append(np.full(val_per_group, label, dtype=np.int64))
        snr_val.append(np.full(val_per_group, int(snr), dtype=np.int64))

        x_test.append(arr[test_idx])
        y_test.append(np.full(test_per_group, label, dtype=np.int64))
        snr_test.append(np.full(test_per_group, int(snr), dtype=np.int64))

    return RadioMLArrays(
        x_train=np.concatenate(x_train, axis=0),
        y_train=np.concatenate(y_train, axis=0),
        snr_train=np.concatenate(snr_train, axis=0),
        x_val=np.concatenate(x_val, axis=0),
        y_val=np.concatenate(y_val, axis=0),
        snr_val=np.concatenate(snr_val, axis=0),
        x_test=np.concatenate(x_test, axis=0),
        y_test=np.concatenate(y_test, axis=0),
        snr_test=np.concatenate(snr_test, axis=0),
        classes=classes,
    )


def raw_iq_bits_per_radioml_sample(n_iq_samples: int = 128, bits_per_i: int = 12, bits_per_q: int = 12) -> int:
    return int(n_iq_samples) * (int(bits_per_i) + int(bits_per_q))

