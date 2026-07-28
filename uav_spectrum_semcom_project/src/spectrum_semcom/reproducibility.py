from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypeVar


T = TypeVar("T")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def grouped_split(
    items: Sequence[T],
    group_key: Callable[[T], str],
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, list[T]]:
    """Deterministically split complete source groups before derived samples exist."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 <= val_ratio < 1.0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("val_ratio must be non-negative and train_ratio + val_ratio must be below 1")

    groups: dict[str, list[T]] = defaultdict(list)
    for item in items:
        groups[str(group_key(item))].append(item)
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)

    n_groups = len(group_ids)
    n_train = int(n_groups * train_ratio)
    n_val = int(n_groups * val_ratio)
    split_group_ids = {
        "train": group_ids[:n_train],
        "val": group_ids[n_train : n_train + n_val],
        "test": group_ids[n_train + n_val :],
    }
    return {
        split: [item for group_id in ids for item in groups[group_id]]
        for split, ids in split_group_ids.items()
    }


def split_group_ids(splits: dict[str, Sequence[T]], group_key: Callable[[T], str]) -> dict[str, set[str]]:
    return {split: {str(group_key(item)) for item in items} for split, items in splits.items()}


def assert_disjoint_groups(splits: dict[str, Sequence[T]], group_key: Callable[[T], str]) -> None:
    ids = split_group_ids(splits, group_key)
    names = sorted(ids)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = ids[left] & ids[right]
            if overlap:
                preview = ", ".join(sorted(overlap)[:5])
                raise ValueError(f"source-group leakage between {left} and {right}: {preview}")


def environment_snapshot(distributions: Iterable[str]) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
    }
