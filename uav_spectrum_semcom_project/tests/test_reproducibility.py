from dataclasses import dataclass

import pytest

from spectrum_semcom.reproducibility import assert_disjoint_groups, grouped_split, split_group_ids


@dataclass(frozen=True)
class Item:
    item_id: str
    sequence_id: str
    channel_variant: int


def make_items() -> list[Item]:
    return [
        Item(item_id=f"{sequence}_{variant}", sequence_id=sequence, channel_variant=variant)
        for sequence in [f"seq_{idx:02d}" for idx in range(20)]
        for variant in range(4)
    ]


def test_grouped_split_keeps_all_channel_variants_together() -> None:
    splits = grouped_split(make_items(), lambda item: item.sequence_id, seed=2026)
    assert_disjoint_groups(splits, lambda item: item.sequence_id)

    group_ids = split_group_ids(splits, lambda item: item.sequence_id)
    assert {len(ids) for ids in group_ids.values()} == {3, 14}
    for sequence in {item.sequence_id for item in make_items()}:
        owning_splits = [name for name, ids in group_ids.items() if sequence in ids]
        assert len(owning_splits) == 1
        assert sum(item.sequence_id == sequence for item in splits[owning_splits[0]]) == 4


def test_grouped_split_is_deterministic() -> None:
    first = grouped_split(make_items(), lambda item: item.sequence_id, seed=19)
    second = grouped_split(make_items(), lambda item: item.sequence_id, seed=19)
    assert first == second


def test_disjoint_group_check_rejects_leakage() -> None:
    item = Item("a", "same_sequence", 0)
    with pytest.raises(ValueError, match="source-group leakage"):
        assert_disjoint_groups({"train": [item], "test": [item]}, lambda value: value.sequence_id)
