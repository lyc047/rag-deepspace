import pytest

from spectrum_semcom.paper_assets import assert_development_only, find_true_final_access_flags, markdown_table


def test_claim_guard_finds_nested_final_access() -> None:
    value = {"nested": [{"final_holdout_accessed": False}, {"labels_or_model_outputs_accessed": True}]}
    assert find_true_final_access_flags(value) == ["root.nested[1].labels_or_model_outputs_accessed"]
    with pytest.raises(ValueError, match="final-access flags"):
        assert_development_only({"x": value}, {"access_count": 0})


def test_claim_guard_and_markdown_table() -> None:
    assert_development_only({"x": {"final_holdout_accessed": False}}, {"access_count": 0})
    assert "| A | B |" in markdown_table(["A", "B"], [[1, 2]])
    with pytest.raises(ValueError, match="row widths"):
        markdown_table(["A", "B"], [[1]])
