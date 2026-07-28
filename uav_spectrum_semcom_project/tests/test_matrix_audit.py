import pytest

from spectrum_semcom.matrix_audit import summarize_coverage


def test_waived_items_complete_development_but_blocked_final_does_not() -> None:
    result = summarize_coverage([
        {"item_id": "a", "scope": "development", "status": "passed"},
        {"item_id": "b", "scope": "development", "status": "waived"},
        {"item_id": "f", "scope": "final", "status": "blocked"},
    ])
    assert result["development_complete"] is True
    assert result["blocked_item_ids"] == ["f"]


def test_unknown_matrix_status_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid coverage status"):
        summarize_coverage([{"item_id": "x", "scope": "development", "status": "partial"}])
