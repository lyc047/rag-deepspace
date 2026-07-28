from spectrum_semcom.stage4_acceptance import AcceptanceItem, acceptance_markdown


def test_acceptance_item_rejects_unknown_status() -> None:
    try:
        AcceptanceItem("x", "development", "unknown", [], "")
    except ValueError as exc:
        assert "invalid acceptance status" in str(exc)
    else:
        raise AssertionError("invalid status was accepted")


def test_acceptance_markdown_contains_statuses() -> None:
    result = {
        "development_acceptance": True,
        "final_acceptance": False,
        "items": [{"item_id": "A", "scope": "development", "status": "passed", "note": "ok"}],
        "claim_boundary": "not final",
    }
    text = acceptance_markdown(result)
    assert "Development acceptance: `True`" in text
    assert "| A | development | passed | ok |" in text
