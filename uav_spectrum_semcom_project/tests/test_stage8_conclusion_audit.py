from scripts.audit_stage8_conclusion import audit


def test_audit_entrypoint_is_available() -> None:
    assert callable(audit)
