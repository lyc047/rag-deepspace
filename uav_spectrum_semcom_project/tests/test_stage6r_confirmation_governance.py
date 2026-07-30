import pytest

from spectrum_semcom.stage6r_confirmation_governance import (
    execution_role,
    validate_execution_preconditions,
)


def _fixtures():
    config = {
        "data_adapter": {
            "role": "confirmation_lockbox",
            "site_count": 2,
        },
        "governance": {
            "expected_freeze_status": "frozen",
            "required_access_counts_before_run": {
                "pilot": 1,
                "stage6_final": 1,
                "confirmation_lockbox": 0,
                "reserve": 0,
            },
        },
    }
    registry = {
        "split": {
            "roles": {
                "confirmation_lockbox": ["a", "b"],
            }
        }
    }
    access = {
        "roles": {
            "pilot": {"access_count": 1},
            "stage6_final": {"access_count": 1},
            "confirmation_lockbox": {"access_count": 0},
            "reserve": {"access_count": 0},
        }
    }
    snapshot = {"combined_sha256": "snapshot"}
    freeze = {
        "status": "frozen",
        "protocol_sha256": "protocol",
        "code_snapshot": snapshot,
        "sites": ["a", "b"],
    }
    return config, registry, access, freeze, snapshot


def test_confirmation_preconditions_accept_frozen_unread_role():
    config, registry, access, freeze, snapshot = _fixtures()
    role, sites = validate_execution_preconditions(
        config=config,
        registry=registry,
        access_state=access,
        freeze=freeze,
        protocol_sha256="protocol",
        code_snapshot=snapshot,
    )
    assert role == "confirmation_lockbox"
    assert sites == ["a", "b"]


def test_confirmation_preconditions_reject_consumed_lockbox():
    config, registry, access, freeze, snapshot = _fixtures()
    access["roles"]["confirmation_lockbox"]["access_count"] = 1
    with pytest.raises(ValueError, match="access count mismatch"):
        validate_execution_preconditions(
            config=config,
            registry=registry,
            access_state=access,
            freeze=freeze,
            protocol_sha256="protocol",
            code_snapshot=snapshot,
        )


def test_confirmation_preconditions_reject_changed_sites_or_snapshot():
    config, registry, access, freeze, snapshot = _fixtures()
    freeze["sites"] = ["b", "a"]
    with pytest.raises(ValueError, match="site list changed"):
        validate_execution_preconditions(
            config=config,
            registry=registry,
            access_state=access,
            freeze=freeze,
            protocol_sha256="protocol",
            code_snapshot=snapshot,
        )
    freeze["sites"] = ["a", "b"]
    with pytest.raises(ValueError, match="code snapshot changed"):
        validate_execution_preconditions(
            config=config,
            registry=registry,
            access_state=access,
            freeze=freeze,
            protocol_sha256="protocol",
            code_snapshot={"combined_sha256": "changed"},
        )


def test_execution_role_rejects_unregistered_roles():
    assert execution_role({"data_adapter": {}}) == "stage6_final"
    with pytest.raises(ValueError, match="unsupported"):
        execution_role({"data_adapter": {"role": "reserve"}})
