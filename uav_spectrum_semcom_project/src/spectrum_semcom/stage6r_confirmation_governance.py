"""Governance checks for frozen Stage-6R ElectroSense executions."""

from __future__ import annotations

from collections.abc import Mapping


def execution_role(config: Mapping) -> str:
    role = str(config.get("data_adapter", {}).get("role", "stage6_final"))
    if role not in {"stage6_final", "confirmation_lockbox"}:
        raise ValueError(f"unsupported ElectroSense execution role: {role}")
    return role


def required_access_counts(config: Mapping) -> dict[str, int]:
    governance = config.get("governance", {})
    declared = governance.get("required_access_counts_before_run")
    if declared is None:
        return {
            "pilot": 1,
            "stage6_final": 0,
            "confirmation_lockbox": 0,
            "reserve": 0,
        }
    return {str(role): int(count) for role, count in declared.items()}


def validate_execution_preconditions(
    *,
    config: Mapping,
    registry: Mapping,
    access_state: Mapping,
    freeze: Mapping,
    protocol_sha256: str,
    code_snapshot: Mapping,
) -> tuple[str, list[str]]:
    """Validate frozen metadata without reading any registered signal values."""

    role = execution_role(config)
    governance = config.get("governance", {})
    expected_status = str(
        governance.get(
            "expected_freeze_status",
            "stage6r_electrosense_final_frozen_unaccessed",
        )
    )
    if freeze.get("status") != expected_status:
        raise ValueError("freeze artifact has invalid status")
    if freeze.get("protocol_sha256") != protocol_sha256:
        raise ValueError("protocol changed after freeze")
    if freeze.get("code_snapshot") != code_snapshot:
        raise ValueError("code snapshot changed after freeze")

    roles = access_state.get("roles", {})
    for access_role, expected_count in required_access_counts(config).items():
        if access_role not in roles:
            raise ValueError(f"access ledger is missing role: {access_role}")
        actual = int(roles[access_role].get("access_count", -1))
        if actual != expected_count:
            raise ValueError(
                f"access count mismatch for {access_role}: "
                f"expected {expected_count}, found {actual}"
            )

    registry_roles = registry.get("split", {}).get("roles", {})
    if role not in registry_roles:
        raise ValueError(f"registry is missing execution role: {role}")
    registered_sites = list(registry_roles[role])
    frozen_sites = freeze.get("sites", freeze.get("final_sites"))
    if registered_sites != frozen_sites:
        raise ValueError("registered site list changed after freeze")
    expected_site_count = int(
        config.get("data_adapter", {}).get(
            "site_count", len(registered_sites)
        )
    )
    if len(registered_sites) != expected_site_count:
        raise ValueError("registered site count differs from protocol")
    return role, registered_sites
