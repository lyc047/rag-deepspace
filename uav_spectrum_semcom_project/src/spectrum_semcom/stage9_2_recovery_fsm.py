"""Bounded recovery-FSM safety checker for Stage-9.2."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class SessionIdentity:
    controller_epoch: int
    receiver_boot_id: int
    catalog_epoch: int


@dataclass(frozen=True)
class ProtocolDesign:
    name: str
    restore_tag_bits: int
    update_tag_bits: int
    ack_tag_bits: int


@dataclass(frozen=True)
class RecoveryState:
    sender_controller_epoch: int = 0
    sender_phase: str = "UNKNOWN"
    sender_known_boot_id: int | None = None
    sender_session: SessionIdentity | None = None
    sender_serial: int = 0
    sender_pending_serial: int | None = None
    sender_confirmed_serial: int | None = None
    receiver_boot_id: int = 0
    receiver_catalog_valid: bool = True
    receiver_catalog_epoch: int = 1
    receiver_active: bool = False
    receiver_session: SessionIdentity | None = None
    receiver_serial: int | None = None
    action_origin: SessionIdentity | None = None


@dataclass(frozen=True)
class ModelCheckResult:
    design: str
    allow_ancient_collision: bool
    depth: int
    explored_state_count: int
    explored_transition_count: int
    counterexample_found: bool
    counterexample_path: tuple[str, ...]
    violated_invariant: str | None
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


EVENTS = (
    "receiver_warm_restart",
    "receiver_cold_restart",
    "sender_restart",
    "current_boot_status",
    "current_restore",
    "stale_restore",
    "current_update",
    "duplicate_update",
    "stale_update",
    "wrong_catalog_update",
    "current_ack",
    "stale_ack",
    "ancient_update_collision",
    "ancient_ack_collision",
    "catalog_corruption",
)


def _session(state: RecoveryState) -> SessionIdentity:
    return SessionIdentity(
        state.sender_controller_epoch,
        state.receiver_boot_id,
        state.receiver_catalog_epoch,
    )


def _serial_newer(new: int, old: int, modulus: int) -> bool:
    distance = (int(new) - int(old)) % int(modulus)
    return 0 < distance < int(modulus) // 2


def invariant_violation(state: RecoveryState) -> str | None:
    if state.receiver_active and not state.receiver_catalog_valid:
        return "active_receiver_has_invalid_catalog"
    if state.receiver_active and (
        state.receiver_session is None
        or state.action_origin is None
        or state.receiver_serial is None
    ):
        return "active_receiver_has_incomplete_context"
    if state.receiver_active and state.action_origin != state.receiver_session:
        return "executable_action_originates_from_stale_session"
    if state.receiver_active and (
        state.receiver_session.receiver_boot_id != state.receiver_boot_id
        or state.receiver_session.catalog_epoch != state.receiver_catalog_epoch
    ):
        return "active_receiver_session_identity_mismatch"
    if state.sender_phase == "READY" and (
        state.sender_session is None
        or state.sender_confirmed_serial is None
    ):
        return "ready_sender_has_no_confirmed_context"
    if (
        state.sender_phase == "READY"
        and state.receiver_active
        and state.sender_session != state.receiver_session
    ):
        return "sender_ready_for_different_receiver_session"
    return None


def apply_event(
    state: RecoveryState,
    event: str,
    *,
    design: ProtocolDesign,
    serial_modulus: int,
    allow_ancient_collision: bool,
) -> RecoveryState:
    if event not in EVENTS:
        raise ValueError("unknown recovery event")
    if event == "receiver_warm_restart":
        if state.receiver_boot_id >= 2:
            return state
        return replace(
            state,
            receiver_boot_id=state.receiver_boot_id + 1,
            receiver_active=False,
            receiver_session=None,
            receiver_serial=None,
            action_origin=None,
        )
    if event == "receiver_cold_restart":
        if state.receiver_boot_id >= 2:
            return state
        return replace(
            state,
            receiver_boot_id=state.receiver_boot_id + 1,
            receiver_catalog_valid=False,
            receiver_active=False,
            receiver_session=None,
            receiver_serial=None,
            action_origin=None,
        )
    if event == "sender_restart":
        if state.sender_controller_epoch >= 2:
            return state
        return replace(
            state,
            sender_controller_epoch=state.sender_controller_epoch + 1,
            sender_phase="UNKNOWN",
            sender_known_boot_id=None,
            sender_session=None,
            sender_pending_serial=None,
            sender_confirmed_serial=None,
        )
    if event == "current_boot_status":
        session = _session(state)
        return replace(
            state,
            sender_phase="RESTORING",
            sender_known_boot_id=state.receiver_boot_id,
            sender_session=session,
            sender_pending_serial=None,
            sender_confirmed_serial=None,
        )
    if event == "current_restore":
        if state.sender_phase != "RESTORING" or state.sender_session is None:
            return state
        if design.restore_tag_bits > 0 and (
            state.sender_session.receiver_boot_id != state.receiver_boot_id
            or state.sender_session.catalog_epoch != state.receiver_catalog_epoch
        ):
            return state
        serial = (state.sender_serial + 1) % serial_modulus
        return replace(
            state,
            sender_phase="WAIT_ACK",
            sender_serial=serial,
            sender_pending_serial=serial,
            receiver_catalog_valid=True,
            receiver_active=True,
            receiver_session=state.sender_session,
            receiver_serial=serial,
            action_origin=state.sender_session,
        )
    if event == "stale_restore":
        stale_boot = max(0, state.receiver_boot_id - 1)
        stale = SessionIdentity(
            state.sender_controller_epoch,
            stale_boot,
            state.receiver_catalog_epoch,
        )
        if design.restore_tag_bits > 0:
            return state
        serial = (state.sender_serial + 1) % serial_modulus
        return replace(
            state,
            receiver_catalog_valid=True,
            receiver_active=True,
            receiver_session=_session(state),
            receiver_serial=serial,
            action_origin=stale,
        )
    if event == "current_update":
        if state.sender_phase != "READY" or not state.receiver_active:
            return state
        serial = (state.sender_serial + 1) % serial_modulus
        current = state.sender_session
        if current is None or state.receiver_serial is None:
            return state
        if design.update_tag_bits > 0 and current != state.receiver_session:
            return state
        if not _serial_newer(serial, state.receiver_serial, serial_modulus):
            return state
        return replace(
            state,
            sender_phase="WAIT_ACK",
            sender_serial=serial,
            sender_pending_serial=serial,
            receiver_serial=serial,
            action_origin=state.receiver_session,
        )
    if event == "duplicate_update":
        return state
    if event in {"stale_update", "wrong_catalog_update"}:
        return state
    if event == "ancient_update_collision":
        if (
            not allow_ancient_collision
            or not state.receiver_active
            or state.receiver_serial is None
            or design.update_tag_bits > 0
        ):
            return state
        serial = (state.receiver_serial + 1) % serial_modulus
        stale = SessionIdentity(
            max(0, state.sender_controller_epoch - 1),
            max(0, state.receiver_boot_id - 1),
            state.receiver_catalog_epoch,
        )
        return replace(
            state,
            receiver_serial=serial,
            action_origin=stale,
        )
    if event == "current_ack":
        if (
            state.sender_phase != "WAIT_ACK"
            or state.sender_pending_serial is None
            or not state.receiver_active
            or state.receiver_serial != state.sender_pending_serial
        ):
            return state
        if design.ack_tag_bits > 0 and state.receiver_session != state.sender_session:
            return state
        return replace(
            state,
            sender_phase="READY",
            sender_confirmed_serial=state.sender_pending_serial,
            sender_pending_serial=None,
        )
    if event == "stale_ack":
        return state
    if event == "ancient_ack_collision":
        if (
            not allow_ancient_collision
            or state.sender_phase != "WAIT_ACK"
            or state.sender_pending_serial is None
            or design.ack_tag_bits > 0
        ):
            return state
        return replace(
            state,
            sender_phase="READY",
            sender_confirmed_serial=state.sender_pending_serial,
            sender_pending_serial=None,
        )
    if event == "catalog_corruption":
        return replace(
            state,
            receiver_catalog_valid=False,
            receiver_active=False,
            receiver_session=None,
            receiver_serial=None,
            action_origin=None,
        )
    return state


def bounded_model_check(
    design: ProtocolDesign,
    *,
    depth: int,
    serial_bits: int,
    allow_ancient_collision: bool,
    maximum_states: int,
) -> ModelCheckResult:
    if depth < 1 or serial_bits < 2 or maximum_states < 1:
        raise ValueError("invalid model-check bounds")
    modulus = 2**int(serial_bits)
    initial = RecoveryState()
    queue = deque([(initial, tuple())])
    visited = {initial: 0}
    transitions = 0
    truncated = False
    while queue:
        state, path = queue.popleft()
        violation = invariant_violation(state)
        if violation is not None:
            return ModelCheckResult(
                design.name,
                allow_ancient_collision,
                depth,
                len(visited),
                transitions,
                True,
                path,
                violation,
                truncated,
            )
        if len(path) >= depth:
            continue
        for event in EVENTS:
            next_state = apply_event(
                state,
                event,
                design=design,
                serial_modulus=modulus,
                allow_ancient_collision=allow_ancient_collision,
            )
            transitions += 1
            if next_state == state:
                continue
            next_path = (*path, event)
            violation = invariant_violation(next_state)
            if violation is not None:
                return ModelCheckResult(
                    design.name,
                    allow_ancient_collision,
                    depth,
                    len(visited),
                    transitions,
                    True,
                    next_path,
                    violation,
                    truncated,
                )
            if next_state not in visited or visited[next_state] > len(next_path):
                if len(visited) >= maximum_states:
                    truncated = True
                    continue
                visited[next_state] = len(next_path)
                queue.append((next_state, next_path))
    return ModelCheckResult(
        design.name,
        allow_ancient_collision,
        depth,
        len(visited),
        transitions,
        False,
        tuple(),
        None,
        truncated,
    )


def protocol_identity_costs(
    *,
    scene_count: int,
    ordinary_updates: int,
    restore_count: int,
    compact_update_bits: int,
    activation_bits: int,
    ack_bits: int,
) -> dict[str, dict[str, float]]:
    updates = int(ordinary_updates)
    restores = int(restore_count)
    acknowledgements = updates + restores
    legacy = (
        updates * compact_update_bits
        + restores * (activation_bits + compact_update_bits)
        + acknowledgements * ack_bits
    )
    restore_tag = legacy + 16 * restores
    full_tag = legacy + 32 * (updates + restores)
    return {
        "legacy_epoch_only": {
            "total_bits": float(legacy),
            "bits_per_scene": float(legacy / scene_count),
            "incremental_bits_per_scene": 0.0,
        },
        "restore_tag16_bounded": {
            "total_bits": float(restore_tag),
            "bits_per_scene": float(restore_tag / scene_count),
            "incremental_bits_per_scene": float((restore_tag - legacy) / scene_count),
        },
        "full_tag16": {
            "total_bits": float(full_tag),
            "bits_per_scene": float(full_tag / scene_count),
            "incremental_bits_per_scene": float((full_tag - legacy) / scene_count),
        },
    }
