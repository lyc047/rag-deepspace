"""Stage-9 software-in-loop receiver lifecycle and recovery simulation.

This module is development-only.  It uses registered synthetic task traces and
does not read any protected Final or confirmation data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


POLICIES = ("fixed10", "boot_event", "no_heartbeat", "stateless_exact")


@dataclass
class ReceiverSnapshot:
    boot_id: int
    catalog_present: bool
    catalog_epoch: int | None
    active: bool
    update_epoch: int | None
    actions: tuple[int, ...] | None


class PersistentReceiver:
    """Fail-closed receiver with explicit durable and volatile state."""

    def __init__(self, state_path: str | Path | None = None) -> None:
        self.state_path = None if state_path is None else Path(state_path)
        self.snapshot = ReceiverSnapshot(
            boot_id=0,
            catalog_present=False,
            catalog_epoch=None,
            active=False,
            update_epoch=None,
            actions=None,
        )

    def _load_durable(self) -> tuple[bool, int | None]:
        if self.state_path is None or not self.state_path.exists():
            return False, None
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        return bool(payload["catalog_present"]), int(payload["catalog_epoch"])

    def _save_durable(self, catalog_epoch: int) -> None:
        if self.state_path is not None:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps(
                    {
                        "catalog_present": True,
                        "catalog_epoch": int(catalog_epoch),
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

    def boot(self, *, boot_id: int, cold: bool) -> dict[str, Any]:
        if cold and self.state_path is not None:
            self.state_path.unlink(missing_ok=True)
        if cold and self.state_path is None:
            present, epoch = False, None
        else:
            present, epoch = self._load_durable()
            if self.state_path is None:
                present = bool(self.snapshot.catalog_present)
                epoch = self.snapshot.catalog_epoch
        self.snapshot = ReceiverSnapshot(
            boot_id=int(boot_id),
            catalog_present=present,
            catalog_epoch=epoch,
            active=False,
            update_epoch=None,
            actions=None,
        )
        return self.status()

    def status(self) -> dict[str, Any]:
        return {
            "boot_id": int(self.snapshot.boot_id),
            "catalog_present": bool(self.snapshot.catalog_present),
            "catalog_epoch": self.snapshot.catalog_epoch,
            "active": bool(self.snapshot.active),
            "update_epoch": self.snapshot.update_epoch,
        }

    def restore(
        self,
        *,
        catalog_epoch: int,
        update_epoch: int,
        actions: Iterable[int],
        full_install: bool,
    ) -> dict[str, Any]:
        if full_install:
            self._save_durable(int(catalog_epoch))
            catalog_present = True
            stored_epoch = int(catalog_epoch)
        else:
            catalog_present = bool(self.snapshot.catalog_present)
            stored_epoch = self.snapshot.catalog_epoch
        accepted = bool(
            catalog_present and stored_epoch == int(catalog_epoch)
        )
        if accepted:
            self.snapshot = ReceiverSnapshot(
                boot_id=self.snapshot.boot_id,
                catalog_present=True,
                catalog_epoch=int(catalog_epoch),
                active=True,
                update_epoch=int(update_epoch),
                actions=tuple(int(value) for value in actions),
            )
        return {"accepted": accepted, **self.status()}

    def update(
        self,
        *,
        catalog_epoch: int,
        update_epoch: int,
        actions: Iterable[int],
    ) -> dict[str, Any]:
        accepted = bool(
            self.snapshot.active
            and self.snapshot.catalog_present
            and self.snapshot.catalog_epoch == int(catalog_epoch)
        )
        if accepted:
            self.snapshot.update_epoch = int(update_epoch)
            self.snapshot.actions = tuple(int(value) for value in actions)
        return {"accepted": accepted, **self.status()}

    def execute(self) -> dict[str, Any]:
        executable = bool(
            self.snapshot.active
            and self.snapshot.catalog_present
            and self.snapshot.actions is not None
        )
        return {
            "available": executable,
            "actions": (
                list(self.snapshot.actions) if executable else None
            ),
            "boot_id": int(self.snapshot.boot_id),
        }


@dataclass(frozen=True)
class FaultSchedule:
    restart: np.ndarray
    cold_restart: np.ndarray
    forward_delivered: np.ndarray
    feedback_delivered: np.ndarray


@dataclass(frozen=True)
class TaskTrace:
    desired_symbol: np.ndarray
    acceptable_symbols: tuple[frozenset[int], ...]


@dataclass(frozen=True)
class SilProtocolBits:
    compact_update_bits: int
    cumulative_ack_bits: int
    heartbeat_request_bits: int
    heartbeat_response_bits: int
    boot_status_bits: int
    preinstalled_activation_bits: int
    exact_action_bits: int
    full_install_bits: int


@dataclass(frozen=True)
class SilTrajectoryResult:
    policy: str
    total_bits: int
    forward_bits: int
    feedback_bits: int
    heartbeat_bits: int
    boot_status_bits: int
    install_bits: int
    update_bits: int
    ack_bits: int
    clean_count: int
    available_count: int
    scene_count: int
    restart_count: int
    recovery_latencies: tuple[int, ...]
    wrong_context_execution_count: int

    @property
    def bits_per_scene(self) -> float:
        return float(self.total_bits / self.scene_count)

    @property
    def clean_rate(self) -> float:
        return float(self.clean_count / self.scene_count)

    @property
    def availability(self) -> float:
        return float(self.available_count / self.scene_count)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "bits_per_scene": self.bits_per_scene,
                "clean_rate": self.clean_rate,
                "availability": self.availability,
            }
        )
        return payload


def generate_task_trace(
    *,
    scene_count: int,
    symbol_count: int,
    change_probability: float,
    equivalence_probability: float,
    equivalence_scenes: int,
    rng: np.random.Generator,
) -> TaskTrace:
    if scene_count < 2 or symbol_count < 2 or equivalence_scenes < 0:
        raise ValueError("invalid task trace parameters")
    desired = np.zeros(scene_count, dtype=np.int64)
    acceptable: list[frozenset[int]] = [frozenset({0})]
    previous = 0
    equivalence_symbol: int | None = None
    equivalence_until = -1
    for scene in range(1, scene_count):
        if rng.random() < float(change_probability):
            candidates = [value for value in range(symbol_count) if value != previous]
            desired[scene] = int(rng.choice(candidates))
            if rng.random() < float(equivalence_probability):
                equivalence_until = scene + int(equivalence_scenes) - 1
                equivalence_symbol = previous
            previous = int(desired[scene])
        else:
            desired[scene] = previous
        allowed = {int(desired[scene])}
        if scene <= equivalence_until and equivalence_symbol is not None:
            allowed.add(int(equivalence_symbol))
        acceptable.append(frozenset(allowed))
    return TaskTrace(desired_symbol=desired, acceptable_symbols=tuple(acceptable))


def generate_fault_schedule(
    *,
    scene_count: int,
    restart_probability: float,
    cold_restart_fraction: float,
    forward_loss_probability: float,
    feedback_loss_probability: float,
    burst_entry_probability: float,
    burst_exit_probability: float,
    rng: np.random.Generator,
) -> FaultSchedule:
    restart = rng.random(scene_count) < float(restart_probability)
    restart[0] = False
    cold = restart & (rng.random(scene_count) < float(cold_restart_fraction))
    burst = False
    burst_mask = np.zeros(scene_count, dtype=bool)
    for scene in range(scene_count):
        if burst:
            burst = not (rng.random() < float(burst_exit_probability))
        else:
            burst = rng.random() < float(burst_entry_probability)
        burst_mask[scene] = burst
    forward = rng.random(scene_count) >= float(forward_loss_probability)
    feedback = rng.random(scene_count) >= float(feedback_loss_probability)
    forward &= ~burst_mask
    feedback &= ~burst_mask
    return FaultSchedule(restart, cold, forward, feedback)


def simulate_sil_trajectory(
    trace: TaskTrace,
    faults: FaultSchedule,
    *,
    policy: str,
    bits: SilProtocolBits,
    heartbeat_interval: int = 10,
    catalog_epoch: int = 1,
) -> SilTrajectoryResult:
    if policy not in POLICIES:
        raise ValueError("unknown Stage-9 policy")
    scene_count = int(trace.desired_symbol.size)
    if any(
        value.size != scene_count
        for value in (
            faults.restart,
            faults.cold_restart,
            faults.forward_delivered,
            faults.feedback_delivered,
        )
    ):
        raise ValueError("trace and fault schedule lengths differ")

    receiver = PersistentReceiver()
    receiver.snapshot.catalog_present = True
    receiver.snapshot.catalog_epoch = int(catalog_epoch)
    receiver.boot(boot_id=1, cold=False)
    pending_restore = policy != "stateless_exact"
    sender_catalog_known: bool | None = True
    sender_confirmed_symbol: int | None = None
    update_epoch = 0
    last_context_confirmation = 0
    boot_id = 1
    recovery_start: int | None = 0 if pending_restore else None
    recovery_latencies: list[int] = []

    forward_bits = feedback_bits = 0
    heartbeat_bits = boot_status_bits = 0
    install_bits = update_bits = ack_bits = 0
    clean_count = available_count = wrong_context = 0

    for scene in range(scene_count):
        desired = int(trace.desired_symbol[scene])
        forward_ok = bool(faults.forward_delivered[scene])
        feedback_ok = bool(faults.feedback_delivered[scene])

        if bool(faults.restart[scene]):
            boot_id += 1
            receiver.boot(
                boot_id=boot_id,
                cold=bool(faults.cold_restart[scene]),
            )
            recovery_start = scene
            if policy == "boot_event":
                sender_catalog_known = None

        if policy == "stateless_exact":
            forward_bits += int(bits.exact_action_bits)
            update_bits += int(bits.exact_action_bits)
            if forward_ok:
                receiver.snapshot = ReceiverSnapshot(
                    boot_id=boot_id,
                    catalog_present=True,
                    catalog_epoch=catalog_epoch,
                    active=True,
                    update_epoch=scene % 256,
                    actions=(desired,),
                )
            else:
                receiver.snapshot.active = False
                receiver.snapshot.actions = None
        else:
            if policy == "boot_event" and not receiver.snapshot.active:
                feedback_bits += int(bits.boot_status_bits)
                boot_status_bits += int(bits.boot_status_bits)
                if feedback_ok:
                    pending_restore = True
                    sender_catalog_known = bool(receiver.snapshot.catalog_present)

            heartbeat_due = bool(
                policy == "fixed10"
                and scene > 0
                and scene - last_context_confirmation >= int(heartbeat_interval)
            )
            if heartbeat_due:
                forward_bits += int(bits.heartbeat_request_bits)
                heartbeat_bits += int(bits.heartbeat_request_bits)
                if forward_ok:
                    feedback_bits += int(bits.heartbeat_response_bits)
                    heartbeat_bits += int(bits.heartbeat_response_bits)
                    if feedback_ok:
                        last_context_confirmation = scene
                        sender_catalog_known = bool(receiver.snapshot.catalog_present)
                        if not receiver.snapshot.active:
                            pending_restore = True

            if pending_restore:
                full_install = sender_catalog_known is not True
                control_bits = (
                    int(bits.full_install_bits)
                    if full_install
                    else int(bits.preinstalled_activation_bits)
                )
                packet_bits = control_bits + int(bits.compact_update_bits)
                forward_bits += packet_bits
                install_bits += control_bits
                update_bits += int(bits.compact_update_bits)
                update_epoch = (update_epoch + 1) % 256
                accepted = False
                if forward_ok:
                    reply = receiver.restore(
                        catalog_epoch=catalog_epoch,
                        update_epoch=update_epoch,
                        actions=(desired,),
                        full_install=full_install,
                    )
                    accepted = bool(reply["accepted"])
                if accepted:
                    feedback_bits += int(bits.cumulative_ack_bits)
                    ack_bits += int(bits.cumulative_ack_bits)
                    if recovery_start is not None:
                        recovery_latencies.append(scene - recovery_start + 1)
                        recovery_start = None
                    if feedback_ok:
                        pending_restore = False
                        sender_confirmed_symbol = desired
                        sender_catalog_known = True
                        last_context_confirmation = scene
            elif desired != sender_confirmed_symbol:
                forward_bits += int(bits.compact_update_bits)
                update_bits += int(bits.compact_update_bits)
                update_epoch = (update_epoch + 1) % 256
                accepted = False
                if forward_ok:
                    reply = receiver.update(
                        catalog_epoch=catalog_epoch,
                        update_epoch=update_epoch,
                        actions=(desired,),
                    )
                    accepted = bool(reply["accepted"])
                if accepted:
                    feedback_bits += int(bits.cumulative_ack_bits)
                    ack_bits += int(bits.cumulative_ack_bits)
                    if feedback_ok:
                        sender_confirmed_symbol = desired
                        last_context_confirmation = scene
                else:
                    pending_restore = True
                    sender_catalog_known = None

        executed = receiver.execute()
        if bool(executed["available"]):
            available_count += 1
            symbol = int(executed["actions"][0])
            if symbol in trace.acceptable_symbols[scene]:
                clean_count += 1
        if receiver.snapshot.active and not receiver.snapshot.catalog_present:
            wrong_context += 1

    total = forward_bits + feedback_bits
    return SilTrajectoryResult(
        policy=policy,
        total_bits=total,
        forward_bits=forward_bits,
        feedback_bits=feedback_bits,
        heartbeat_bits=heartbeat_bits,
        boot_status_bits=boot_status_bits,
        install_bits=install_bits,
        update_bits=update_bits,
        ack_bits=ack_bits,
        clean_count=clean_count,
        available_count=available_count,
        scene_count=scene_count,
        restart_count=int(np.sum(faults.restart)),
        recovery_latencies=tuple(recovery_latencies),
        wrong_context_execution_count=wrong_context,
    )
