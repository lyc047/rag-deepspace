"""Stage-9.1 hybrid boot-event and low-frequency watchdog research.

The module is isolated from the frozen Stage-9 runtime so historical hashes
remain meaningful.  It uses synthetic traces only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from spectrum_semcom.stage9_sil_runtime import (
    FaultSchedule,
    PersistentReceiver,
    ReceiverSnapshot,
    SilProtocolBits,
    SilTrajectoryResult,
    TaskTrace,
    generate_fault_schedule,
)


POLICIES = (
    "fixed10",
    "boot_event",
    "boot_event_watchdog30",
    "boot_event_watchdog60",
    "no_heartbeat",
)


@dataclass(frozen=True)
class HybridFaultSchedule:
    base: FaultSchedule
    boot_emitter_available: np.ndarray


class HardenedPersistentReceiver(PersistentReceiver):
    """Receiver that rejects stale serial numbers and wrong catalog epochs."""

    @staticmethod
    def _newer_serial(new: int, old: int) -> bool:
        distance = (int(new) - int(old)) % 256
        return 0 < distance < 128

    def update(
        self,
        *,
        catalog_epoch: int,
        update_epoch: int,
        actions: Iterable[int],
    ) -> dict[str, Any]:
        values = tuple(int(value) for value in actions)
        base_valid = bool(
            self.snapshot.active
            and self.snapshot.catalog_present
            and self.snapshot.catalog_epoch == int(catalog_epoch)
            and self.snapshot.update_epoch is not None
        )
        duplicate = bool(
            base_valid
            and int(update_epoch) == int(self.snapshot.update_epoch)
            and values == self.snapshot.actions
        )
        fresh = bool(
            base_valid
            and self._newer_serial(
                int(update_epoch), int(self.snapshot.update_epoch)
            )
        )
        accepted = duplicate or fresh
        if fresh:
            self.snapshot.update_epoch = int(update_epoch)
            self.snapshot.actions = values
        return {
            "accepted": accepted,
            "duplicate": duplicate,
            "stale_or_wrong_epoch": not accepted,
            **self.status(),
        }


def generate_hybrid_fault_schedule(
    *,
    scene_count: int,
    boot_emitter_failure_probability_per_boot: float,
    rng: np.random.Generator,
    **base_parameters: float,
) -> HybridFaultSchedule:
    base = generate_fault_schedule(
        scene_count=scene_count,
        rng=rng,
        **base_parameters,
    )
    available = np.ones(scene_count, dtype=bool)
    current = True
    for scene in range(scene_count):
        if bool(base.restart[scene]):
            current = bool(
                rng.random()
                >= float(boot_emitter_failure_probability_per_boot)
            )
        available[scene] = current
    return HybridFaultSchedule(base=base, boot_emitter_available=available)


def _watchdog_interval(policy: str) -> int | None:
    if policy == "fixed10":
        return 10
    if policy == "boot_event_watchdog30":
        return 30
    if policy == "boot_event_watchdog60":
        return 60
    return None


def simulate_hybrid_trajectory(
    trace: TaskTrace,
    faults: HybridFaultSchedule,
    *,
    policy: str,
    bits: SilProtocolBits,
    catalog_epoch: int = 1,
) -> SilTrajectoryResult:
    if policy not in POLICIES:
        raise ValueError("unknown Stage-9.1 policy")
    scene_count = int(trace.desired_symbol.size)
    if faults.boot_emitter_available.size != scene_count:
        raise ValueError("boot-emitter schedule length mismatch")

    receiver = PersistentReceiver()
    receiver.snapshot.catalog_present = True
    receiver.snapshot.catalog_epoch = int(catalog_epoch)
    receiver.boot(boot_id=1, cold=False)
    pending_restore = True
    sender_catalog_known: bool | None = True
    sender_confirmed_symbol: int | None = None
    update_epoch = 0
    last_confirmation = 0
    boot_id = 1
    recovery_start: int | None = 0
    recovery_latencies: list[int] = []

    forward_bits = feedback_bits = 0
    heartbeat_bits = boot_status_bits = 0
    install_bits = update_bits = ack_bits = 0
    clean_count = available_count = wrong_context = 0

    for scene in range(scene_count):
        desired = int(trace.desired_symbol[scene])
        forward_ok = bool(faults.base.forward_delivered[scene])
        feedback_ok = bool(faults.base.feedback_delivered[scene])

        if bool(faults.base.restart[scene]):
            boot_id += 1
            receiver.boot(
                boot_id=boot_id,
                cold=bool(faults.base.cold_restart[scene]),
            )
            recovery_start = scene
            if policy != "fixed10" and policy != "no_heartbeat":
                sender_catalog_known = None

        event_enabled = policy in {
            "boot_event",
            "boot_event_watchdog30",
            "boot_event_watchdog60",
        }
        if (
            event_enabled
            and not receiver.snapshot.active
            and bool(faults.boot_emitter_available[scene])
        ):
            feedback_bits += int(bits.boot_status_bits)
            boot_status_bits += int(bits.boot_status_bits)
            if feedback_ok:
                pending_restore = True
                sender_catalog_known = bool(receiver.snapshot.catalog_present)

        interval = _watchdog_interval(policy)
        watchdog_due = bool(
            interval is not None
            and scene > 0
            and scene - last_confirmation >= interval
        )
        if watchdog_due:
            forward_bits += int(bits.heartbeat_request_bits)
            heartbeat_bits += int(bits.heartbeat_request_bits)
            if forward_ok:
                feedback_bits += int(bits.heartbeat_response_bits)
                heartbeat_bits += int(bits.heartbeat_response_bits)
                if feedback_ok:
                    last_confirmation = scene
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
            forward_bits += control_bits + int(bits.compact_update_bits)
            install_bits += control_bits
            update_bits += int(bits.compact_update_bits)
            update_epoch = (update_epoch + 1) % 256
            accepted = False
            if forward_ok:
                accepted = bool(
                    receiver.restore(
                        catalog_epoch=catalog_epoch,
                        update_epoch=update_epoch,
                        actions=(desired,),
                        full_install=full_install,
                    )["accepted"]
                )
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
                    last_confirmation = scene
        elif desired != sender_confirmed_symbol:
            forward_bits += int(bits.compact_update_bits)
            update_bits += int(bits.compact_update_bits)
            update_epoch = (update_epoch + 1) % 256
            accepted = False
            if forward_ok:
                accepted = bool(
                    receiver.update(
                        catalog_epoch=catalog_epoch,
                        update_epoch=update_epoch,
                        actions=(desired,),
                    )["accepted"]
                )
            if accepted:
                feedback_bits += int(bits.cumulative_ack_bits)
                ack_bits += int(bits.cumulative_ack_bits)
                if feedback_ok:
                    sender_confirmed_symbol = desired
                    last_confirmation = scene
            else:
                pending_restore = True
                sender_catalog_known = None

        executed = receiver.execute()
        if bool(executed["available"]):
            available_count += 1
            if int(executed["actions"][0]) in trace.acceptable_symbols[scene]:
                clean_count += 1
        if receiver.snapshot.active and not receiver.snapshot.catalog_present:
            wrong_context += 1

    return SilTrajectoryResult(
        policy=policy,
        total_bits=forward_bits + feedback_bits,
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
        restart_count=int(np.sum(faults.base.restart)),
        recovery_latencies=tuple(recovery_latencies),
        wrong_context_execution_count=wrong_context,
    )

