"""Stage-9.5 explicitly causal feedback state machine with RESET-NACK."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from spectrum_semcom.stage9_1_hybrid_recovery import HybridFaultSchedule
from spectrum_semcom.stage9_3_tagged_recovery import (
    RecoverySessionIdentity,
    TaggedRecoveryReceiver,
    recovery_session_tag,
)
from spectrum_semcom.stage9_sil_runtime import SilProtocolBits, TaskTrace


POLICIES = (
    "causal_hbw30_reset_nack",
    "causal_hbw30_no_nack",
    "causal_fixed10_reset_nack",
    "causal_boot_event_reset_nack",
)
RESET_NACK_BITS = 32
CATALOG_DIGEST = "ab" * 32
_NACK_MAGIC = 0xB
_NACK_VERSION = 1


def _uint_to_bits(value: int, width: int) -> np.ndarray:
    if not 0 <= int(value) < 2**width:
        raise ValueError("integer does not fit RESET-NACK field")
    return np.asarray(
        [(int(value) >> shift) & 1 for shift in range(width - 1, -1, -1)],
        dtype=np.uint8,
    )


def _bits_to_uint(bits: np.ndarray) -> int:
    value = 0
    for bit in np.asarray(bits, dtype=np.uint8).reshape(-1):
        value = (value << 1) | int(bit)
    return int(value)


@dataclass(frozen=True)
class ResetNack:
    reason: int
    receiver_boot_id: int
    catalog_epoch: int


def encode_reset_nack(message: ResetNack) -> np.ndarray:
    if not 0 <= int(message.reason) <= 3:
        raise ValueError("RESET-NACK reason exceeds two bits")
    return np.concatenate(
        [
            _uint_to_bits(_NACK_MAGIC, 4),
            _uint_to_bits(_NACK_VERSION, 2),
            _uint_to_bits(message.reason, 2),
            _uint_to_bits(message.receiver_boot_id, 16),
            _uint_to_bits(message.catalog_epoch, 8),
        ]
    )


def decode_reset_nack(bits: np.ndarray) -> ResetNack:
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if data.size != RESET_NACK_BITS or np.any((data != 0) & (data != 1)):
        raise ValueError("invalid RESET-NACK bitstream")
    if _bits_to_uint(data[:4]) != _NACK_MAGIC or _bits_to_uint(data[4:6]) != _NACK_VERSION:
        raise ValueError("invalid RESET-NACK identity")
    return ResetNack(
        reason=_bits_to_uint(data[6:8]),
        receiver_boot_id=_bits_to_uint(data[8:24]),
        catalog_epoch=_bits_to_uint(data[24:32]),
    )


@dataclass
class SenderBelief:
    controller_epoch: int
    catalog_epoch: int
    known_boot_id: int | None
    catalog_known_present: bool | None
    recovery_tag: int | None
    restoring: bool
    confirmed_symbol: int | None
    update_epoch: int
    last_confirmation_scene: int

    def receive_context_status(
        self,
        *,
        boot_id: int,
        catalog_present: bool,
        catalog_epoch: int,
    ) -> None:
        self.known_boot_id = int(boot_id)
        self.catalog_known_present = bool(catalog_present)
        self.catalog_epoch = int(catalog_epoch)
        self.recovery_tag = recovery_session_tag(
            RecoverySessionIdentity(
                self.controller_epoch,
                self.known_boot_id,
                self.catalog_epoch,
            )
        )
        self.restoring = True


@dataclass(frozen=True)
class CausalTrajectoryResult:
    policy: str
    total_bits: int
    forward_bits: int
    feedback_bits: int
    heartbeat_bits: int
    boot_status_bits: int
    reset_nack_bits: int
    install_bits: int
    update_bits: int
    ack_bits: int
    restore_frame_count: int
    reset_nack_frame_count: int
    feedback_transition_count: int
    sender_internal_state_read_count: int
    uncharged_feedback_transition_count: int
    clean_count: int
    available_count: int
    scene_count: int
    restart_count: int
    recovery_latencies: tuple[int, ...]
    wrong_context_execution_count: int
    stale_restore_acceptance_count: int

    @property
    def bits_per_scene(self) -> float:
        return float(self.total_bits / self.scene_count)

    @property
    def clean_rate(self) -> float:
        return float(self.clean_count / self.scene_count)

    @property
    def availability(self) -> float:
        return float(self.available_count / self.scene_count)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.update(
            {
                "bits_per_scene": self.bits_per_scene,
                "clean_rate": self.clean_rate,
                "availability": self.availability,
            }
        )
        return payload


def _policy_parameters(policy: str) -> tuple[bool, int | None, bool]:
    if policy == "causal_hbw30_reset_nack":
        return True, 30, True
    if policy == "causal_hbw30_no_nack":
        return True, 30, False
    if policy == "causal_fixed10_reset_nack":
        return False, 10, True
    if policy == "causal_boot_event_reset_nack":
        return True, None, True
    raise ValueError("unknown Stage-9.5 policy")


def _provisioned_receiver(catalog_epoch: int) -> TaggedRecoveryReceiver:
    receiver = TaggedRecoveryReceiver()
    receiver.boot(boot_id=0, cold=True)
    setup = receiver.begin_recovery(controller_epoch=1, catalog_epoch=catalog_epoch)
    receiver.restore(
        session_tag=int(setup["recovery_session_tag"]),
        catalog_epoch=catalog_epoch,
        update_epoch=0,
        actions=(0,),
        full_install=True,
        catalog_digest=CATALOG_DIGEST,
    )
    receiver.boot(boot_id=1, cold=False)
    return receiver


def _emit_receiver_status(
    receiver: TaggedRecoveryReceiver,
    *,
    controller_epoch: int,
    catalog_epoch: int,
) -> tuple[int, bool, int, int]:
    begun = receiver.begin_recovery(
        controller_epoch=controller_epoch, catalog_epoch=catalog_epoch
    )
    return (
        int(receiver.snapshot.boot_id),
        bool(receiver.snapshot.catalog_present),
        int(catalog_epoch),
        int(begun["recovery_session_tag"]),
    )


def simulate_causal_trajectory(
    trace: TaskTrace,
    faults: HybridFaultSchedule,
    *,
    policy: str,
    bits: SilProtocolBits,
    catalog_epoch: int = 1,
) -> CausalTrajectoryResult:
    async_boot, watchdog_interval, reset_nack_enabled = _policy_parameters(policy)
    scene_count = int(trace.desired_symbol.size)
    if faults.boot_emitter_available.size != scene_count:
        raise ValueError("boot-emitter schedule length mismatch")

    receiver = _provisioned_receiver(catalog_epoch)
    receiver_status = _emit_receiver_status(
        receiver, controller_epoch=1, catalog_epoch=catalog_epoch
    )
    sender = SenderBelief(
        controller_epoch=1,
        catalog_epoch=catalog_epoch,
        known_boot_id=receiver_status[0],
        catalog_known_present=receiver_status[1],
        recovery_tag=receiver_status[3],
        restoring=True,
        confirmed_symbol=None,
        update_epoch=0,
        last_confirmation_scene=0,
    )
    boot_id = 1
    recovery_start: int | None = 0
    recovery_latencies: list[int] = []

    forward_bits = feedback_bits = 0
    heartbeat_bits = boot_status_bits = reset_nack_bits = 0
    install_bits = update_bits = ack_bits = 0
    restore_frames = nack_frames = feedback_transitions = 0
    internal_reads = uncharged_transitions = 0
    clean_count = available_count = wrong_context = stale_accepts = 0

    for scene in range(scene_count):
        desired = int(trace.desired_symbol[scene])
        forward_ok = bool(faults.base.forward_delivered[scene])
        feedback_ok = bool(faults.base.feedback_delivered[scene])

        if bool(faults.base.restart[scene]):
            boot_id = (boot_id + 1) % 65536
            receiver.boot(
                boot_id=boot_id,
                cold=bool(faults.base.cold_restart[scene]),
            )
            recovery_start = scene

        if (
            async_boot
            and not receiver.snapshot.active
            and bool(faults.boot_emitter_available[scene])
        ):
            status = _emit_receiver_status(
                receiver, controller_epoch=sender.controller_epoch, catalog_epoch=catalog_epoch
            )
            feedback_bits += int(bits.boot_status_bits)
            boot_status_bits += int(bits.boot_status_bits)
            if feedback_ok:
                sender.receive_context_status(
                    boot_id=status[0],
                    catalog_present=status[1],
                    catalog_epoch=status[2],
                )
                feedback_transitions += 1

        watchdog_due = bool(
            watchdog_interval is not None
            and scene > 0
            and scene - sender.last_confirmation_scene >= watchdog_interval
        )
        if watchdog_due:
            forward_bits += int(bits.heartbeat_request_bits)
            heartbeat_bits += int(bits.heartbeat_request_bits)
            if forward_ok:
                feedback_bits += int(bits.heartbeat_response_bits)
                heartbeat_bits += int(bits.heartbeat_response_bits)
                if not receiver.snapshot.active:
                    status = _emit_receiver_status(
                        receiver,
                        controller_epoch=sender.controller_epoch,
                        catalog_epoch=catalog_epoch,
                    )
                else:
                    status = (
                        int(receiver.snapshot.boot_id),
                        bool(receiver.snapshot.catalog_present),
                        int(catalog_epoch),
                        -1,
                    )
                if feedback_ok:
                    sender.last_confirmation_scene = scene
                    if not receiver.snapshot.active:
                        sender.receive_context_status(
                            boot_id=status[0],
                            catalog_present=status[1],
                            catalog_epoch=status[2],
                        )
                    feedback_transitions += 1

        if sender.restoring and sender.recovery_tag is not None:
            full_install = sender.catalog_known_present is not True
            control_bits = (
                int(bits.full_install_bits)
                if full_install
                else int(bits.preinstalled_activation_bits)
            ) + 16
            forward_bits += control_bits + int(bits.compact_update_bits)
            install_bits += control_bits
            update_bits += int(bits.compact_update_bits)
            restore_frames += 1
            sender.update_epoch = (sender.update_epoch + 1) % 256
            if forward_ok:
                reply = receiver.restore(
                    session_tag=int(sender.recovery_tag),
                    catalog_epoch=catalog_epoch,
                    update_epoch=sender.update_epoch,
                    actions=(desired,),
                    full_install=full_install,
                    catalog_digest=CATALOG_DIGEST,
                )
                if bool(reply["accepted"]):
                    feedback_bits += int(bits.cumulative_ack_bits)
                    ack_bits += int(bits.cumulative_ack_bits)
                    if feedback_ok:
                        sender.restoring = False
                        sender.confirmed_symbol = desired
                        sender.catalog_known_present = True
                        sender.last_confirmation_scene = scene
                        feedback_transitions += 1
                        if recovery_start is not None:
                            recovery_latencies.append(scene - recovery_start + 1)
                            recovery_start = None
                elif reset_nack_enabled:
                    status = _emit_receiver_status(
                        receiver,
                        controller_epoch=sender.controller_epoch,
                        catalog_epoch=catalog_epoch,
                    )
                    encode_reset_nack(
                        ResetNack(
                            0 if status[1] else 1,
                            status[0],
                            status[2],
                        )
                    )
                    feedback_bits += RESET_NACK_BITS
                    reset_nack_bits += RESET_NACK_BITS
                    nack_frames += 1
                    if feedback_ok:
                        sender.receive_context_status(
                            boot_id=status[0],
                            catalog_present=status[1],
                            catalog_epoch=status[2],
                        )
                        feedback_transitions += 1
        elif desired != sender.confirmed_symbol:
            forward_bits += int(bits.compact_update_bits)
            update_bits += int(bits.compact_update_bits)
            sender.update_epoch = (sender.update_epoch + 1) % 256
            if forward_ok:
                reply = receiver.update(
                    catalog_epoch=catalog_epoch,
                    update_epoch=sender.update_epoch,
                    actions=(desired,),
                )
                if bool(reply["accepted"]):
                    feedback_bits += int(bits.cumulative_ack_bits)
                    ack_bits += int(bits.cumulative_ack_bits)
                    if feedback_ok:
                        sender.confirmed_symbol = desired
                        sender.last_confirmation_scene = scene
                        feedback_transitions += 1
                elif reset_nack_enabled:
                    status = _emit_receiver_status(
                        receiver,
                        controller_epoch=sender.controller_epoch,
                        catalog_epoch=catalog_epoch,
                    )
                    nack = encode_reset_nack(
                        ResetNack(
                            0 if status[1] else 1,
                            status[0],
                            status[2],
                        )
                    )
                    decoded = decode_reset_nack(nack)
                    feedback_bits += RESET_NACK_BITS
                    reset_nack_bits += RESET_NACK_BITS
                    nack_frames += 1
                    if feedback_ok:
                        sender.receive_context_status(
                            boot_id=decoded.receiver_boot_id,
                            catalog_present=decoded.reason == 0,
                            catalog_epoch=decoded.catalog_epoch,
                        )
                        feedback_transitions += 1

        executed = receiver.execute()
        if bool(executed["available"]):
            available_count += 1
            if int(executed["actions"][0]) in trace.acceptable_symbols[scene]:
                clean_count += 1
        if receiver.snapshot.active and (
            not receiver.snapshot.catalog_present
            or receiver.expected_identity is None
            or receiver.expected_identity.receiver_boot_id != receiver.snapshot.boot_id
            or receiver.expected_identity.catalog_epoch != receiver.snapshot.catalog_epoch
        ):
            wrong_context += 1

    total = forward_bits + feedback_bits
    components = (
        heartbeat_bits
        + boot_status_bits
        + reset_nack_bits
        + install_bits
        + update_bits
        + ack_bits
    )
    if total != components:
        raise RuntimeError("causal protocol bit components do not sum to total")
    return CausalTrajectoryResult(
        policy=policy,
        total_bits=total,
        forward_bits=forward_bits,
        feedback_bits=feedback_bits,
        heartbeat_bits=heartbeat_bits,
        boot_status_bits=boot_status_bits,
        reset_nack_bits=reset_nack_bits,
        install_bits=install_bits,
        update_bits=update_bits,
        ack_bits=ack_bits,
        restore_frame_count=restore_frames,
        reset_nack_frame_count=nack_frames,
        feedback_transition_count=feedback_transitions,
        sender_internal_state_read_count=internal_reads,
        uncharged_feedback_transition_count=uncharged_transitions,
        clean_count=clean_count,
        available_count=available_count,
        scene_count=scene_count,
        restart_count=int(np.sum(faults.base.restart)),
        recovery_latencies=tuple(recovery_latencies),
        wrong_context_execution_count=wrong_context,
        stale_restore_acceptance_count=stale_accepts,
    )

