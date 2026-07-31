"""Stage-9.4 end-to-end integration of HBW30 and tagged recovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from spectrum_semcom.stage9_1_hybrid_recovery import HybridFaultSchedule
from spectrum_semcom.stage9_3_tagged_recovery import TaggedRecoveryReceiver
from spectrum_semcom.stage9_sil_runtime import SilProtocolBits, TaskTrace


CATALOG_DIGEST = "ab" * 32


@dataclass(frozen=True)
class TaggedHybridTrajectoryResult:
    policy: str
    total_bits: int
    forward_bits: int
    feedback_bits: int
    heartbeat_bits: int
    boot_status_bits: int
    install_bits: int
    update_bits: int
    ack_bits: int
    recovery_tag_bits: int
    restore_frame_count: int
    identity_sync_wait_scenes: int
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


def _provisioned_receiver(
    *, controller_epoch: int, catalog_epoch: int
) -> TaggedRecoveryReceiver:
    """Represent a factory-preinstalled catalog without charging link bits."""

    receiver = TaggedRecoveryReceiver()
    receiver.boot(boot_id=0, cold=True)
    setup = receiver.begin_recovery(
        controller_epoch=controller_epoch, catalog_epoch=catalog_epoch
    )
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


def simulate_tagged_hbw30_trajectory(
    trace: TaskTrace,
    faults: HybridFaultSchedule,
    *,
    bits: SilProtocolBits,
    restore_tag_bits: int = 16,
    catalog_epoch: int = 1,
    controller_epoch: int = 1,
    watchdog_interval: int = 30,
) -> TaggedHybridTrajectoryResult:
    """Run HBW30 with explicit boot-identity synchronization.

    ``restore_tag_bits=0`` is the registered zero-tag accounting shadow.  It
    still executes the real tag checks; only link accounting omits the tag.
    """

    if restore_tag_bits not in {0, 16}:
        raise ValueError("Stage-9.4 registers only zero- and 16-bit tag accounting")
    scene_count = int(trace.desired_symbol.size)
    if faults.boot_emitter_available.size != scene_count:
        raise ValueError("boot-emitter schedule length mismatch")

    receiver = _provisioned_receiver(
        controller_epoch=controller_epoch, catalog_epoch=catalog_epoch
    )
    initial = receiver.begin_recovery(
        controller_epoch=controller_epoch, catalog_epoch=catalog_epoch
    )
    current_session_tag: int | None = int(initial["recovery_session_tag"])
    identity_ready = True
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
    tag_bits_total = restore_frames = identity_wait = 0
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
            identity_ready = False
            current_session_tag = None
            sender_catalog_known = None

        if (
            not receiver.snapshot.active
            and bool(faults.boot_emitter_available[scene])
        ):
            feedback_bits += int(bits.boot_status_bits)
            boot_status_bits += int(bits.boot_status_bits)
            if feedback_ok:
                pending_restore = True
                sender_catalog_known = bool(receiver.snapshot.catalog_present)
                begun = receiver.begin_recovery(
                    controller_epoch=controller_epoch,
                    catalog_epoch=catalog_epoch,
                )
                current_session_tag = int(begun["recovery_session_tag"])
                identity_ready = True

        watchdog_due = bool(
            scene > 0
            and scene - last_confirmation >= int(watchdog_interval)
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
                        begun = receiver.begin_recovery(
                            controller_epoch=controller_epoch,
                            catalog_epoch=catalog_epoch,
                        )
                        current_session_tag = int(
                            begun["recovery_session_tag"]
                        )
                        identity_ready = True

        if pending_restore and identity_ready:
            full_install = sender_catalog_known is not True
            base_control_bits = (
                int(bits.full_install_bits)
                if full_install
                else int(bits.preinstalled_activation_bits)
            )
            control_bits = base_control_bits + int(restore_tag_bits)
            forward_bits += control_bits + int(bits.compact_update_bits)
            install_bits += control_bits
            update_bits += int(bits.compact_update_bits)
            tag_bits_total += int(restore_tag_bits)
            restore_frames += 1
            update_epoch = (update_epoch + 1) % 256
            accepted = False
            if forward_ok:
                accepted = bool(
                    receiver.restore(
                        session_tag=int(current_session_tag),
                        catalog_epoch=catalog_epoch,
                        update_epoch=update_epoch,
                        actions=(desired,),
                        full_install=full_install,
                        catalog_digest=CATALOG_DIGEST,
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
        elif pending_restore:
            identity_wait += 1
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
                # A lost ordinary update does not imply a receiver reboot.
                # Keep the current recovery identity and cached executable
                # action until an actual boot/status transition is observed.

        executed = receiver.execute()
        if bool(executed["available"]):
            available_count += 1
            if int(executed["actions"][0]) in trace.acceptable_symbols[scene]:
                clean_count += 1
        if receiver.snapshot.active and (
            not receiver.snapshot.catalog_present
            or receiver.expected_identity is None
            or receiver.expected_identity.receiver_boot_id
            != receiver.snapshot.boot_id
            or receiver.expected_identity.catalog_epoch
            != receiver.snapshot.catalog_epoch
        ):
            wrong_context += 1

    return TaggedHybridTrajectoryResult(
        policy="tagged_hbw30" if restore_tag_bits else "zero_tag_shadow",
        total_bits=forward_bits + feedback_bits,
        forward_bits=forward_bits,
        feedback_bits=feedback_bits,
        heartbeat_bits=heartbeat_bits,
        boot_status_bits=boot_status_bits,
        install_bits=install_bits,
        update_bits=update_bits,
        ack_bits=ack_bits,
        recovery_tag_bits=tag_bits_total,
        restore_frame_count=restore_frames,
        identity_sync_wait_scenes=identity_wait,
        clean_count=clean_count,
        available_count=available_count,
        scene_count=scene_count,
        restart_count=int(np.sum(faults.base.restart)),
        recovery_latencies=tuple(recovery_latencies),
        wrong_context_execution_count=wrong_context,
        stale_restore_acceptance_count=stale_accepts,
    )
