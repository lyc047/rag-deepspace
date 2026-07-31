"""Stage-9.3 tagged recovery codecs and fail-closed receiver runtime.

The 16-bit tag is deliberately limited to accidental stale-session detection.
It is not a MAC and must not be used as an authentication primitive.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from spectrum_semcom.stage6_context_codec import (
    DecodedContextInstall,
    decode_context_install,
    encode_context_install,
)
from spectrum_semcom.stage6_task_codec import CodebookSession
from spectrum_semcom.stage6r_codebook_activation import (
    PreinstalledCodebookCatalog,
    ResolvedCodebookActivation,
    decode_codebook_activation,
    encode_codebook_activation,
)


TAG_BITS = 16
MAX_PACKET_AGE_UPDATES = 127
_DOMAIN = b"S9-RCV2-v1\x00"


def _uint_to_bits(value: int, width: int) -> np.ndarray:
    if not 0 <= int(value) < 2**width:
        raise ValueError("integer does not fit tagged-recovery field")
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
class RecoverySessionIdentity:
    controller_epoch: int
    receiver_boot_id: int
    catalog_epoch: int

    def __post_init__(self) -> None:
        if (
            not 0 <= int(self.controller_epoch) <= 255
            or not 0 <= int(self.receiver_boot_id) <= 65535
            or not 0 <= int(self.catalog_epoch) <= 255
        ):
            raise ValueError("recovery identity exceeds its fixed-width schema")

    def packed(self) -> bytes:
        return bytes([int(self.controller_epoch)]) + int(
            self.receiver_boot_id
        ).to_bytes(2, "big") + bytes([int(self.catalog_epoch)])


def recovery_session_tag(identity: RecoverySessionIdentity) -> int:
    digest = hashlib.sha256(_DOMAIN + identity.packed()).digest()
    return int.from_bytes(digest[:2], "big")


def encode_tagged_activation(
    catalog: PreinstalledCodebookCatalog,
    *,
    identity: RecoverySessionIdentity,
    node_id: int,
    bank_id: int,
    codebook_epoch: int,
    activation_epoch: int,
) -> np.ndarray:
    if int(catalog.catalog_epoch) != int(identity.catalog_epoch):
        raise ValueError("activation catalog does not match recovery identity")
    base = encode_codebook_activation(
        catalog,
        node_id=node_id,
        bank_id=bank_id,
        codebook_epoch=codebook_epoch,
        activation_epoch=activation_epoch,
    )
    return np.concatenate(
        [base, _uint_to_bits(recovery_session_tag(identity), TAG_BITS)]
    )


def decode_tagged_activation(
    bits: np.ndarray,
    catalog: PreinstalledCodebookCatalog,
    *,
    expected_identity: RecoverySessionIdentity,
) -> ResolvedCodebookActivation:
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if data.size != 80 or np.any((data != 0) & (data != 1)):
        raise ValueError("invalid tagged activation bitstream")
    if int(catalog.catalog_epoch) != int(expected_identity.catalog_epoch):
        raise ValueError("receiver catalog does not match recovery identity")
    if _bits_to_uint(data[-TAG_BITS:]) != recovery_session_tag(expected_identity):
        raise ValueError("stale or mismatched recovery session tag")
    return decode_codebook_activation(data[:-TAG_BITS], catalog)


def encode_tagged_context_install(
    session: CodebookSession,
    *,
    identity: RecoverySessionIdentity,
    node_id: int,
) -> np.ndarray:
    base = encode_context_install(session, node_id=node_id)
    return np.concatenate(
        [base, _uint_to_bits(recovery_session_tag(identity), TAG_BITS)]
    )


def decode_tagged_context_install(
    bits: np.ndarray,
    *,
    expected_identity: RecoverySessionIdentity,
) -> DecodedContextInstall:
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if data.size <= TAG_BITS or np.any((data != 0) & (data != 1)):
        raise ValueError("invalid tagged context install bitstream")
    if _bits_to_uint(data[-TAG_BITS:]) != recovery_session_tag(expected_identity):
        raise ValueError("stale or mismatched recovery session tag")
    decoded = decode_context_install(data[:-TAG_BITS])
    if int(decoded.session.epoch) != int(expected_identity.catalog_epoch):
        raise ValueError("installed context epoch does not match recovery identity")
    return decoded


@dataclass
class TaggedReceiverSnapshot:
    boot_id: int = 0
    catalog_present: bool = False
    catalog_epoch: int | None = None
    catalog_digest: str | None = None
    active: bool = False
    update_epoch: int | None = None
    actions: tuple[int, ...] | None = None


@dataclass(frozen=True)
class _QueuedUpdate:
    packet_id: str
    catalog_epoch: int
    update_epoch: int
    actions: tuple[int, ...]
    enqueued_clock: int
    generation: int


class TaggedRecoveryReceiver:
    """Persistent receiver with tagged recovery and half-window serial checks."""

    def __init__(self, state_path: str | Path | None = None) -> None:
        self.state_path = None if state_path is None else Path(state_path)
        self.snapshot = TaggedReceiverSnapshot()
        self.expected_identity: RecoverySessionIdentity | None = None
        self.durable_corruption_detected = False
        self._memory_durable: dict[str, Any] | None = None
        self._queue: dict[str, _QueuedUpdate] = {}
        self._queue_generation = 0
        self._update_clock = 0

    @staticmethod
    def _canonical(payload: dict[str, Any]) -> bytes:
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")

    @classmethod
    def _envelope(cls, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "payload": payload,
            "sha256": hashlib.sha256(cls._canonical(payload)).hexdigest(),
        }

    def _read_durable(self) -> dict[str, Any] | None:
        if self.state_path is None:
            envelope = self._memory_durable
        elif self.state_path.exists():
            envelope = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            envelope = None
        if envelope is None:
            return None
        payload = envelope["payload"]
        expected = hashlib.sha256(self._canonical(payload)).hexdigest()
        if envelope.get("sha256") != expected:
            raise ValueError("durable catalog checksum mismatch")
        if (
            payload.get("catalog_present") is not True
            or not 0 <= int(payload["catalog_epoch"]) <= 255
            or len(str(payload["catalog_digest"])) != 64
        ):
            raise ValueError("invalid durable catalog payload")
        int(str(payload["catalog_digest"]), 16)
        return payload

    def _write_durable(self, catalog_epoch: int, catalog_digest: str) -> None:
        if len(str(catalog_digest)) != 64:
            raise ValueError("catalog digest must be SHA-256")
        int(str(catalog_digest), 16)
        payload = {
            "catalog_present": True,
            "catalog_epoch": int(catalog_epoch),
            "catalog_digest": str(catalog_digest).lower(),
        }
        envelope = self._envelope(payload)
        self._memory_durable = envelope
        if self.state_path is not None:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps(envelope, sort_keys=True), encoding="utf-8"
            )

    def boot(self, *, boot_id: int, cold: bool) -> dict[str, Any]:
        if not 0 <= int(boot_id) <= 65535:
            raise ValueError("boot id exceeds 16 bits")
        if cold:
            self._memory_durable = None
            if self.state_path is not None:
                self.state_path.unlink(missing_ok=True)
        self.durable_corruption_detected = False
        try:
            payload = self._read_durable()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            payload = None
            self.durable_corruption_detected = True
        self.snapshot = TaggedReceiverSnapshot(
            boot_id=int(boot_id),
            catalog_present=payload is not None,
            catalog_epoch=(None if payload is None else int(payload["catalog_epoch"])),
            catalog_digest=(None if payload is None else str(payload["catalog_digest"])),
        )
        self.expected_identity = None
        self._change_queue_generation()
        return self.status()

    def _change_queue_generation(self) -> None:
        self._queue_generation += 1
        self._queue.clear()

    def status(self) -> dict[str, Any]:
        return {
            "boot_id": int(self.snapshot.boot_id),
            "catalog_present": bool(self.snapshot.catalog_present),
            "catalog_epoch": self.snapshot.catalog_epoch,
            "catalog_digest": self.snapshot.catalog_digest,
            "active": bool(self.snapshot.active),
            "update_epoch": self.snapshot.update_epoch,
            "recovery_session_tag": (
                None
                if self.expected_identity is None
                else recovery_session_tag(self.expected_identity)
            ),
            "queue_depth": len(self._queue),
            "durable_corruption_detected": bool(
                self.durable_corruption_detected
            ),
        }

    def begin_recovery(
        self, *, controller_epoch: int, catalog_epoch: int
    ) -> dict[str, Any]:
        self.expected_identity = RecoverySessionIdentity(
            controller_epoch=int(controller_epoch),
            receiver_boot_id=int(self.snapshot.boot_id),
            catalog_epoch=int(catalog_epoch),
        )
        self.snapshot.active = False
        self.snapshot.update_epoch = None
        self.snapshot.actions = None
        self._change_queue_generation()
        return self.status()

    def restore(
        self,
        *,
        session_tag: int,
        catalog_epoch: int,
        update_epoch: int,
        actions: Iterable[int],
        full_install: bool,
        catalog_digest: str,
    ) -> dict[str, Any]:
        identity = self.expected_identity
        tag_valid = bool(
            identity is not None
            and int(identity.receiver_boot_id) == int(self.snapshot.boot_id)
            and int(identity.catalog_epoch) == int(catalog_epoch)
            and int(session_tag) == recovery_session_tag(identity)
        )
        accepted = False
        if tag_valid:
            if full_install:
                self._write_durable(int(catalog_epoch), str(catalog_digest))
                self.snapshot.catalog_present = True
                self.snapshot.catalog_epoch = int(catalog_epoch)
                self.snapshot.catalog_digest = str(catalog_digest).lower()
            catalog_valid = bool(
                self.snapshot.catalog_present
                and self.snapshot.catalog_epoch == int(catalog_epoch)
                and self.snapshot.catalog_digest == str(catalog_digest).lower()
            )
            if catalog_valid:
                self.snapshot.active = True
                self.snapshot.update_epoch = int(update_epoch) % 256
                self.snapshot.actions = tuple(int(value) for value in actions)
                accepted = True
        return {"accepted": accepted, **self.status()}

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
            and int(update_epoch) % 256 == int(self.snapshot.update_epoch)
            and values == self.snapshot.actions
        )
        fresh = bool(
            base_valid
            and self._newer_serial(
                int(update_epoch) % 256, int(self.snapshot.update_epoch)
            )
        )
        if fresh:
            self.snapshot.update_epoch = int(update_epoch) % 256
            self.snapshot.actions = values
            self._update_clock += 1
        return {
            "accepted": duplicate or fresh,
            "duplicate": duplicate,
            "fresh": fresh,
            **self.status(),
        }

    def enqueue_update(
        self,
        *,
        packet_id: str,
        catalog_epoch: int,
        update_epoch: int,
        actions: Iterable[int],
    ) -> dict[str, Any]:
        self._queue[str(packet_id)] = _QueuedUpdate(
            packet_id=str(packet_id),
            catalog_epoch=int(catalog_epoch),
            update_epoch=int(update_epoch) % 256,
            actions=tuple(int(value) for value in actions),
            enqueued_clock=int(self._update_clock),
            generation=int(self._queue_generation),
        )
        return self.status()

    def advance_update_clock(self, *, count: int = 1) -> dict[str, Any]:
        if int(count) < 0:
            raise ValueError("clock advance must be nonnegative")
        self._update_clock += int(count)
        expired = [
            packet_id
            for packet_id, packet in self._queue.items()
            if self._update_clock - packet.enqueued_clock
            > MAX_PACKET_AGE_UPDATES
        ]
        for packet_id in expired:
            del self._queue[packet_id]
        return {"expired_count": len(expired), **self.status()}

    def deliver_queued(self, *, packet_id: str) -> dict[str, Any]:
        packet = self._queue.pop(str(packet_id), None)
        if packet is None or packet.generation != self._queue_generation:
            return {"accepted": False, "queue_rejected": True, **self.status()}
        if self._update_clock - packet.enqueued_clock > MAX_PACKET_AGE_UPDATES:
            return {"accepted": False, "queue_rejected": True, **self.status()}
        return self.update(
            catalog_epoch=packet.catalog_epoch,
            update_epoch=packet.update_epoch,
            actions=packet.actions,
        )

    def execute(self) -> dict[str, Any]:
        executable = bool(
            self.snapshot.active
            and self.snapshot.catalog_present
            and self.snapshot.catalog_digest is not None
            and self.snapshot.actions is not None
            and self.expected_identity is not None
            and self.expected_identity.receiver_boot_id == self.snapshot.boot_id
            and self.expected_identity.catalog_epoch == self.snapshot.catalog_epoch
        )
        return {
            "available": executable,
            "actions": list(self.snapshot.actions) if executable else None,
            "boot_id": int(self.snapshot.boot_id),
        }

    def corrupt_durable_for_test(self) -> None:
        """Inject checksum corruption for registered fault-validation only."""

        if self.state_path is None:
            if self._memory_durable is None:
                raise ValueError("no durable catalog to corrupt")
            self._memory_durable["payload"]["catalog_epoch"] = (
                int(self._memory_durable["payload"]["catalog_epoch"]) + 1
            ) % 256
            return
        if not self.state_path.exists():
            raise ValueError("no durable catalog to corrupt")
        envelope = json.loads(self.state_path.read_text(encoding="utf-8"))
        envelope["payload"]["catalog_epoch"] = (
            int(envelope["payload"]["catalog_epoch"]) + 1
        ) % 256
        self.state_path.write_text(json.dumps(envelope), encoding="utf-8")
