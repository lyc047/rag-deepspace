"""Stage-9.6 binary endpoint protocol used by independent worker processes.

The JSONL worker envelope is test control-plane metadata.  Only the packed
``wire`` payload is counted as protocol traffic.  Recovery context frames do
not carry an action; a compact update is mandatory before execution resumes.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spectrum_semcom.stage6_context_codec import (
    decode_compact_update,
    encode_compact_update,
)
from spectrum_semcom.stage6_task_codebook import (
    GreedyTaskCodebook,
    QueryCostProfile,
    SpectrumTaskQuery,
    SpectrumTaskState,
    TaskCodeword,
)
from spectrum_semcom.stage6_task_codec import CodebookSession, install_codebook
from spectrum_semcom.stage6r_codebook_activation import (
    PreinstalledCodebookCatalog,
    build_preinstalled_catalog,
)
from spectrum_semcom.stage9_3_tagged_recovery import (
    RecoverySessionIdentity,
    TaggedRecoveryReceiver,
    decode_tagged_activation,
    decode_tagged_context_install,
    encode_tagged_activation,
    encode_tagged_context_install,
    recovery_session_tag,
)
from spectrum_semcom.stage9_5_causal_feedback import (
    ResetNack,
    decode_reset_nack,
    encode_reset_nack,
)


CATALOG_EPOCH = 7
BANK_ID = 2
NODE_ID = 1
REGISTERED_WIDTHS = {
    "reset_nack": 32,
    "activation": 80,
    "full_install": 419,
    "update": 42,
}


def protocol_codebook() -> GreedyTaskCodebook:
    queries = (
        SpectrumTaskQuery(8),
        SpectrumTaskQuery(16),
        SpectrumTaskQuery(24),
    )
    return GreedyTaskCodebook(
        n_channels=32,
        queries=queries,
        epsilon_db=0.2,
        codewords=tuple(
            TaskCodeword(index, (index, index, index), 0)
            for index in range(3)
        ),
        training_scene_count=0,
        exact_action_tuple_count=0,
        training_covered_count=0,
    )


def protocol_session() -> CodebookSession:
    return install_codebook(protocol_codebook(), epoch=CATALOG_EPOCH)


def protocol_catalog() -> PreinstalledCodebookCatalog:
    return build_preinstalled_catalog(
        ((BANK_ID, protocol_codebook()),), catalog_epoch=CATALOG_EPOCH
    )


def protocol_state(symbol: int) -> SpectrumTaskState:
    """Build a deterministic state accepted only by the selected fixture word."""

    choice = int(symbol)
    if choice not in range(3):
        raise ValueError("fixture symbol must be 0, 1, or 2")
    profiles = []
    for query in protocol_codebook().queries:
        candidate_count = 32 - int(query.demand_channels) + 1
        costs = [1.0] * candidate_count
        costs[choice] = 0.0
        profiles.append(
            QueryCostProfile(
                query=query,
                candidate_costs_dbm=tuple(costs),
                optimum_start=choice,
                optimum_cost_dbm=0.0,
                epsilon_optimal_starts=(choice,),
                runner_up_gap_db=1.0,
            )
        )
    return SpectrumTaskState(
        n_channels=32, epsilon_db=0.2, profiles=tuple(profiles)
    )


def bits_to_wire(bits: np.ndarray) -> dict[str, Any]:
    data = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if data.size < 1 or np.any((data != 0) & (data != 1)):
        raise ValueError("wire input must contain binary bits")
    packed = np.packbits(data, bitorder="big")
    return {"bit_length": int(data.size), "payload_hex": packed.tobytes().hex()}


def wire_to_bits(wire: dict[str, Any]) -> np.ndarray:
    bit_length = int(wire["bit_length"])
    raw = bytes.fromhex(str(wire["payload_hex"]))
    if bit_length < 1 or len(raw) != (bit_length + 7) // 8:
        raise ValueError("wire byte count does not match declared bit length")
    unpacked = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="big")
    if np.any(unpacked[bit_length:] != 0):
        raise ValueError("wire padding bits must be zero")
    return unpacked[:bit_length].astype(np.uint8, copy=True)


@dataclass
class SenderEndpoint:
    controller_epoch: int = 1
    last_nack: ResetNack | None = None
    identity: RecoverySessionIdentity | None = None
    feedback_transition_count: int = 0

    def status(self) -> dict[str, Any]:
        return {
            "pid": os.getpid(),
            "controller_epoch": int(self.controller_epoch),
            "last_nack": None if self.last_nack is None else asdict(self.last_nack),
            "identity": None if self.identity is None else asdict(self.identity),
            "feedback_transition_count": int(self.feedback_transition_count),
        }

    def receive_reset_nack(self, *, wire: dict[str, Any]) -> dict[str, Any]:
        message = decode_reset_nack(wire_to_bits(wire))
        self.last_nack = message
        self.identity = RecoverySessionIdentity(
            controller_epoch=int(self.controller_epoch),
            receiver_boot_id=int(message.receiver_boot_id),
            catalog_epoch=int(message.catalog_epoch),
        )
        self.feedback_transition_count += 1
        return {"accepted": True, **self.status()}

    def make_recovery(self, *, activation_epoch: int = 0) -> dict[str, Any]:
        if self.last_nack is None or self.identity is None:
            raise ValueError("RESET-NACK is required before recovery")
        if int(self.last_nack.reason) == 0:
            kind = "activation"
            bits = encode_tagged_activation(
                protocol_catalog(),
                identity=self.identity,
                node_id=NODE_ID,
                bank_id=BANK_ID,
                codebook_epoch=CATALOG_EPOCH,
                activation_epoch=int(activation_epoch) % 256,
            )
        else:
            kind = "full_install"
            bits = encode_tagged_context_install(
                protocol_session(), identity=self.identity, node_id=NODE_ID
            )
        return {"kind": kind, "wire": bits_to_wire(bits), **self.status()}

    def make_update(self, *, symbol: int, update_epoch: int) -> dict[str, Any]:
        bits, decision = encode_compact_update(
            protocol_state(symbol),
            protocol_session(),
            node_id=NODE_ID,
            update_epoch=int(update_epoch) % 256,
        )
        if decision.uses_fallback:
            raise AssertionError("registered fixture unexpectedly used fallback")
        return {
            "kind": "update",
            "symbol": int(symbol),
            "wire": bits_to_wire(bits),
            **self.status(),
        }


class ReceiverEndpoint:
    def __init__(self, state_path: str | Path) -> None:
        self.runtime = TaggedRecoveryReceiver(state_path)
        self.catalog = protocol_catalog()
        self.session: CodebookSession | None = None
        self.pending_session: CodebookSession | None = None
        self.pending_full_install = False
        self.pending_tag: int | None = None

    def boot(self, *, boot_id: int, cold: bool) -> dict[str, Any]:
        result = self.runtime.boot(boot_id=int(boot_id), cold=bool(cold))
        self.session = None
        self.pending_session = None
        self.pending_full_install = False
        self.pending_tag = None
        return {"pid": os.getpid(), **result}

    def status(self) -> dict[str, Any]:
        return {
            "pid": os.getpid(),
            "pending_context": self.pending_session is not None,
            "pending_full_install": bool(self.pending_full_install),
            **self.runtime.status(),
        }

    def execute(self) -> dict[str, Any]:
        return {"pid": os.getpid(), **self.runtime.execute()}

    def _nack_reason(self) -> int:
        if self.runtime.durable_corruption_detected:
            return 2
        if not self.runtime.snapshot.catalog_present:
            return 1
        return 0

    def _make_nack(self) -> dict[str, Any]:
        self.runtime.begin_recovery(
            controller_epoch=1, catalog_epoch=CATALOG_EPOCH
        )
        message = ResetNack(
            reason=self._nack_reason(),
            receiver_boot_id=int(self.runtime.snapshot.boot_id),
            catalog_epoch=CATALOG_EPOCH,
        )
        return bits_to_wire(encode_reset_nack(message))

    def receive_recovery(
        self, *, kind: str, wire: dict[str, Any]
    ) -> dict[str, Any]:
        identity = self.runtime.expected_identity
        if identity is None:
            return {"accepted_context": False, "error": "recovery_not_requested", **self.status()}
        try:
            bits = wire_to_bits(wire)
            if kind == "activation":
                catalog_valid = bool(
                    self.runtime.snapshot.catalog_present
                    and self.runtime.snapshot.catalog_epoch == CATALOG_EPOCH
                    and self.runtime.snapshot.catalog_digest
                    == protocol_session().manifest_sha256
                )
                if not catalog_valid:
                    raise ValueError("activation requires matching durable catalog")
                resolved = decode_tagged_activation(
                    bits, self.catalog, expected_identity=identity
                )
                session = resolved.session
                full_install = False
            elif kind == "full_install":
                decoded = decode_tagged_context_install(
                    bits, expected_identity=identity
                )
                session = decoded.session
                full_install = True
            else:
                raise ValueError("unknown recovery frame kind")
        except (KeyError, TypeError, ValueError) as exc:
            return {"accepted_context": False, "error": str(exc), **self.status()}
        self.pending_session = session
        self.pending_full_install = full_install
        self.pending_tag = recovery_session_tag(identity)
        return {"accepted_context": True, **self.status()}

    def receive_update(self, *, wire: dict[str, Any]) -> dict[str, Any]:
        try:
            bits = wire_to_bits(wire)
        except (KeyError, TypeError, ValueError) as exc:
            return {"accepted": False, "error": str(exc), **self.status()}

        if self.pending_session is not None:
            try:
                decoded = decode_compact_update(bits, self.pending_session)
            except ValueError as exc:
                return {"accepted": False, "error": str(exc), **self.status()}
            restored = self.runtime.restore(
                session_tag=int(self.pending_tag),
                catalog_epoch=CATALOG_EPOCH,
                update_epoch=int(decoded.update_epoch),
                actions=decoded.decoder_actions,
                full_install=bool(self.pending_full_install),
                catalog_digest=self.pending_session.manifest_sha256,
            )
            if restored["accepted"]:
                self.session = self.pending_session
                self.pending_session = None
                self.pending_full_install = False
                self.pending_tag = None
            return {"restored": bool(restored["accepted"]), **restored, "pid": os.getpid()}

        if not self.runtime.snapshot.active or self.session is None:
            return {
                "accepted": False,
                "feedback_kind": "reset_nack",
                "feedback_wire": self._make_nack(),
                **self.status(),
            }
        try:
            decoded = decode_compact_update(bits, self.session)
        except ValueError as exc:
            return {"accepted": False, "error": str(exc), **self.status()}
        updated = self.runtime.update(
            catalog_epoch=CATALOG_EPOCH,
            update_epoch=int(decoded.update_epoch),
            actions=decoded.decoder_actions,
        )
        return {**updated, "pid": os.getpid()}

    def corrupt_durable_for_test(self) -> dict[str, Any]:
        self.runtime.corrupt_durable_for_test()
        return self.status()

