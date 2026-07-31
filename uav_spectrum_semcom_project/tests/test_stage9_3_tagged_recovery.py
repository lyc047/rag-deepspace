from __future__ import annotations

import json

import numpy as np
import pytest

from spectrum_semcom.stage6_task_codebook import (
    GreedyTaskCodebook,
    SpectrumTaskQuery,
    TaskCodeword,
)
from spectrum_semcom.stage6_task_codec import install_codebook
from spectrum_semcom.stage6r_codebook_activation import build_preinstalled_catalog
from spectrum_semcom.stage9_3_tagged_recovery import (
    RecoverySessionIdentity,
    TaggedRecoveryReceiver,
    decode_tagged_activation,
    decode_tagged_context_install,
    encode_tagged_activation,
    encode_tagged_context_install,
    recovery_session_tag,
)


def _codebook() -> GreedyTaskCodebook:
    return GreedyTaskCodebook(
        n_channels=32,
        queries=(SpectrumTaskQuery(8), SpectrumTaskQuery(16), SpectrumTaskQuery(24)),
        epsilon_db=0.2,
        codewords=tuple(
            TaskCodeword(index, (index, index, index), 0)
            for index in range(3)
        ),
        training_scene_count=0,
        exact_action_tuple_count=0,
        training_covered_count=0,
    )


def test_tagged_codecs_have_registered_widths_and_reject_stale_identity():
    codebook = _codebook()
    session = install_codebook(codebook, epoch=7)
    catalog = build_preinstalled_catalog(((2, codebook),), catalog_epoch=7)
    identity = RecoverySessionIdentity(3, 1001, 7)
    activation = encode_tagged_activation(
        catalog,
        identity=identity,
        node_id=1,
        bank_id=2,
        codebook_epoch=7,
        activation_epoch=9,
    )
    install = encode_tagged_context_install(
        session, identity=identity, node_id=1
    )
    assert activation.size == 80
    assert install.size == 419
    assert decode_tagged_activation(
        activation, catalog, expected_identity=identity
    ).session.manifest_sha256 == session.manifest_sha256
    assert decode_tagged_context_install(
        install, expected_identity=identity
    ).session.manifest_sha256 == session.manifest_sha256
    stale = RecoverySessionIdentity(3, 1000, 7)
    with pytest.raises(ValueError):
        decode_tagged_activation(activation, catalog, expected_identity=stale)
    with pytest.raises(ValueError):
        decode_tagged_context_install(install, expected_identity=stale)


def _activate(receiver: TaggedRecoveryReceiver, *, boot=1, controller=1):
    digest = "ab" * 32
    receiver.boot(boot_id=boot, cold=True)
    reply = receiver.begin_recovery(controller_epoch=controller, catalog_epoch=7)
    tag = int(reply["recovery_session_tag"])
    assert receiver.restore(
        session_tag=tag,
        catalog_epoch=7,
        update_epoch=254,
        actions=(1, 2, 3),
        full_install=True,
        catalog_digest=digest,
    )["accepted"]
    return digest, tag


def test_receiver_rechecks_session_catalog_and_serial_wrap(tmp_path):
    receiver = TaggedRecoveryReceiver(tmp_path / "state.json")
    digest, old_tag = _activate(receiver)
    assert receiver.update(
        catalog_epoch=7, update_epoch=255, actions=(2, 3, 4)
    )["accepted"]
    assert receiver.update(
        catalog_epoch=7, update_epoch=0, actions=(3, 4, 5)
    )["accepted"]
    duplicate = receiver.update(
        catalog_epoch=7, update_epoch=0, actions=(3, 4, 5)
    )
    assert duplicate["accepted"] and duplicate["duplicate"]
    assert not receiver.update(
        catalog_epoch=7, update_epoch=254, actions=(9, 9, 9)
    )["accepted"]

    receiver.boot(boot_id=2, cold=False)
    receiver.begin_recovery(controller_epoch=1, catalog_epoch=7)
    assert not receiver.restore(
        session_tag=old_tag,
        catalog_epoch=7,
        update_epoch=1,
        actions=(9, 9, 9),
        full_install=False,
        catalog_digest=digest,
    )["accepted"]
    wrong_catalog_tag = recovery_session_tag(RecoverySessionIdentity(1, 2, 8))
    assert not receiver.restore(
        session_tag=wrong_catalog_tag,
        catalog_epoch=8,
        update_epoch=1,
        actions=(9, 9, 9),
        full_install=False,
        catalog_digest=digest,
    )["accepted"]
    assert not receiver.execute()["available"]


def test_queue_lifetime_flush_and_corrupt_storage_fail_closed(tmp_path):
    state_path = tmp_path / "state.json"
    receiver = TaggedRecoveryReceiver(state_path)
    digest, _ = _activate(receiver)
    receiver.enqueue_update(
        packet_id="old", catalog_epoch=7, update_epoch=255, actions=(4, 4, 4)
    )
    receiver.advance_update_clock(count=128)
    assert not receiver.deliver_queued(packet_id="old")["accepted"]
    receiver.enqueue_update(
        packet_id="flush", catalog_epoch=7, update_epoch=255, actions=(4, 4, 4)
    )
    receiver.begin_recovery(controller_epoch=2, catalog_epoch=7)
    assert not receiver.deliver_queued(packet_id="flush")["accepted"]

    envelope = json.loads(state_path.read_text(encoding="utf-8"))
    envelope["payload"]["catalog_digest"] = "cd" * 32
    state_path.write_text(json.dumps(envelope), encoding="utf-8")
    restarted = TaggedRecoveryReceiver(state_path)
    status = restarted.boot(boot_id=2, cold=False)
    assert status["durable_corruption_detected"]
    assert not status["catalog_present"]
    assert not restarted.execute()["available"]

