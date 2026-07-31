from spectrum_semcom.stage9_2_recovery_fsm import (
    ProtocolDesign,
    _serial_newer,
    bounded_model_check,
    protocol_identity_costs,
)


LEGACY = ProtocolDesign("legacy_epoch_only", 0, 0, 0)
RESTORE = ProtocolDesign("restore_tag16_bounded", 16, 0, 0)
FULL = ProtocolDesign("full_tag16", 16, 16, 16)


def _check(design, ancient):
    return bounded_model_check(
        design,
        depth=6,
        serial_bits=3,
        allow_ancient_collision=ancient,
        maximum_states=500_000,
    )


def test_modular_serial_ordering_wraps_with_half_window():
    assert _serial_newer(0, 7, 8)
    assert _serial_newer(3, 0, 8)
    assert not _serial_newer(4, 0, 8)
    assert not _serial_newer(7, 0, 8)


def test_registered_model_check_pattern():
    assert _check(LEGACY, True).counterexample_found
    assert not _check(RESTORE, False).counterexample_found
    assert _check(RESTORE, True).counterexample_found
    assert not _check(FULL, True).counterexample_found


def test_restore_tag_cost_stays_below_one_bit_per_scene():
    for restores in (5, 20, 50):
        costs = protocol_identity_costs(
            scene_count=1000,
            ordinary_updates=80,
            restore_count=restores,
            compact_update_bits=42,
            activation_bits=64,
            ack_bits=24,
        )
        assert (
            costs["restore_tag16_bounded"]["incremental_bits_per_scene"]
            < 1.0
        )
        assert (
            costs["full_tag16"]["incremental_bits_per_scene"]
            > costs["restore_tag16_bounded"]["incremental_bits_per_scene"]
        )
