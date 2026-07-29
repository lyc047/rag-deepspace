from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.run_stage6r_recovery_fault_attribution import (
    _adjusted_install_bits,
    _shapley,
)


def test_shapley_recovers_additive_factor_values() -> None:
    values = {
        frozenset(): 5.0,
        frozenset({"ack"}): 8.0,
        frozenset({"reset"}): 7.0,
        frozenset({"ack", "reset"}): 10.0,
    }
    result = _shapley(values, ("ack", "reset"))
    assert result["ack"] == pytest.approx(3.0)
    assert result["reset"] == pytest.approx(2.0)
    assert sum(result.values()) == pytest.approx(5.0)


def test_known_bank_replaces_every_install_attempt() -> None:
    result = SimpleNamespace(
        bit_breakdown=SimpleNamespace(
            actual_total_application_bits=2000,
            initial_install_bits=400,
            recovery_install_bits=800,
        ),
        context_install_count=3,
    )
    assert _adjusted_install_bits(
        result,
        decision="BANK:known",
        frame_bits=64,
        scenario="known_bank_only",
    ) == 992


def test_cached_ood_keeps_first_full_install() -> None:
    result = SimpleNamespace(
        bit_breakdown=SimpleNamespace(
            actual_total_application_bits=2000,
            initial_install_bits=400,
            recovery_install_bits=800,
        ),
        context_install_count=3,
    )
    assert _adjusted_install_bits(
        result,
        decision="OOD",
        frame_bits=64,
        scenario="cached_all",
    ) == 1328
    assert _adjusted_install_bits(
        result,
        decision="OOD",
        frame_bits=64,
        scenario="known_bank_only",
    ) == 2000
