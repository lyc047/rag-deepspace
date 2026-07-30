import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _read(relative: str) -> dict:
    return json.loads(
        (PROJECT_DIR / relative).read_text(encoding="utf-8")
    )


def test_stage7_development_split_is_site_disjoint_and_excludes_reserve() -> None:
    split = _read("configs/stage7_development_split_v1.json")["split"]
    registry = _read(
        "results/stage6r/electrosense_final_registry_v1/registry.json"
    )["split"]["roles"]
    flattened = [site for values in split.values() for site in values]
    expected = (
        registry["pilot"]
        + registry["stage6_final"]
        + registry["confirmation_lockbox"]
    )
    assert len(flattened) == 34
    assert len(set(flattened)) == 34
    assert set(flattened) == set(expected)
    assert set(flattened).isdisjoint(registry["reserve"])


def test_stage7_s7_0_result_passes_registered_upper_bound_gate_only() -> None:
    protocol = _read("configs/stage7_s7_0_oracle_headroom_v1.json")
    result = _read(
        "results/stage7/s7_0_oracle_headroom_v1/result.json"
    )
    assert result["development_site_count"] == 34
    assert result["reserve_access_count_before_and_after"] == 0
    assert result["decision"]["oracle_headroom_gate_passed"]
    assert result["decision"]["passing_n_count"] == 4
    assert protocol["reliability"]["oracle_is_noncausal_upper_bound"]
    assert result["governance_checks"]["external_final_claim_forbidden"]
    for row in result["n_results"].values():
        assert row["gate"]["wrong_codebook_actions_zero"]
        assert (
            row["gate"]["bit_headroom_pass"]
            or row["gate"]["clean_headroom_pass"]
        )
