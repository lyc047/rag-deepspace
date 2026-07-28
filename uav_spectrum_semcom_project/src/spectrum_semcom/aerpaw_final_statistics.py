"""Pre-registered C1-only final branch when AERPAW time alignment gate M1 fails."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .final_statistics import (
    FAMILY_METHODS,
    _bootstrap_difference,
    _cvar_randomization_p,
    _mean_randomization_p,
    _paired_arrays,
    _summary,
    holm_adjust,
)


AERPAW_C1_FINAL_METRICS_SCHEMA = "aerpaw_c1_only_final_scene_metrics_v1"


def validate_aerpaw_c1_metrics(payload: Mapping[str, Any], minimum_scenes: int = 200) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != AERPAW_C1_FINAL_METRICS_SCHEMA:
        errors.append(f"schema_version must be {AERPAW_C1_FINAL_METRICS_SCHEMA}")
    for field in ("registry_sha256", "access_receipt_sha256"):
        value = payload.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
            errors.append(f"{field} must be a SHA-256 string")
    families = payload.get("families")
    if not isinstance(families, Mapping) or set(families) != {"C1_resource_loss"}:
        return errors + ["families must contain exactly C1_resource_loss in the M1-failed branch"]
    rows = families["C1_resource_loss"].get("rows") if isinstance(families["C1_resource_loss"], Mapping) else None
    if not isinstance(rows, list):
        return errors + ["C1_resource_loss.rows must be a list"]
    methods = FAMILY_METHODS["C1_resource_loss"]
    owners: set[tuple[str, str]] = set()
    method_scenes = {method: set() for method in methods}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"C1_resource_loss.rows[{index}] must be an object")
            continue
        scene_id, method = row.get("scene_id"), row.get("method")
        if not isinstance(scene_id, str) or not scene_id:
            errors.append(f"C1_resource_loss.rows[{index}].scene_id must be non-empty")
            continue
        if method not in methods:
            errors.append(f"C1_resource_loss.rows[{index}].method is not preregistered: {method}")
            continue
        key = (scene_id, method)
        if key in owners:
            errors.append(f"duplicate C1 scene-method row: {scene_id}/{method}")
        owners.add(key)
        method_scenes[method].add(scene_id)
        for metric in ("regret", "actual_bits"):
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                errors.append(f"C1_resource_loss.rows[{index}].{metric} must be finite and non-negative")
    if any(len(values) < minimum_scenes for values in method_scenes.values()):
        errors.append(f"C1_resource_loss requires at least {minimum_scenes} scenes per method")
    if method_scenes[methods[0]] != method_scenes[methods[1]]:
        errors.append("C1 method scene sets do not match")
    return errors


def evaluate_aerpaw_c1_only_statistics(
    payload: Mapping[str, Any], base_protocol: Mapping[str, Any], aerpaw_protocol: Mapping[str, Any]
) -> dict[str, Any]:
    branch = aerpaw_protocol["M1_time_alignment_decision"]
    if branch.get("fail_statistics_engine") != "aerpaw_c1_only_final_statistics_v1":
        raise ValueError("AERPAW C1-only branch is not pre-registered")
    final = base_protocol["final_evaluation_protocol"]
    minimum = int(final["minimum_independent_scene_count"])
    errors = validate_aerpaw_c1_metrics(payload, minimum)
    if errors:
        raise ValueError("invalid AERPAW C1-only metrics payload: " + "; ".join(errors))
    rows = payload["families"]["C1_resource_loss"]["rows"]
    methods = FAMILY_METHODS["C1_resource_loss"]
    scene_ids, regret_p, regret_b = _paired_arrays(rows, *methods, "regret")
    _, bits_p, bits_b = _paired_arrays(rows, *methods, "actual_bits")
    repetitions = int(final["bootstrap_repetitions"])
    cvar_alpha = float(final["cvar_alpha"])
    rng = np.random.default_rng(int(final["bootstrap_seed"]))
    indices = rng.integers(0, len(scene_ids), size=(repetitions, len(scene_ids)))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(repetitions, len(scene_ids)))
    swaps = rng.integers(0, 2, size=(repetitions, len(scene_ids)), dtype=np.int8).astype(bool)

    observed, samples = _bootstrap_difference(regret_p, regret_b, indices, "mean", cvar_alpha)
    regret = _summary(observed, samples, 0.0, _mean_randomization_p(regret_p, regret_b, 0.0, signs))
    observed, samples = _bootstrap_difference(regret_p, regret_b, indices, "cvar", cvar_alpha)
    cvar = _summary(observed, samples, 0.0, _cvar_randomization_p(regret_p, regret_b, observed, swaps, cvar_alpha))
    observed, samples = _bootstrap_difference(bits_p, bits_b, indices, "mean", cvar_alpha)
    bits = _summary(observed, samples, 0.0, _mean_randomization_p(bits_p, bits_b, 0.0, signs))

    raw_family_p = max(
        regret["one_sided_paired_randomization_p"],
        cvar["one_sided_paired_randomization_p"],
    )
    adjusted = holm_adjust({"C1_resource_loss": raw_family_p})["C1_resource_loss"]
    alpha = float(final["familywise_alpha"])
    requirements = {
        "regret_ci_upper_below_0": regret["ci95"][1] < 0,
        "cvar_ci_upper_below_0": cvar["ci95"][1] < 0,
        "bit_ci_contains_0": bits["ci95"][0] <= 0 <= bits["ci95"][1],
        "single_family_adjusted_p_at_most_alpha": adjusted <= alpha,
    }
    return {
        "analysis_id": "aerpaw_c1_only_final_statistics_v1",
        "schema_version": AERPAW_C1_FINAL_METRICS_SCHEMA,
        "registry_sha256": payload["registry_sha256"],
        "access_receipt_sha256": payload["access_receipt_sha256"],
        "M1_status": "failed_before_model_execution",
        "scene_count": len(scene_ids),
        "statistical_unit": "independent_scene",
        "c1": {"regret": regret, "cvar_0_9_regret": cvar, "actual_bits": bits, "requirements": requirements},
        "family_p_values": {"C1_resource_loss": {"raw": raw_family_p, "holm_adjusted": adjusted}},
        "family_decision": all(requirements.values()),
        "claim_boundary": "C1 real-power resource proxy only. Selective G2 was not evaluated because M1 failed; no asynchronous fusion is permitted.",
    }
