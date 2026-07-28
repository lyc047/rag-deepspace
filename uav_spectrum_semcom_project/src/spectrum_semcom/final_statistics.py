"""Frozen scene-level statistical decision engine for stage-4 final claims."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np


FINAL_METRICS_SCHEMA = "stage4_final_scene_metrics_v1"
FAMILY_METHODS = {
    "C1_resource_loss": ("detection_plus_resource", "detection_only"),
    "selective_G2_digital_reporting": ("selective_G2", "all_G2"),
}


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def validate_final_metrics_payload(payload: Mapping[str, Any], minimum_scenes: int = 200) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != FINAL_METRICS_SCHEMA:
        errors.append(f"schema_version must be {FINAL_METRICS_SCHEMA}")
    for field in ("registry_sha256", "access_receipt_sha256"):
        value = payload.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
            errors.append(f"{field} must be a SHA-256 string")
    families = payload.get("families")
    if not isinstance(families, Mapping) or set(families) != set(FAMILY_METHODS):
        return errors + ["families must contain exactly the two preregistered primary families"]

    family_scene_sets: dict[str, set[str]] = {}
    for family, methods in FAMILY_METHODS.items():
        entry = families[family]
        rows = entry.get("rows") if isinstance(entry, Mapping) else None
        if not isinstance(rows, list):
            errors.append(f"{family}.rows must be a list")
            continue
        owners: dict[tuple[str, str], int] = {}
        method_scenes = {method: set() for method in methods}
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                errors.append(f"{family}.rows[{index}] must be an object")
                continue
            scene_id, method = row.get("scene_id"), row.get("method")
            if not isinstance(scene_id, str) or not scene_id:
                errors.append(f"{family}.rows[{index}].scene_id must be non-empty")
                continue
            if method not in methods:
                errors.append(f"{family}.rows[{index}].method is not preregistered: {method}")
                continue
            key = (scene_id, method)
            if key in owners:
                errors.append(f"duplicate scene-method row in {family}: {scene_id}/{method}")
            owners[key] = index
            method_scenes[method].add(scene_id)
            for metric in ("regret", "actual_bits"):
                value = row.get(metric)
                if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                    errors.append(f"{family}.rows[{index}].{metric} must be finite and non-negative")
        if any(len(scenes) < minimum_scenes for scenes in method_scenes.values()):
            errors.append(f"{family} requires at least {minimum_scenes} scenes per method")
        if method_scenes[methods[0]] != method_scenes[methods[1]]:
            errors.append(f"{family} method scene sets do not match")
        family_scene_sets[family] = method_scenes[methods[0]] & method_scenes[methods[1]]
    if len(family_scene_sets) == 2 and len(set.intersection(*family_scene_sets.values())) != len(set.union(*family_scene_sets.values())):
        errors.append("the two primary families must use exactly the same independent scene set")
    return errors


def validate_access_binding(payload: Mapping[str, Any], access_state: Mapping[str, Any]) -> list[str]:
    errors = []
    if access_state.get("status") != "access_consumed" or access_state.get("access_count") != 1:
        errors.append("final access state must be access_consumed with count 1")
        return errors
    receipt = access_state.get("receipt")
    if not isinstance(receipt, Mapping):
        return ["final access receipt is missing"]
    if payload.get("registry_sha256") != receipt.get("registry_sha256"):
        errors.append("payload registry SHA does not match access receipt")
    if payload.get("access_receipt_sha256") != canonical_sha256(receipt):
        errors.append("payload access receipt SHA does not match state receipt")
    return errors


def _paired_arrays(rows: list[Mapping[str, Any]], proposed: str, baseline: str, metric: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    by_method = {proposed: {}, baseline: {}}
    for row in rows:
        if row["method"] in by_method:
            by_method[row["method"]][row["scene_id"]] = float(row[metric])
    scene_ids = sorted(set(by_method[proposed]) & set(by_method[baseline]))
    return scene_ids, np.asarray([by_method[proposed][x] for x in scene_ids]), np.asarray([by_method[baseline][x] for x in scene_ids])


def _cvar(values: np.ndarray, alpha: float) -> float:
    count = max(1, int(np.ceil((1.0 - alpha) * values.size)))
    return float(np.mean(np.partition(values, values.size - count)[-count:]))


def _bootstrap_difference(proposed: np.ndarray, baseline: np.ndarray, indices: np.ndarray, statistic: str, cvar_alpha: float) -> tuple[float, np.ndarray]:
    if statistic == "mean":
        observed = float(np.mean(proposed - baseline))
        samples = np.mean(proposed[indices] - baseline[indices], axis=1)
    elif statistic == "cvar":
        observed = _cvar(proposed, cvar_alpha) - _cvar(baseline, cvar_alpha)
        count = max(1, int(np.ceil((1.0 - cvar_alpha) * proposed.size)))
        left = np.partition(proposed[indices], proposed.size - count, axis=1)[:, -count:].mean(axis=1)
        right = np.partition(baseline[indices], baseline.size - count, axis=1)[:, -count:].mean(axis=1)
        samples = left - right
    else:
        raise ValueError(f"unsupported statistic: {statistic}")
    return observed, samples


def _summary(observed: float, samples: np.ndarray, threshold: float, randomization_p: float) -> dict[str, Any]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "difference": observed,
        "ci95": [float(low), float(high)],
        "one_sided_paired_randomization_p": randomization_p,
        "threshold": threshold,
        "bootstrap_repetitions": int(samples.size),
    }


def _mean_randomization_p(proposed: np.ndarray, baseline: np.ndarray, threshold: float, signs: np.ndarray) -> float:
    centered = proposed - baseline - threshold
    observed = float(np.mean(centered))
    permuted = np.mean(signs * centered[None, :], axis=1)
    return (1.0 + float(np.count_nonzero(permuted <= observed))) / (permuted.size + 1.0)


def _cvar_randomization_p(proposed: np.ndarray, baseline: np.ndarray, observed: float, swaps: np.ndarray, alpha: float) -> float:
    left = np.where(swaps, baseline[None, :], proposed[None, :])
    right = np.where(swaps, proposed[None, :], baseline[None, :])
    count = max(1, int(np.ceil((1.0 - alpha) * proposed.size)))
    difference = np.partition(left, proposed.size - count, axis=1)[:, -count:].mean(axis=1) - np.partition(right, baseline.size - count, axis=1)[:, -count:].mean(axis=1)
    return (1.0 + float(np.count_nonzero(difference <= observed))) / (difference.size + 1.0)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: p_values[key])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, key in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * float(p_values[key])))
        adjusted[key] = running
    return adjusted


def evaluate_final_statistics(payload: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    final = protocol["final_evaluation_protocol"]
    minimum = int(final["minimum_independent_scene_count"])
    errors = validate_final_metrics_payload(payload, minimum)
    if errors:
        raise ValueError("invalid final metrics payload: " + "; ".join(errors))
    repetitions = int(final["bootstrap_repetitions"])
    seed = int(final["bootstrap_seed"])
    cvar_alpha = float(final["cvar_alpha"])
    scene_count = len(payload["families"]["C1_resource_loss"]["rows"]) // 2
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, scene_count, size=(repetitions, scene_count))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(repetitions, scene_count))
    swaps = rng.integers(0, 2, size=(repetitions, scene_count), dtype=np.int8).astype(bool)

    c1_rows = payload["families"]["C1_resource_loss"]["rows"]
    scene_ids, c1_regret_p, c1_regret_b = _paired_arrays(c1_rows, *FAMILY_METHODS["C1_resource_loss"], "regret")
    _, c1_bits_p, c1_bits_b = _paired_arrays(c1_rows, *FAMILY_METHODS["C1_resource_loss"], "actual_bits")
    observed, samples = _bootstrap_difference(c1_regret_p, c1_regret_b, indices, "mean", cvar_alpha)
    c1_regret = _summary(observed, samples, 0.0, _mean_randomization_p(c1_regret_p, c1_regret_b, 0.0, signs))
    observed, samples = _bootstrap_difference(c1_regret_p, c1_regret_b, indices, "cvar", cvar_alpha)
    c1_cvar = _summary(observed, samples, 0.0, _cvar_randomization_p(c1_regret_p, c1_regret_b, observed, swaps, cvar_alpha))
    observed, samples = _bootstrap_difference(c1_bits_p, c1_bits_b, indices, "mean", cvar_alpha)
    c1_bits = _summary(observed, samples, 0.0, _mean_randomization_p(c1_bits_p, c1_bits_b, 0.0, signs))

    selective_rows = payload["families"]["selective_G2_digital_reporting"]["rows"]
    _, selective_regret_p, selective_regret_b = _paired_arrays(selective_rows, *FAMILY_METHODS["selective_G2_digital_reporting"], "regret")
    _, selective_bits_p, selective_bits_b = _paired_arrays(selective_rows, *FAMILY_METHODS["selective_G2_digital_reporting"], "actual_bits")
    margin = float(final["primary_claims"]["selective_G2_digital_reporting"]["regret_noninferiority_margin"])
    observed, samples = _bootstrap_difference(selective_regret_p, selective_regret_b, indices, "mean", cvar_alpha)
    selective_regret = _summary(observed, samples, margin, _mean_randomization_p(selective_regret_p, selective_regret_b, margin, signs))
    observed, samples = _bootstrap_difference(selective_bits_p, selective_bits_b, indices, "mean", cvar_alpha)
    selective_bits = _summary(observed, samples, 0.0, _mean_randomization_p(selective_bits_p, selective_bits_b, 0.0, signs))

    raw_family_p = {
        "C1_resource_loss": max(c1_regret["one_sided_paired_randomization_p"], c1_cvar["one_sided_paired_randomization_p"]),
        "selective_G2_digital_reporting": max(selective_regret["one_sided_paired_randomization_p"], selective_bits["one_sided_paired_randomization_p"]),
    }
    adjusted = holm_adjust(raw_family_p)
    alpha = float(final["familywise_alpha"])
    c1_requirements = {
        "regret_ci_upper_below_0": c1_regret["ci95"][1] < 0,
        "cvar_ci_upper_below_0": c1_cvar["ci95"][1] < 0,
        "bit_ci_contains_0": c1_bits["ci95"][0] <= 0 <= c1_bits["ci95"][1],
        "holm_adjusted_family_p_at_most_alpha": adjusted["C1_resource_loss"] <= alpha,
    }
    selective_requirements = {
        "regret_ci_upper_below_margin": selective_regret["ci95"][1] < margin,
        "actual_bit_ci_upper_below_0": selective_bits["ci95"][1] < 0,
        "holm_adjusted_family_p_at_most_alpha": adjusted["selective_G2_digital_reporting"] <= alpha,
    }
    decisions = {
        "C1_resource_loss": all(c1_requirements.values()),
        "selective_G2_digital_reporting": all(selective_requirements.values()),
    }
    return {
        "analysis_id": "stage4_final_statistics_v1",
        "schema_version": FINAL_METRICS_SCHEMA,
        "registry_sha256": payload["registry_sha256"],
        "access_receipt_sha256": payload["access_receipt_sha256"],
        "scene_count": len(scene_ids),
        "statistical_unit": "independent_scene",
        "c1": {"regret": c1_regret, "cvar_0_9_regret": c1_cvar, "actual_bits": c1_bits, "requirements": c1_requirements},
        "selective_g2": {"regret": selective_regret, "actual_bits": selective_bits, "requirements": selective_requirements},
        "family_p_values": {key: {"raw": raw_family_p[key], "holm_adjusted": adjusted[key]} for key in raw_family_p},
        "family_decisions": decisions,
        "any_primary_family_passed": any(decisions.values()),
        "all_primary_families_passed": all(decisions.values()),
        "claim_boundary": "H3 is not evaluated by this engine. A failed family cannot be rescued by secondary metrics.",
    }
