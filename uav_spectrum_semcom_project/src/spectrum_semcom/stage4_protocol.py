"""Validation helpers for the frozen stage-4 research protocol.

The registry deliberately permits a protocol-only state.  This lets algorithm
development proceed without pretending that an independent final holdout has
already been acquired.  A materialized registry is held to stricter checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .research_scope import validate_experiment_metadata
from .reproducibility import sha256_strings


STAGE4_SPLITS = ("train", "calibration", "validation", "final_holdout")
STAGE4_GRANULARITIES = ("G0", "G1", "G2", "G3")


def load_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _validate_ratios(ratios: object) -> list[str]:
    if not isinstance(ratios, Mapping):
        return ["target_split_ratios must be an object"]
    if set(ratios) != set(STAGE4_SPLITS):
        return ["target_split_ratios must contain train, calibration, validation, and final_holdout"]
    values = [ratios[name] for name in STAGE4_SPLITS]
    if any(not isinstance(value, (int, float)) or value <= 0 for value in values):
        return ["all target split ratios must be positive numbers"]
    if abs(sum(float(value) for value in values) - 1.0) > 1e-9:
        return ["target split ratios must sum to one"]
    return []


def validate_stage4_protocol(protocol: Mapping[str, Any]) -> list[str]:
    errors = validate_experiment_metadata(protocol)
    if protocol.get("split_id") != "stage4_scene_registry_v1":
        errors.append("stage4 protocol must use split_id stage4_scene_registry_v1")

    data = protocol.get("data_protocol")
    if not isinstance(data, Mapping):
        errors.append("data_protocol must be an object")
    else:
        if data.get("group_key") != "scene_id":
            errors.append("stage4 data_protocol group_key must be scene_id")
        if data.get("split_before_node_or_channel_augmentation") is not True:
            errors.append("scenes must be split before node or channel augmentation")
        if data.get("link_repeats_are_nested_within_scene") is not True:
            errors.append("link repeats must be nested within scene")
        errors.extend(_validate_ratios(data.get("target_split_ratios")))

    granularities = protocol.get("semantic_granularities")
    if not isinstance(granularities, Mapping) or set(granularities) != set(STAGE4_GRANULARITIES):
        errors.append("semantic_granularities must define exactly G0, G1, G2, and G3")
    else:
        if granularities["G0"].get("probability_bits") != [0]:
            errors.append("G0 must be a zero-bit silence action")
        for name in ("G1", "G2", "G3"):
            bits = granularities[name].get("probability_bits")
            if not isinstance(bits, list) or not bits or any(value not in {1, 2, 4, 8} for value in bits):
                errors.append(f"{name} probability_bits must be a non-empty subset of 1,2,4,8")

    final = protocol.get("final_holdout")
    if not isinstance(final, Mapping):
        errors.append("final_holdout must be an object")
    else:
        if final.get("access_count") != 0:
            errors.append("initial stage4 final_holdout access_count must be zero")
        if final.get("retuning_after_access") is not False:
            errors.append("retuning_after_access must be false")

    final_evaluation = protocol.get("final_evaluation_protocol")
    if not isinstance(final_evaluation, Mapping):
        errors.append("final_evaluation_protocol must be preregistered")
    else:
        if final_evaluation.get("statistics_engine") != "stage4_final_statistics_v1":
            errors.append("final statistics engine must be stage4_final_statistics_v1")
        if final_evaluation.get("scene_level_schema_version") != "stage4_final_scene_metrics_v1":
            errors.append("final scene-level schema version is not frozen")
        if final_evaluation.get("bootstrap_repetitions") != 10000:
            errors.append("final bootstrap repetitions must equal 10000")
        if final_evaluation.get("cvar_alpha") != 0.9:
            errors.append("final CVaR alpha must equal 0.9")
        if not isinstance(final_evaluation.get("bootstrap_seed"), int):
            errors.append("final bootstrap seed must be an integer")

    gate_a = protocol.get("gate_a")
    if not isinstance(gate_a, Mapping):
        errors.append("gate_a must be preregistered")
    else:
        if gate_a.get("selection_split") != "validation":
            errors.append("Gate A selection must use validation")
        if gate_a.get("evaluation_split") != "final_holdout":
            errors.append("Gate A final evaluation must use final_holdout")

    value = protocol.get("value_prediction_protocol")
    if not isinstance(value, Mapping):
        errors.append("value_prediction_protocol must be preregistered")
    else:
        seeds = value.get("training_seeds")
        if not isinstance(seeds, list) or len(seeds) < 5 or len(set(seeds)) != len(seeds):
            errors.append("value prediction requires at least five unique training seeds")
        prohibited = value.get("prohibited_inputs", [])
        if not {"truth_occupancy", "candidate_high_precision_report"}.issubset(set(prohibited)):
            errors.append("value prediction must prohibit truth and candidate high-precision reports")
        if value.get("checkpoint_rule") != "fixed_final_epoch_no_validation_checkpoint_selection":
            errors.append("value prediction checkpoint selection must not tune on validation")

    preview = protocol.get("preview_identifiability_diagnostic")
    if not isinstance(preview, Mapping):
        errors.append("preview_identifiability_diagnostic must be preregistered")
    elif preview.get("control_probability_bits") != 1 or preview.get("treatment_probability_bits") != 2:
        errors.append("preview diagnostic must compare G1 1-bit control with 2-bit treatment")
    return errors


def validate_stage4_registry(registry: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if registry.get("split_id") != "stage4_scene_registry_v1":
        errors.append("stage4 registry has an unexpected split_id")
    if registry.get("group_key") != "scene_id":
        errors.append("stage4 registry group_key must be scene_id")
    errors.extend(_validate_ratios(registry.get("target_split_ratios")))

    splits = registry.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(STAGE4_SPLITS):
        errors.append("registry splits must define exactly the four stage4 splits")
        return errors

    owner: dict[str, str] = {}
    for split_name in STAGE4_SPLITS:
        entry = splits[split_name]
        if not isinstance(entry, Mapping):
            errors.append(f"{split_name} split entry must be an object")
            continue
        scene_ids = entry.get("scene_ids")
        if not isinstance(scene_ids, list) or any(not isinstance(scene_id, str) or not scene_id for scene_id in scene_ids):
            errors.append(f"{split_name} scene_ids must be a list of non-empty strings")
            continue
        if len(scene_ids) != len(set(scene_ids)):
            errors.append(f"{split_name} contains duplicate scene_ids")
        if entry.get("scene_count") != len(scene_ids):
            errors.append(f"{split_name} scene_count does not match scene_ids")
        for scene_id in scene_ids:
            previous = owner.setdefault(scene_id, split_name)
            if previous != split_name:
                errors.append(f"scene leakage between {previous} and {split_name}: {scene_id}")
        declared_hash = entry.get("scene_ids_sha256")
        if scene_ids and declared_hash != sha256_strings(scene_ids):
            errors.append(f"{split_name} scene_ids_sha256 does not match")
        if not scene_ids and declared_hash is not None:
            errors.append(f"{split_name} empty split must use a null scene_ids_sha256")

    final = registry.get("final_holdout")
    if not isinstance(final, Mapping):
        errors.append("registry final_holdout must be an object")
    else:
        if final.get("access_count") != 0:
            errors.append("unfrozen registry must have final holdout access_count zero")
        if final.get("labels_or_model_outputs_accessed") is not False:
            errors.append("final holdout labels or outputs must remain unaccessed")

    if registry.get("status") == "materialized":
        for split_name in STAGE4_SPLITS:
            if not splits[split_name].get("scene_ids"):
                errors.append(f"materialized registry requires non-empty {split_name}")
    return errors


def assert_valid_stage4_protocol(protocol: Mapping[str, Any]) -> None:
    errors = validate_stage4_protocol(protocol)
    if errors:
        raise ValueError("invalid stage4 protocol: " + "; ".join(errors))


def assert_valid_stage4_registry(registry: Mapping[str, Any]) -> None:
    errors = validate_stage4_registry(registry)
    if errors:
        raise ValueError("invalid stage4 registry: " + "; ".join(errors))
