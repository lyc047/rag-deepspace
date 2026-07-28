from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SCOPE_PATH = PROJECT_DIR / "configs" / "research_scope.json"


def load_research_scope(path: str | Path = DEFAULT_SCOPE_PATH) -> dict[str, Any]:
    """Load the canonical thesis scope used by formal experiments."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_experiment_metadata(
    metadata: Mapping[str, Any],
    scope: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return validation errors for a formal thesis experiment record."""
    scope = scope or load_research_scope()
    errors: list[str] = []

    required = set(scope["required_experiment_metadata"])
    missing = sorted(key for key in required if key not in metadata)
    if missing:
        errors.append(f"missing required metadata: {', '.join(missing)}")

    valid_hypotheses = {item["id"] for item in scope["hypotheses"]}
    hypothesis_ids = metadata.get("hypothesis_ids", [])
    if not isinstance(hypothesis_ids, list) or not hypothesis_ids:
        errors.append("hypothesis_ids must be a non-empty list")
    else:
        unknown = sorted(set(hypothesis_ids) - valid_hypotheses)
        if unknown:
            errors.append(f"unknown hypothesis ids: {', '.join(unknown)}")

    repeat_count = metadata.get("repeat_count")
    if repeat_count is not None and (not isinstance(repeat_count, int) or repeat_count < 1):
        errors.append("repeat_count must be a positive integer")

    budget_type = metadata.get("fairness_budget_type")
    valid_budget_types = {"bits", "latency", "energy", "time_frequency_resource", "matched_task_performance"}
    if budget_type is not None and budget_type not in valid_budget_types:
        errors.append(f"unsupported fairness_budget_type: {budget_type}")

    metrics = metadata.get("metrics")
    if metrics is not None and not isinstance(metrics, Mapping):
        errors.append("metrics must be an object")

    return errors


def assert_valid_experiment_metadata(
    metadata: Mapping[str, Any],
    scope: Mapping[str, Any] | None = None,
) -> None:
    """Raise ValueError when a formal experiment record violates stage-0 scope."""
    errors = validate_experiment_metadata(metadata, scope)
    if errors:
        raise ValueError("invalid experiment metadata: " + "; ".join(errors))
