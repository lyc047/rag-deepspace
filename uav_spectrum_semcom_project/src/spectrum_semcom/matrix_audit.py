"""Evidence-status helpers for the gate-conditioned stage-4 experiment matrix."""

from __future__ import annotations

from typing import Any


VALID_COVERAGE_STATUSES = {"passed", "waived", "blocked", "failed"}


def summarize_coverage(items: list[dict[str, Any]]) -> dict[str, Any]:
    for item in items:
        if item.get("status") not in VALID_COVERAGE_STATUSES:
            raise ValueError(f"invalid coverage status for {item.get('item_id')}: {item.get('status')}")
    development = [item for item in items if item.get("scope") == "development"]
    return {
        "development_complete": all(item["status"] in {"passed", "waived"} for item in development),
        "counts": {status: sum(item["status"] == status for item in items) for status in sorted(VALID_COVERAGE_STATUSES)},
        "failed_item_ids": [item["item_id"] for item in items if item["status"] == "failed"],
        "blocked_item_ids": [item["item_id"] for item in items if item["status"] == "blocked"],
    }
