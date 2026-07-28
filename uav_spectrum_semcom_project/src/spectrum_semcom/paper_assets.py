"""Claim-safety and formatting helpers for stage-4 paper assets."""

from __future__ import annotations

from typing import Any


def find_true_final_access_flags(value: Any, path: str = "root") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in {"final_holdout_accessed", "labels_or_model_outputs_accessed"} and child is True:
                findings.append(child_path)
            findings.extend(find_true_final_access_flags(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(find_true_final_access_flags(child, f"{path}[{index}]"))
    return findings


def assert_development_only(named_results: dict[str, Any], access_state: dict[str, Any]) -> None:
    if access_state.get("access_count") != 0:
        raise ValueError("paper development assets require final access_count=0")
    findings = []
    for name, value in named_results.items():
        findings.extend(find_true_final_access_flags(value, name))
    if findings:
        raise ValueError("final-access flags found in development sources: " + ", ".join(findings))


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    if not headers or any(len(row) != len(headers) for row in rows):
        raise ValueError("headers and row widths must align")
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)
