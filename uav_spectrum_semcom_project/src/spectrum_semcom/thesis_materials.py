"""Validation rules for stage-4 thesis drafts and frozen placeholders."""

from __future__ import annotations

import re
from pathlib import Path


REQUIRED_CHAPTER_MARKERS = (
    "[[FINAL_PENDING_C1]]",
    "[[FINAL_PENDING_SELECTIVE_G2]]",
    "[[FINAL_PENDING_HOLM]]",
    "H3必须保持“不主张”",
    "Gate B未通过",
    "Gate C失败",
)

REQUIRED_FINAL_TEMPLATE_MARKERS = (
    "[[PENDING_REGISTRY_SHA256]]",
    "[[PENDING_FINAL_EXECUTABLE_SHA256]]",
    "[[MUST_EQUAL_1_AFTER_RUN]]",
    "0.0026813",
    "Holm",
)


def validate_required_markers(text: str, markers: tuple[str, ...]) -> list[str]:
    return [f"missing required marker: {marker}" for marker in markers if marker not in text]


def local_markdown_links(text: str) -> list[str]:
    links = re.findall(r"!?(?:\[[^\]]*\])\(([^)]+)\)", text)
    return [link for link in links if not re.match(r"^[a-z]+://", link, flags=re.IGNORECASE) and not link.startswith("#")]


def validate_local_links(document: str | Path) -> list[str]:
    document = Path(document)
    text = document.read_text(encoding="utf-8")
    errors = []
    for link in local_markdown_links(text):
        target = (document.parent / link).resolve()
        if not target.exists():
            errors.append(f"broken local link: {link}")
    return errors


def validate_reference_records(records: list[dict], minimum: int = 7) -> list[str]:
    """Validate the machine-readable, primary-source literature registry."""
    errors: list[str] = []
    if len(records) < minimum:
        errors.append(f"too few verified references: {len(records)}/{minimum}")
    identifiers = [str(record.get("id", "")) for record in records]
    if len(set(identifiers)) != len(identifiers) or any(not item for item in identifiers):
        errors.append("reference ids must be non-empty and unique")
    for record in records:
        ref_id = record.get("id", "<missing>")
        if not str(record.get("status", "")).startswith("verified"):
            errors.append(f"unverified reference: {ref_id}")
        if not str(record.get("url", "")).startswith("https://"):
            errors.append(f"invalid primary-source url: {ref_id}")
        if not record.get("title") or not record.get("year") or not record.get("identifier"):
            errors.append(f"incomplete reference metadata: {ref_id}")
    return errors
