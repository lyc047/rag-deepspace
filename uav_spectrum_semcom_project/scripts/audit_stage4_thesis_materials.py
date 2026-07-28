#!/usr/bin/env python
"""Audit generated paper assets, claim boundaries, and final placeholders."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.thesis_materials import REQUIRED_CHAPTER_MARKERS, REQUIRED_FINAL_TEMPLATE_MARKERS, validate_local_links, validate_reference_records, validate_required_markers


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_DIR)
    parser.add_argument("--output", type=Path, default=Path("results/stage4/thesis_materials_audit_v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    chapter = root / "docs/STAGE4_THESIS_CHAPTER_DRAFT.md"
    claims = root / "docs/STAGE4_CLAIM_MATRIX.md"
    final_template = root / "docs/STAGE4_FINAL_RESULTS_TEMPLATE.md"
    related_work = root / "docs/STAGE4_RELATED_WORK_MATRIX.md"
    bibliography = root / "docs/references/stage4_verified_refs.bib"
    literature_registry = root / "results/stage4/literature_audit_v1/verified_references.json"
    manifest_path = root / "results/stage4/paper_assets_v1/paper_assets_manifest.json"
    access = json.loads((root / "configs/stage4_final_access_state.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    items = []
    chapter_errors = validate_required_markers(chapter.read_text(encoding="utf-8"), REQUIRED_CHAPTER_MARKERS) + validate_local_links(chapter)
    items.append({"item_id": "T01_CHAPTER_DRAFT", "status": "passed" if not chapter_errors else "failed", "errors": chapter_errors, "evidence": str(chapter.relative_to(root)).replace("\\", "/")})
    template_errors = validate_required_markers(final_template.read_text(encoding="utf-8"), REQUIRED_FINAL_TEMPLATE_MARKERS) + validate_local_links(final_template)
    items.append({"item_id": "T02_FINAL_TEMPLATE", "status": "passed" if not template_errors else "failed", "errors": template_errors, "evidence": str(final_template.relative_to(root)).replace("\\", "/")})
    claim_text = claims.read_text(encoding="utf-8")
    claim_errors = validate_required_markers(claim_text, ("C1资源后悔损失", "选择性G2", "H3空间协同", "IQ压缩", "MIMO属性"))
    items.append({"item_id": "T03_CLAIM_MATRIX", "status": "passed" if not claim_errors else "failed", "errors": claim_errors, "evidence": str(claims.relative_to(root)).replace("\\", "/")})

    asset_errors = []
    if manifest.get("final_access_count") != 0 or manifest.get("acceptance", {}).get("final_flags_found") is not False:
        asset_errors.append("paper asset manifest is not development-only")
    for record in manifest.get("source_files", []) + manifest.get("generated_files", []):
        path = root / record["path"]
        if not path.is_file():
            asset_errors.append(f"missing manifest file: {record['path']}")
        elif sha256(path) != record["sha256"]:
            asset_errors.append(f"manifest SHA mismatch: {record['path']}")
    items.append({"item_id": "T04_GENERATED_ASSETS", "status": "passed" if not asset_errors else "failed", "errors": asset_errors, "evidence": str(manifest_path.relative_to(root)).replace("\\", "/")})

    access_errors = [] if access.get("access_count") == 0 else ["final access count is not zero"]
    items.append({"item_id": "T05_FINAL_ACCESS_GUARD", "status": "passed" if not access_errors else "failed", "errors": access_errors, "evidence": "configs/stage4_final_access_state.json"})

    literature_errors = []
    literature = json.loads(literature_registry.read_text(encoding="utf-8"))
    literature_errors += validate_reference_records(literature.get("references", []), minimum=7)
    literature_errors += validate_local_links(related_work)
    literature_errors += validate_required_markers(
        related_work.read_text(encoding="utf-8"),
        ("occupancy regret", "可审计数字包", "不能称MIMO算法", "Gate B", "C3失败"),
    )
    bib_text = bibliography.read_text(encoding="utf-8")
    literature_errors += validate_required_markers(
        bib_text,
        ("yi2025integrated", "li2024digital", "hu2024pragmatic", "cai2025endtoend", "liu2025mmcooper", "chen2026entropy", "li2026reliable"),
    )
    if literature.get("reference_count") != len(literature.get("references", [])):
        literature_errors.append("reference_count does not match registry records")
    if literature.get("acceptance", {}).get("unverified_references_in_core_matrix") != 0:
        literature_errors.append("core matrix contains unverified references")
    items.append({
        "item_id": "T06_VERIFIED_RELATED_WORK",
        "status": "passed" if not literature_errors else "failed",
        "errors": literature_errors,
        "evidence": "results/stage4/literature_audit_v1/verified_references.json",
    })
    result = {
        "audit_id": "stage4_thesis_materials_audit_v1",
        "acceptance": all(item["status"] == "passed" for item in items),
        "items": items,
        "failed_item_ids": [item["item_id"] for item in items if item["status"] == "failed"],
        "claim_boundary": "Draft and assets are development-only; final placeholders must remain until the single preregistered run.",
        "final_access_count": access.get("access_count"),
    }
    (output / "thesis_materials_audit_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Stage-4 thesis-material audit", "", f"Acceptance: `{result['acceptance']}`; final access count: `{result['final_access_count']}`.", "", "| ID | Status | Evidence | Errors |", "|---|---|---|---|"]
    for item in items:
        lines.append(f"| {item['item_id']} | {item['status']} | {item['evidence']} | {'; '.join(item['errors']) or '-'} |")
    lines += ["", f"> {result['claim_boundary']}", ""]
    (output / "thesis_materials_audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"acceptance": result["acceptance"], "failed_item_ids": result["failed_item_ids"], "final_access_count": result["final_access_count"]}, indent=2))


if __name__ == "__main__":
    main()
