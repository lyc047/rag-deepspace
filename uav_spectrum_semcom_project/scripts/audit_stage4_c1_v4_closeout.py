#!/usr/bin/env python
"""Audit immutable Final evidence, paper assets, and authoritative closeout docs."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_holdout import atomic_write_json, canonical_json_sha256  # noqa: E402
from spectrum_semcom.reproducibility import sha256_file  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check(name: str, condition: bool, evidence: str) -> dict:
    return {"id": name, "passed": bool(condition), "evidence": evidence}


def main() -> None:
    access_path = PROJECT_DIR / "configs/stage4_c1_v4_final_access_state.json"
    protocol_path = PROJECT_DIR / "configs/stage4_c1_v4_final_protocol.json"
    inventory_path = PROJECT_DIR / "results/stage4/c1_v4_final_inventory_v1/final_inventory.json"
    snapshot_path = PROJECT_DIR / "results/stage4/c1_v4_final_freeze_v1/executable_snapshot.json"
    result_path = PROJECT_DIR / "results/stage4/c1_v4_final_v1/c1_v4_final_result.json"
    manifest_path = PROJECT_DIR / "results/stage4/c1_v4_final_paper_assets_v1/paper_assets_manifest.json"
    access, protocol, inventory, snapshot, result, manifest = map(load, (access_path, protocol_path, inventory_path, snapshot_path, result_path, manifest_path))
    changed = [row["path"] for row in snapshot["files"] if not (PROJECT_DIR / row["path"]).is_file() or sha256_file(PROJECT_DIR / row["path"]) != row["sha256"]]
    outputs_ok = all((PROJECT_DIR / row["path"]).is_file() and sha256_file(PROJECT_DIR / row["path"]) == row["sha256"] for row in manifest["generated_files"])
    sources_ok = all((PROJECT_DIR / row["path"]).is_file() and sha256_file(PROJECT_DIR / row["path"]) == row["sha256"] for row in manifest["source_files"])
    h1 = result["hypotheses"]["H1_task_sufficient_representation"]
    h2 = result["hypotheses"]["H2_age_guarded_zero_bit_fallback"]
    final_docs = [PROJECT_DIR / "docs/STAGE4_C1_V4_FINAL_RESULTS.md", PROJECT_DIR / "docs/STAGE4_THESIS_CHAPTER_FINAL.md", PROJECT_DIR / "docs/STAGE4_CLAIM_MATRIX.md", PROJECT_DIR / "docs/STAGE4_CLOSEOUT_REPORT.md"]
    docs_exist = all(path.is_file() for path in final_docs)
    docs_text = "\n".join(path.read_text(encoding="utf-8") for path in final_docs if path.is_file())
    items = [
        check("immutable_snapshot", canonical_json_sha256(snapshot["files"]) == snapshot["executable_snapshot_sha256"] and not changed, f"changed={changed}"),
        check("single_use_access", access.get("status") == "access_consumed" and access.get("access_count") == 1, access.get("receipt_sha256", "")),
        check("registered_independence", inventory.get("scene_count") == 280 and inventory.get("cluster_count") == 74 and inventory.get("prior_provenance_overlap") == 0, "280 scenes, 74 clusters, overlap 0"),
        check("H1_confirmed", h1.get("confirmatory_pass") is True and all(h1["gates"].values()), str(h1["gates"])),
        check("H2_confirmed", h2.get("confirmatory_pass_under_fixed_sequence") is True and all(h2["gates"].values()), str(h2["gates"])),
        check("overall_final", result.get("overall_final_success") is True, result.get("status", "")),
        check("paper_assets_integrity", sources_ok and outputs_ok and len(manifest["generated_files"]) >= 6, f"sources={sources_ok}, outputs={outputs_ok}, files={len(manifest['generated_files'])}"),
        check("authoritative_docs", docs_exist and "[[FINAL_PENDING" not in docs_text, f"docs={len(final_docs)}"),
        check("claim_boundary", "跨数据集" in docs_text and "MIMO" in docs_text and "同一AERPAW" in docs_text, protocol["claim_boundary"]),
    ]
    result_audit = {
        "version": "1.0", "status": "stage4_closeout_complete" if all(item["passed"] for item in items) else "stage4_closeout_failed",
        "passed": all(item["passed"] for item in items), "items": items,
        "final_result_sha256": sha256_file(result_path), "paper_assets_manifest_sha256": sha256_file(manifest_path),
        "claim_boundary": protocol["claim_boundary"],
    }
    out = PROJECT_DIR / "results/stage4/c1_v4_closeout_v1"; out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "closeout_audit.json", result_audit)
    lines = ["# 阶段4收尾审计", "", f"- 总状态：`{result_audit['status']}`", f"- 通过：`{result_audit['passed']}`", "", "| 项目 | 通过 | 证据 |", "|---|---|---|"]
    lines.extend(f"| {item['id']} | {item['passed']} | {item['evidence']} |" for item in items)
    (out / "closeout_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out / 'closeout_audit.json'), "passed": result_audit["passed"], "failed": [item["id"] for item in items if not item["passed"]]}, indent=2))
    if not result_audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
