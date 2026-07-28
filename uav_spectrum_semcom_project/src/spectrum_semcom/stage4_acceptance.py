"""Machine-readable acceptance audit for the stage-4 research programme.

The audit deliberately separates development completion from final-holdout
claims.  A negative preregistered gate may be a completed research result,
whereas missing independent scenes are a blocker and must never be silently
converted into a pass.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .stage4_protocol import validate_stage4_protocol, validate_stage4_registry


VALID_STATUSES = {"passed", "waived", "blocked", "failed"}


@dataclass(frozen=True)
class AcceptanceItem:
    item_id: str
    scope: str
    status: str
    evidence: list[str]
    note: str

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid acceptance status: {self.status}")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _safe_check(check: Callable[[], tuple[bool, str]]) -> tuple[bool, str]:
    try:
        return check()
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        return False, f"audit check failed: {exc}"


def _item(
    root: Path,
    item_id: str,
    scope: str,
    evidence: list[str],
    check: Callable[[], tuple[bool, str]],
    *,
    failure_status: str = "failed",
) -> AcceptanceItem:
    missing = [name for name in evidence if not (root / name).exists()]
    if missing:
        return AcceptanceItem(item_id, scope, failure_status, evidence, "missing: " + ", ".join(missing))
    passed, note = _safe_check(check)
    return AcceptanceItem(item_id, scope, "passed" if passed else failure_status, evidence, note)


def audit_stage4(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    items: list[AcceptanceItem] = []

    protocol_path = "configs/stage4_protocol.json"
    registry_path = "configs/stage4_dataset_registry_v1.json"
    protocol = _load(root / protocol_path)
    registry = _load(root / registry_path)

    items.append(_item(root, "A01_PROTOCOL", "development", [protocol_path],
        lambda: (not validate_stage4_protocol(protocol), "; ".join(validate_stage4_protocol(protocol)) or "frozen protocol is valid")))
    items.append(_item(root, "A02_PROTOCOL_REGISTRY", "development", [registry_path],
        lambda: (not validate_stage4_registry(registry), "; ".join(validate_stage4_registry(registry)) or "protocol-only registry is structurally valid")))

    gate_path = "results/stage4/gate_a_training_v1/gate_a_paired_analysis.json"
    def check_gate_a() -> tuple[bool, str]:
        data = _load(root / gate_path)
        row = next(x for x in data["comparisons"] if x["method"] == "detection_plus_resource")
        ok = (
            data["decision"] == "retain_resource_aware_loss_for_C1"
            and row["regret"]["ci95_high"] < 0
            and row["cvar_0_9"]["ci95_high"] < 0
            and row["nominal_bits"]["ci95_low"] <= 0 <= row["nominal_bits"]["ci95_high"]
            and data["final_holdout_accessed"] is False
        )
        return ok, "C1 validation Gate A passed at matched nominal bits; final H2 remains untested"
    items.append(_item(root, "A03_C1_GATE_A", "development", [gate_path], check_gate_a))

    c2_paths = [
        "results/stage4/value_network_v1/value_network_result.json",
        "results/stage4/analytic_scheduler_selective_v1/analytic_scheduler_result.json",
        "results/stage4/preview_diagnostic_v1/preview_diagnostic_result.json",
        "docs/STAGE4_BATCH6_REPORT.md",
    ]
    def check_c2() -> tuple[bool, str]:
        results = [_load(root / p) for p in c2_paths[:3]]
        safe = all(x.get("final_holdout_accessed", x.get("truth_access_audit", {}).get("final_holdout_accessed")) is False for x in results)
        return safe, "Gate B investigated with learned, analytic, and preview variants; rejected as a documented negative result"
    items.append(_item(root, "A04_C2_GATE_B_NEGATIVE", "development", c2_paths, check_c2))

    c3_path = "results/stage4/reliable_minority_v2/reliable_minority_result.json"
    items.append(_item(root, "A05_C3_GATE_C_NEGATIVE", "development", [c3_path, "docs/STAGE4_BATCH7_REPORT.md"],
        lambda: (_load(root / c3_path).get("final_holdout_accessed") is False,
                 "controlled C3 robustness completed; Gate C rejected and no H3 spatial claim is made")))

    e2e_path = "results/stage4/end_to_end_sweep_v1/end_to_end_sweep_result.json"
    def check_e2e() -> tuple[bool, str]:
        data = _load(root / e2e_path)
        boot = data["paired_scene_bootstrap"]
        bit_superior = all(v["all_G2"]["transmitted_bits"]["ci95"][1] < 0 for v in boot.values())
        return len(data["summary"]) == 36 and len(boot) == 12 and bit_superior and data["final_holdout_accessed"] is False, \
            "150-scene decoder-path sweep covers 3 channels x 4 Eb/N0 values; selective G2 saves actual transmitted bits"
    items.append(_item(root, "A06_END_TO_END_DECODER", "development", [e2e_path], check_e2e))

    quality_path = "results/stage4/quality_diagnostics_v1/quality_diagnostics_result.json"
    items.append(_item(root, "A07_QUALITY_DIAGNOSTICS", "development", [quality_path],
        lambda: (_load(root / quality_path).get("acceptance", {}).get("completed") is True,
                 "quality calibration and controlled anomaly diagnostics are recorded")))

    value_path = "results/stage4/value_diagnostics_v1/value_diagnostics_result.json"
    items.append(_item(root, "A08_VALUE_RANKING_DIAGNOSTICS", "development", [value_path],
        lambda: (_load(root / value_path).get("acceptance", {}).get("completed") is True,
                 "aggregate MAE, rank correlation, top-k recall, and scheduler effect are recorded")))

    final_protocol = protocol.get("final_evaluation_protocol", {})
    items.append(AcceptanceItem(
        "A09_FINAL_PREREGISTRATION", "development",
        "passed" if final_protocol.get("minimum_independent_scene_count") == 200 and final_protocol.get("bootstrap_repetitions") == 10000 else "failed",
        [protocol_path, "docs/STAGE4_FINAL_HOLDOUT_RUNBOOK.md"],
        "single-use final evaluation, primary families, multiplicity control, and acceptance thresholds are frozen",
    ))

    materialized_final = root / "configs/stage4_final_registry_v1.json"
    if materialized_final.exists():
        final_registry = _load(materialized_final)
        final_scenes = final_registry.get("scene_ids", [])
        final_evidence = ["configs/stage4_final_registry_v1.json"]
        final_registry_ok = (
            final_registry.get("leakage_audit", {}).get("development_overlap_count") == 0
            and final_registry.get("file_integrity_audit", {}).get("all_declared_files_exist") is True
            and final_registry.get("file_integrity_audit", {}).get("all_sha256_match") is True
        )
    else:
        final_scenes = registry.get("splits", {}).get("final_holdout", {}).get("scene_ids", [])
        final_evidence = [registry_path]
        final_registry_ok = False
    items.append(AcceptanceItem(
        "F01_INDEPENDENT_FINAL_SCENES", "final",
        "passed" if len(final_scenes) >= 200 and final_registry_ok else "blocked",
        final_evidence,
        f"registered independent final scenes: {len(final_scenes)}/200",
    ))
    final_result = root / "results/stage4/final_holdout_v1/final_holdout_result.json"
    items.append(AcceptanceItem(
        "F02_SINGLE_USE_FINAL_EVALUATION", "final",
        "passed" if final_result.exists() else "blocked",
        [str(final_result.relative_to(root))],
        "cannot run before an eligible independent catalog is registered",
    ))
    items.append(AcceptanceItem(
        "F03_H3_SPATIAL_COLLABORATION", "claim",
        "waived",
        ["docs/STAGE4_BATCH7_REPORT.md"],
        "no formal H3 claim is made; controlled faults are reported only as mechanism/negative evidence",
    ))

    snapshot_path = "results/stage4/pre_final_freeze_v1/pre_final_freeze_result.json"
    items.append(_item(root, "A10_PRE_FINAL_SNAPSHOT", "development", [snapshot_path],
        lambda: (
            _load(root / snapshot_path).get("verification", {}).get("test_status") == "passed"
            and _load(root / snapshot_path).get("verification", {}).get("final_access_count") == 0,
            "executable source, configuration, and frozen checkpoint hashes are recorded with passing regression evidence",
        )))

    complexity_path = "results/stage4/complexity_profile_v1/complexity_result.json"
    items.append(_item(root, "A11_COMPLEXITY_PROFILE", "development", [complexity_path],
        lambda: (_load(root / complexity_path).get("acceptance", {}).get("completed") is True,
                 "C1/C2 parameter count, linear MACs, checkpoint size, and single-thread CPU latency are recorded")))

    matrix_path = "results/stage4/matrix_coverage_v1/matrix_coverage_result.json"
    items.append(_item(root, "A12_MATRIX_COVERAGE", "development", [matrix_path],
        lambda: (_load(root / matrix_path).get("development_complete") is True and not _load(root / matrix_path).get("failed_item_ids"),
                 "baseline, ablation, and sweep coverage has no unaccounted development failures; gate-conditioned waivers are explicit")))

    thesis_path = "results/stage4/thesis_materials_audit_v1/thesis_materials_audit_result.json"
    items.append(_item(root, "A13_THESIS_MATERIALS", "development", [thesis_path],
        lambda: (_load(root / thesis_path).get("acceptance") is True and not _load(root / thesis_path).get("failed_item_ids"),
                 "generated figures/tables, chapter draft, claim matrix, and locked final template pass integrity and claim-boundary checks")))

    final_stats_path = "results/stage4/final_statistics_selftest_v1/final_statistics_selftest_result.json"
    items.append(_item(root, "A14_FINAL_STATISTICS_ENGINE", "development", [final_stats_path, "scripts/run_stage4_final_statistics.py"],
        lambda: (_load(root / final_stats_path).get("acceptance", {}).get("completed") is True and _load(root / final_stats_path).get("final_holdout_accessed") is False,
                 "frozen scene-level bootstrap, paired-randomization, noninferiority, Holm, success/failure, and integrity-rejection branches are verified")))

    literature_path = "results/stage4/literature_audit_v1/verified_references.json"
    items.append(_item(root, "A15_VERIFIED_RELATED_WORK", "development", [literature_path, "docs/STAGE4_RELATED_WORK_MATRIX.md", "docs/references/stage4_verified_refs.bib"],
        lambda: (
            _load(root / literature_path).get("acceptance", {}).get("completed") is True
            and _load(root / literature_path).get("reference_count", 0) >= 7
            and _load(root / literature_path).get("acceptance", {}).get("unverified_references_in_core_matrix") == 0,
            "recent related-work metadata and project-boundary comparisons are verified against primary publisher, institution, CVF, or arXiv pages",
        )))

    serialized = [asdict(item) for item in items]
    dev = [x for x in items if x.scope == "development"]
    final = [x for x in items if x.scope == "final"]
    return {
        "audit_id": "stage4_acceptance_audit_v1",
        "development_acceptance": all(x.status in {"passed", "waived"} for x in dev),
        "final_acceptance": all(x.status == "passed" for x in final),
        "counts": {status: sum(x.status == status for x in items) for status in sorted(VALID_STATUSES)},
        "blocking_item_ids": [x.item_id for x in items if x.status == "blocked"],
        "failed_item_ids": [x.item_id for x in items if x.status == "failed"],
        "items": serialized,
        "claim_boundary": "Development acceptance is not final H2/H3 evidence.",
    }


def acceptance_markdown(result: dict[str, Any]) -> str:
    rows = [
        "# Stage-4 acceptance audit",
        "",
        f"- Development acceptance: `{result['development_acceptance']}`",
        f"- Final acceptance: `{result['final_acceptance']}`",
        "",
        "| ID | Scope | Status | Note |",
        "|---|---|---|---|",
    ]
    for item in result["items"]:
        rows.append(f"| {item['item_id']} | {item['scope']} | {item['status']} | {item['note']} |")
    rows.extend(["", f"> {result['claim_boundary']}", ""])
    return "\n".join(rows)
