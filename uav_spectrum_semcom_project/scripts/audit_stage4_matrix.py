#!/usr/bin/env python
"""Audit baseline, ablation, and sweep coverage after Gate A/B/C decisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.matrix_audit import summarize_coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_DIR)
    parser.add_argument("--output", type=Path, default=Path("results/stage4/matrix_coverage_v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    items = [
        {"item_id": "B01_REPRESENTATION", "scope": "development", "status": "passed", "evidence": ["docs/STAGE2_FINAL_REPORT.md", "results/stage4/end_to_end_sweep_v1/end_to_end_sweep_result.json"], "note": "hard semantic, strong PSD, and fixed G1/G2 were compared; G3 cost is present in C2/C3 diagnostics"},
        {"item_id": "B02_NODE_SELECTION", "scope": "development", "status": "passed", "evidence": ["results/stage4/value_network_v1/value_network_result.json", "results/stage4/analytic_scheduler_selective_v1/analytic_scheduler_result.json"], "note": "no-upgrade, SNR, confidence, prior value/bit, learned/analytic, and oracle baselines"},
        {"item_id": "B03_FUSION", "scope": "development", "status": "passed", "evidence": ["results/stage3/frozen_test_classical/stage3_fusion_smoke_result.json", "results/stage3/deepset_pilot/stage3_deepset_result.json", "results/stage4/reliable_minority_v2/reliable_minority_result.json"], "note": "classical, learned pilot, audited, and reliable-minority variants; negative results retained"},
        {"item_id": "B04_RETRANSMISSION", "scope": "development", "status": "passed", "evidence": ["docs/STAGE3_ADAPTIVE_FALLBACK_CONCLUSION.md", "results/stage4/reliable_minority_v2/reliable_minority_result.json"], "note": "no/CRC/fixed retransmission are stage-2 controls; ACK and task-conflict G3 are stage-3/4 controls"},
        {"item_id": "A01_REMOVE_RESOURCE_LOSS", "scope": "development", "status": "passed", "evidence": ["results/stage4/gate_a_training_v1/gate_a_paired_analysis.json"], "note": "detection-only is the paired removal"},
        {"item_id": "A02_RATE_TERM", "scope": "development", "status": "passed", "evidence": ["results/stage4/gate_a_training_v1/gate_a_paired_analysis.json"], "note": "detection+rate is a frozen negative control"},
        {"item_id": "A03_CONFIDENCE_FOR_VALUE", "scope": "development", "status": "passed", "evidence": ["results/stage4/value_network_v1/value_network_result.json"], "note": "confidence-G3 is compared directly"},
        {"item_id": "A04_REDUNDANCY_PENALTY", "scope": "development", "status": "waived", "evidence": ["docs/STAGE4_BATCH7_REPORT.md"], "note": "Gate B rejected all dynamic schedulers before a full RAVES redundancy term; no full-method claim is made"},
        {"item_id": "A05_CVAR_LOSS", "scope": "development", "status": "waived", "evidence": ["results/stage4/gate_a_training_v1/gate_a_paired_analysis.json"], "note": "tail term existed in full-joint negative control; isolated retuning was stopped by D4-003"},
        {"item_id": "A06_RELIABLE_MINORITY", "scope": "development", "status": "passed", "evidence": ["results/stage4/reliable_minority_v2/reliable_minority_result.json"], "note": "direct protection and confirmation retransmission were both evaluated and rejected"},
        {"item_id": "A07_ANOMALY_QUALITY", "scope": "development", "status": "passed", "evidence": ["results/stage4/quality_diagnostics_v1/quality_diagnostics_result.json", "results/stage4/reliable_minority_v2/reliable_minority_result.json"], "note": "quality calibration and controlled anomaly sensitivity are reported"},
        {"item_id": "A08_ACK_FALLBACK", "scope": "development", "status": "passed", "evidence": ["docs/STAGE3_ADAPTIVE_FALLBACK_CONCLUSION.md"], "note": "ACK fallback bit/latency tradeoff is retained from the frozen stage-3 component"},
        {"item_id": "A09_FIXED_GRANULARITY", "scope": "development", "status": "passed", "evidence": ["results/stage4/end_to_end_sweep_v1/end_to_end_sweep_result.json"], "note": "all-G1 and all-G2 are decoder-path controls"},
        {"item_id": "A10_FROZEN_DETECTOR", "scope": "development", "status": "passed", "evidence": ["results/stage4/gate_a_training_v1/gate_a_training_result.json"], "note": "C1 uses the frozen detector cache"},
        {"item_id": "A11_JOINT_FINETUNE", "scope": "development", "status": "waived", "evidence": ["docs/STAGE4_BATCH3_REPORT.md"], "note": "full-joint head control failed; detector unfreezing was stopped by D4-003 to avoid validation retuning"},
        {"item_id": "A12_FULL_RAVES", "scope": "development", "status": "waived", "evidence": ["docs/STAGE4_DEVELOPMENT_SYNTHESIS.md"], "note": "C2/C3 gates failed, so no nonexistent full RAVES method is claimed"},
        {"item_id": "M01_NODE_COUNT_1_2_4_8", "scope": "development", "status": "passed", "evidence": ["results/phase1/multi_uav_pressure_sweep.json", "results/stage4/multinode_cache_v1/multinode_cache_result.json"], "note": "node-count pressure exists; unrelated aggregation remains H1/H4 diagnosis, not H3"},
        {"item_id": "M02_RESOURCE_AND_DEMAND_GRID", "scope": "development", "status": "passed", "evidence": ["results/stage4/task_difficulty_audit_v2/stage4_task_difficulty_result.json"], "note": "4/8/16 resources and 1/2/4 contiguous demand were truth-only audited"},
        {"item_id": "M03_BUDGET_GRID", "scope": "development", "status": "waived", "evidence": ["results/stage4/analytic_scheduler_selective_v1/analytic_scheduler_result.json", "results/stage4/gate_a_training_v1/gate_a_training_result.json"], "note": "512-bit Gate A and local C2 budget curves were run; 1/2/4/8-kbit full-method sweep was stopped after Gate B"},
        {"item_id": "M04_SENSING_SNR_GRID", "scope": "development", "status": "waived", "evidence": ["results/stage4/multinode_cache_v1/multinode_cache_result.json", "docs/STAGE3_ROBUSTNESS_CONCLUSION.md"], "note": "heterogeneous -6/0/6/12 and prior robustness evidence exist; full -12-to-12 full-method sweep stopped after gates"},
        {"item_id": "M05_REPORTING_CHANNEL_GRID", "scope": "development", "status": "passed", "evidence": ["results/stage4/end_to_end_sweep_v1/end_to_end_sweep_result.json", "docs/STAGE2_FINAL_REPORT.md"], "note": "stage4 covers AWGN/Rayleigh/Rician at 0/3/6/9 dB; stage2 supplies the 12-dB reference"},
        {"item_id": "M06_FAILURE_AND_AGE", "scope": "development", "status": "passed", "evidence": ["results/stage3/robustness_holdout/stage3_robustness_result.json", "results/stage4/reliable_minority_v2/reliable_minority_result.json"], "note": "dropout and stale-report conditions are included"},
        {"item_id": "M07_ANOMALY_GRID", "scope": "development", "status": "passed", "evidence": ["results/stage4/reliable_minority_v2/reliable_minority_result.json", "results/stage4/quality_diagnostics_v1/quality_diagnostics_result.json"], "note": "high-confidence wrong, clipping/leakage diagnostics, shift, stale, and dropout are covered"},
        {"item_id": "M08_RELIABLE_MINORITY_2V2", "scope": "development", "status": "waived", "evidence": ["docs/STAGE4_BATCH8_REPORT.md"], "note": "1-v-3 family failed Gate C, so 2-v-2 expansion was not used for threshold search"},
        {"item_id": "M09_FEC_AND_RETRANSMISSION", "scope": "development", "status": "passed", "evidence": ["configs/stage2_digital_link.json", "docs/STAGE2_FINAL_REPORT.md"], "note": "Hamming and LDPC reference paths and 0/1/2 retransmission controls are stage-2 frozen components"},
        {"item_id": "M10_MODEL_COMPLEXITY", "scope": "development", "status": "passed", "evidence": ["results/stage4/complexity_profile_v1/complexity_result.json"], "note": "parameters, linear MACs, checkpoint bytes, and CPU latency are reported"},
        {"item_id": "F04_FULL_FINAL_MATRIX", "scope": "final", "status": "blocked", "evidence": ["configs/stage4_dataset_registry_v1.json"], "note": "new independent final scenes do not exist; only preregistered C1 and selective-G2 primary families will run"},
        {"item_id": "F05_REAL_H3_MATRIX", "scope": "final", "status": "blocked", "evidence": ["configs/stage4_dataset_registry_v1.json"], "note": "30+ real multi-receiver scenes and H3 comparison set are unavailable"},
    ]
    for item in items:
        missing = [path for path in item["evidence"] if not (root / path).exists()]
        if missing and item["status"] != "blocked":
            item["status"] = "failed"
            item["note"] += "; missing evidence: " + ", ".join(missing)
    summary = summarize_coverage(items)
    result = {
        "audit_id": "stage4_matrix_coverage_v1",
        **summary,
        "items": items,
        "interpretation": "Waived means a preregistered gate stopped that branch; it is not a positive result. Blocked items require independent final data.",
        "final_holdout_accessed": False,
    }
    (output / "matrix_coverage_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Stage-4 baseline, ablation, and matrix coverage", "", f"Development complete: `{result['development_complete']}`; counts: `{result['counts']}`.", "", "| ID | Scope | Status | Note |", "|---|---|---|---|"]
    lines += [f"| {x['item_id']} | {x['scope']} | {x['status']} | {x['note']} |" for x in items]
    lines += ["", f"> {result['interpretation']}", ""]
    (output / "matrix_coverage_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
