#!/usr/bin/env python
"""Exercise frozen final decision branches with synthetic metrics only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.final_statistics import FINAL_METRICS_SCHEMA, evaluate_final_statistics, validate_final_metrics_payload


def make_payload(success: bool) -> dict:
    c1, selective = [], []
    for index in range(240):
        scene = f"synthetic-statistics-selftest-{index:03d}"
        base = .025 + .012 * ((index * 37) % 101) / 100
        tail = .018 if index % 17 == 0 else 0.0
        c1_delta = (-.0035 - (.002 if index % 17 == 0 else 0.0)) if success else .0008
        c1.extend([
            {"scene_id": scene, "method": "detection_only", "regret": base + tail, "actual_bits": 1024 + index % 7},
            {"scene_id": scene, "method": "detection_plus_resource", "regret": base + tail + c1_delta, "actual_bits": 1024 + index % 7 + ((index % 3) - 1)},
        ])
        selective_delta = .0006 if success else .0042
        selective.extend([
            {"scene_id": scene, "method": "all_G2", "regret": base, "actual_bits": 2350 + index % 11},
            {"scene_id": scene, "method": "selective_G2", "regret": base + selective_delta, "actual_bits": 1800 + index % 11},
        ])
    return {
        "schema_version": FINAL_METRICS_SCHEMA,
        "registry_sha256": "a" * 64,
        "access_receipt_sha256": "b" * 64,
        "families": {"C1_resource_loss": {"rows": c1}, "selective_G2_digital_reporting": {"rows": selective}},
    }


def main() -> None:
    output = PROJECT_DIR / "results/stage4/final_statistics_selftest_v1"
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((PROJECT_DIR / "configs/stage4_protocol.json").read_text(encoding="utf-8"))
    success = evaluate_final_statistics(make_payload(True), protocol)
    failure = evaluate_final_statistics(make_payload(False), protocol)
    broken = make_payload(True)
    broken["families"]["selective_G2_digital_reporting"]["rows"].pop()
    integrity_errors = validate_final_metrics_payload(broken, 200)
    result = {
        "experiment_id": "stage4_final_statistics_synthetic_selftest_v1",
        "data_role": "synthetic_statistics_engine_selftest_not_RF_data",
        "success_branch": success,
        "failure_branch": failure,
        "integrity_rejection_errors": integrity_errors,
        "acceptance": {
            "success_branch_passes_both": all(success["family_decisions"].values()),
            "failure_branch_fails_both": not any(failure["family_decisions"].values()),
            "broken_payload_rejected": bool(integrity_errors),
            "completed": all(success["family_decisions"].values()) and not any(failure["family_decisions"].values()) and bool(integrity_errors),
        },
        "final_holdout_accessed": False,
        "claim_boundary": "This validates statistical code branches only and contains no sensing, link, or final evidence.",
    }
    (output / "final_statistics_selftest_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Final-statistics synthetic self-test", "", f"Success branch: `{success['family_decisions']}`.", f"Failure branch: `{failure['family_decisions']}`.", f"Broken payload rejection: `{bool(integrity_errors)}`.", "", f"> {result['claim_boundary']}", ""]
    (output / "final_statistics_selftest_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result["acceptance"], indent=2))


if __name__ == "__main__":
    main()
