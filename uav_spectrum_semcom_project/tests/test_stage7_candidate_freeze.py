import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stage7_candidate_freeze_hashes_and_governance() -> None:
    result = json.loads(
        (
            ROOT
            / "results/stage7/stage7_candidate_freeze_v1/result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["status"] == "FROZEN"
    assert result["candidate_id"] == "S6R-FH10-v1"
    assert all(
        row["passed"] for row in result["decision_checks"].values()
    )
    assert len(result["artifact_hashes"]) == 17
    for relative, metadata in result["artifact_hashes"].items():
        payload = (ROOT / relative).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]
        assert len(payload) == metadata["bytes"]
    governance = result["governance"]
    assert governance["stage7_has_no_external_final"]
    assert governance["internal_development_test_must_remain_closed"]
    assert governance["reserve_must_remain_closed"]
    assert governance["future_final_requires_new_independent_units"] >= 20
