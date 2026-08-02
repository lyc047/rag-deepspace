import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".json", ".md", ".py"}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_stage9_closeout_hashes_decisions_and_boundaries() -> None:
    result = json.loads(
        (
            ROOT
            / "results/stage9/stage9_closeout_freeze_v1/result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["status"] == "FROZEN"
    assert result["evidence_bundle_id"] == "S9-EVIDENCE-BUNDLE-v1"
    assert "not a separately tested" in result["bundle_semantics"]
    assert all(row["passed"] for row in result["required_check_results"])

    for relative, metadata in result["artifact_hashes"].items():
        path = ROOT / relative
        payload = path.read_bytes()
        if metadata["hash_mode"] == "lf_normalized_bytes":
            assert path.suffix.lower() in TEXT_SUFFIXES or path.name == ".gitattributes"
            payload = payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        else:
            assert metadata["hash_mode"] == "exact_bytes"
        assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]
        assert len(payload) == metadata["bytes"]

    bundle_material = {
        "config_sha256": result["config_sha256"],
        "artifact_hashes": result["artifact_hashes"],
        "required_check_results": result["required_check_results"],
        "frozen_stack": result["frozen_stack"],
        "governance": result["governance"],
    }
    assert _canonical_sha256(bundle_material) == result["bundle_sha256"]

    stack = result["frozen_stack"]
    assert stack["external_data_algorithm_boundary"] == "S6R-FH10-v1"
    assert stack["causal_recovery_semantics"] == "S9-CRN32-HBW30-v1"
    assert stack["compact_udp_framing"] == "S9-CH8-v1"
    assert "rejected" in stack["temporal_batching"]

    governance = result["governance"]
    assert governance["stage9_is_protocol_and_software_evidence_only"]
    assert governance["stage6_external_final_status_must_remain_failed"]
    assert governance["stage9_4_causal_leakage_must_remain_disclosed"]
    assert governance["stage9_9_negative_result_must_remain_disclosed"]
    assert governance["b2_b4_must_not_enter_frozen_stack"]
