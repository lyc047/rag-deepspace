import json
from pathlib import Path

from scripts.audit_stage8_local_metadata import ROOT, audit


def test_stage8_local_metadata_audit_does_not_authorize_final(
    tmp_path: Path,
) -> None:
    result = audit(
        ROOT / "configs/stage8_local_metadata_sources_v1.json",
        tmp_path / "result.json",
    )
    assert result["status"] == "METADATA_AUDIT_COMPLETE"
    assert result["signal_values_accessed"] is False
    assert result["archive_payloads_opened"] is False
    assert result["eligible_new_independent_units"] == 0
    assert result["decision"]["local_external_final_ready"] is False
    assert result["decision"]["new_external_data_required"] is True
    assert result["decision"]["execution_authorized"] is False


def test_stage8_local_source_ids_are_unique() -> None:
    path = ROOT / "configs/stage8_local_metadata_sources_v1.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    source_ids = [source["source_id"] for source in config["sources"]]
    assert len(source_ids) == len(set(source_ids))
    assert all(int(source["eligible_new_units"]) == 0 for source in config["sources"])
