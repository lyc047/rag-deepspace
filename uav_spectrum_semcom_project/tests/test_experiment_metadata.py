import json
from pathlib import Path

from spectrum_semcom.research_scope import validate_experiment_metadata


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_experiment_metadata_template_satisfies_stage0_scope() -> None:
    path = PROJECT_DIR / "configs" / "experiment_metadata.template.json"
    with path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert validate_experiment_metadata(metadata) == []


def test_experiment_metadata_rejects_missing_claim_mapping() -> None:
    assert "hypothesis_ids must be a non-empty list" in validate_experiment_metadata({"hypothesis_ids": []})


def test_experiment_metadata_rejects_unknown_hypothesis() -> None:
    path = PROJECT_DIR / "configs" / "experiment_metadata.template.json"
    with path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata["hypothesis_ids"] = ["H9"]

    assert validate_experiment_metadata(metadata) == ["unknown hypothesis ids: H9"]


def test_stage2_digital_link_config_is_formal_and_fair() -> None:
    path = PROJECT_DIR / "configs" / "stage2_digital_link.json"
    with path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert validate_experiment_metadata(metadata) == []
    assert metadata["hypothesis_ids"] == ["H1", "H4"]
    assert metadata["fairness_budget_type"] == "latency"
    assert metadata["packet"]["crc_bits"] == 16
    assert metadata["reporting_channel"]["fec"] == "hamming74"
    iq = next(item for item in metadata["source_representations"] if item["name"] == "iq_int12_complex")
    assert "theoretical" in iq["source"]
    assert "not a measured codec result" in iq["caveat"]
    enhancement = metadata["stage2_enhancement"]
    assert enhancement["robustness"]["channel_models"] == ["awgn", "rayleigh", "rician"]
    assert enhancement["pareto"]["latency_budget_s_values"] == [0.005, 0.025, 0.1, 0.25]
    assert enhancement["observations_per_pressure_decision"] == 4
    quantizer = metadata["psd_quantizer_selection"]
    assert quantizer["selection_split"] == "val"
    assert quantizer["test_split"] == "test"
    assert quantizer["candidate_bits"] == [1, 2, 3, 4, 6, 8]
    full_test = metadata["full_test_validation"]
    assert full_test["use_all_frames"] is True
    assert full_test["expected_frame_count"] == 20001
    assert full_test["psd_selection_result"].endswith("stage2_psd_quantizer_result.json")
    ldpc = metadata["ldpc_validation"]
    assert ldpc["library_version"] == "0.7.9"
    assert ldpc["energy_definition"] == "EbN0_per_transmitted_coded_bit"
    assert ldpc["code_length"] == 1200
