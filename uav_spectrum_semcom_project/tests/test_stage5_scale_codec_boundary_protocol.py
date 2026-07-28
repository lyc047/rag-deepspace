import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_scale_boundary_protocol_is_final_closed() -> None:
    protocol = json.loads(
        (ROOT / "configs/stage5_scale_codec_boundary_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["scale_grid"]["n_channels"] == [8, 16, 32, 64]
    assert set(protocol["scale_grid"]["query_families"]) == {"q3", "q5", "q7"}
    assert protocol["scale_grid"]["application_header_bits"] == [40, 152]
    assert set(protocol["link"]["fec_schemes"]) == {
        "none",
        "hamming74",
        "repetition3",
    }
    assert protocol["governance"]["stage4_final_measurements_may_be_loaded"] is False
    assert protocol["governance"]["stage4_final_metrics_may_be_loaded"] is False
    assert protocol["governance"]["output_is_confirmatory_final"] is False


def test_scale_boundary_result_is_complete_and_final_closed() -> None:
    result = json.loads(
        (
            ROOT
            / "results/stage5/scale_codec_boundary_v1"
            / "scale_codec_boundary_result.json"
        ).read_text(encoding="utf-8")
    )
    assert result["status"] == "stage5_scale_codec_boundary_complete"
    assert len(result["rows"]) == 4320
    assert len(result["comparisons"]) == 3456
    assert result["governance"]["stage4_final_measurements_loaded"] is False
    assert result["governance"]["stage4_final_metrics_loaded"] is False
    assert result["governance"]["confirmatory_final"] is False
    assert result["governance"]["measured_scaled_spectra_loaded"] is False

    primary = [
        row
        for row in result["rows"]
        if row["n_channels"] == 64
        and row["query_family"] == "q7"
        and row["application_header_bits"] == 40
        and row["fec"] == "hamming74"
        and row["channel"] == "awgn"
        and row["ebn0_db"] == 6.0
    ]
    nominal = {
        row["representation"]: row["nominal_transmitted_bits"]
        for row in primary
    }
    assert nominal["bundle_all_queries"] == 336.0
    assert nominal["occupancy_vector_all_queries"] == 378.0
    assert nominal["soft_power_all_queries"] == 714.0
