import csv
import io
from zipfile import ZipFile

from scripts.analyze_stage8_aadm_development import (
    lag1_correlation,
    parse_lora_flight,
    run_length_encode,
    runs_of_ones,
)


def test_runs_and_lag1_detect_bursty_failures() -> None:
    sequence = [0, 0, 1, 1, 1, 0, 0]
    assert runs_of_ones(sequence) == [3]
    assert lag1_correlation(sequence) is not None
    assert lag1_correlation(sequence) > 0
    assert run_length_encode(sequence) == [[0, 2], [1, 3], [0, 2]]


def test_parser_deduplicates_gateways_and_segments_counter_restart(
    tmp_path,
) -> None:
    archive_path = tmp_path / "aadm.zip"
    member = "AADM2025Dryad/LORA/flight-a/LORAlog.csv"
    rows = [
        ("p1", 1, "2025-01-01T00:00:01+00:00", "g1"),
        ("p1", 1, "2025-01-01T00:00:01+00:00", "g2"),
        ("p2", 2, "2025-01-01T00:00:02+00:00", "g1"),
        ("p4", 4, "2025-01-01T00:00:04+00:00", "g1"),
        ("p5", 1, "2025-01-01T00:00:05+00:00", "g1"),
        ("p7", 3, "2025-01-01T00:00:07+00:00", "g1"),
    ]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "deduplication_id",
            "f_cnt",
            "rx_info_time",
            "rx_info_gatewayId",
            "rx_info_snr",
            "rx_info_rssi",
        ]
    )
    for packet_id, counter, timestamp, gateway in rows:
        writer.writerow([packet_id, counter, timestamp, gateway, 4.0, -90.0])
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(member, buffer.getvalue())

    with ZipFile(archive_path) as archive:
        result = parse_lora_flight(archive, "flight-a")
    assert result["row_count"] == 6
    assert result["unique_packet_count"] == 5
    assert result["counter_restart_events"] == 1
    assert result["segment_count"] == 2
    assert result["internal_gap_count"] == 2
    assert result["internal_frame_span"] == 7
    assert result["maximum_loss_run"] == 1
