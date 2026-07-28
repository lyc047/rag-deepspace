"""Measure CPU wall-clock cost of the actual LDPC/BP waveform decoder."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.ldpc_link import LdpcCodeConfig, make_ldpc_matrices, simulate_ldpc_packet_success
from spectrum_semcom.reproducibility import environment_snapshot, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile actual pyldpc BP decoding used in stage 2.")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "configs" / "stage2_digital_link.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results" / "stage2" / "ldpc_validation")
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")); settings = config["ldpc_validation"]
    code = LdpcCodeConfig(code_length=int(settings["code_length"]), variable_node_degree=int(settings["variable_node_degree"]), check_node_degree=int(settings["check_node_degree"]), systematic=bool(settings["systematic"]), matrix_seed=int(settings["matrix_seed"]), max_decoder_iterations=int(settings["max_decoder_iterations"]), application_header_bits=int(settings["application_header_bits"]), crc_bits=int(settings["crc_bits"]))
    matrices = make_ldpc_matrices(code); rows = []
    for index, ebn0 in enumerate([2.0, 3.0]):
        started = time.perf_counter()
        outcome = simulate_ldpc_packet_success(matrices, code, ebn0, args.trials, min(args.trials, int(settings["batch_size"])), int(config["seed"]) + 70_000_000 + index)
        elapsed = time.perf_counter() - started
        rows.append({"ebn0_db": ebn0, "trials": args.trials, "max_bp_iterations": code.max_decoder_iterations, "wall_time_s": elapsed, "mean_wall_time_ms_per_codeword": 1000 * elapsed / args.trials, "success_rate": outcome["success_rate"]})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ldpc_decoder_profile.json").write_text(json.dumps({"config_sha256": sha256_file(args.config), "rows": rows, "scope": "CPU wall-clock encode plus BP decode; excludes RF front-end and energy measurement", "environment": environment_snapshot(["numpy", "pyldpc", "numba"])}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
