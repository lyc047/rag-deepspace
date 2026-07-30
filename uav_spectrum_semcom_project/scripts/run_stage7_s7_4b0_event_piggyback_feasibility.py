#!/usr/bin/env python
"""Run the registered S7.4B-0 event-context piggyback scan."""

from pathlib import Path

from run_stage7_s7_4a1_checkpoint_protocol_scan import main


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_DIR
    / "configs/stage7_s7_4b0_event_piggyback_feasibility_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results/stage7/s7_4b0_event_piggyback_feasibility_v1/result.json"
)


if __name__ == "__main__":
    main(DEFAULT_CONFIG, DEFAULT_OUTPUT)
