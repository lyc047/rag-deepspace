#!/usr/bin/env python
"""Single-use Stage-6R supportive confirmation on six lockbox sites."""

from __future__ import annotations

from pathlib import Path

import run_stage6r_electrosense_external_final as executor


PROJECT_DIR = Path(__file__).resolve().parents[1]
executor.DEFAULT_CONFIG = (
    PROJECT_DIR / "configs/stage6r_confirmation_lockbox_protocol_v1.json"
)
executor.DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "results/stage6r/confirmation_lockbox_v1/result.json"
)


if __name__ == "__main__":
    executor.main()
