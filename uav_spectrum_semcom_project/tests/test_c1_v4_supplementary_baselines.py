import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_stage4_c1_v4_supplementary_baselines import representation_values


def test_supplementary_protocol_never_loads_final() -> None:
    protocol = json.loads((ROOT / "configs/stage4_c1_v4_supplementary_baselines.json").read_text(encoding="utf-8"))
    assert protocol["governance"]["final_measurements_or_final_metrics_may_be_loaded"] is False
    assert protocol["governance"]["may_change_final_claim_or_select_algorithm"] is False
    assert "final" not in protocol["representation_cache_result"]["path"].lower()
    assert "final" not in protocol["reliability_cache_result"]["path"].lower()


def test_hard_and_soft_baseline_decisions_are_frozen() -> None:
    cache = {
        "features": np.asarray([[[0.0, 0.2], [0.0, 0.4], [0.0, 0.1], [0.0, 0.5], [0.0, 0.0], [0.0, 0.6], [0.0, 0.2], [0.0, 0.1]]]),
        "occupancy": np.asarray([[0.0, 0.2, 0.8, 1.0, 0.5, 0.2, 0.0, 0.0]]),
        "normalized_channel_power": np.asarray([[0.0, 0.1, 0.8, 1.0, 0.7, 0.3, 0.1, 0.0]]),
        "best_block_indicator": np.asarray([[1.0, 1.0, 1.0, 1.0, 0.0]]),
    }
    hard, kind, bits = representation_values("hard_mean_energy_1bit", cache, 0, 0.375)
    assert kind == "channel" and bits == 1
    assert hard.tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    soft, _, bits = representation_values("soft_channel_power_8bit", cache, 0, 0.375)
    assert bits == 8 and np.all((soft >= 0) & (soft <= 1))
    indicator, kind, bits = representation_values("proposed_best_block_indicator_1bit", cache, 0, 0.375)
    assert kind == "block" and bits == 1 and int(np.argmin(indicator)) == 4
