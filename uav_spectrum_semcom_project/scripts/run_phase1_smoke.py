from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from spectrum_semcom.baselines import energy_detector
from spectrum_semcom.bit_budget import BitBudget, iq_payload_bits, semantic_packet_bits, stft_payload_bits
from spectrum_semcom.metrics import best_box_iou
from spectrum_semcom.preprocessing import stft_power
from spectrum_semcom.synthetic import generate_synthetic_iq_frame
from spectrum_semcom.types import SemanticPacket


def main() -> None:
    frame = generate_synthetic_iq_frame(
        n_samples=65_536,
        sample_rate_hz=10e6,
        n_signals=3,
        snr_db=8.0,
        seed=7,
        frame_id="smoke_0001",
    )
    stft = stft_power(frame.iq, frame.sample_rate_hz, n_fft=512, hop_length=128)
    pred_boxes = energy_detector(stft)
    packet = SemanticPacket(uav_id=0, frame_id=frame.frame_id, boxes=pred_boxes, metadata_bits=128)

    budgets = [
        BitBudget("raw_iq_12bit_iq", iq_payload_bits(frame.n_samples, 12, 12), frame.duration_s),
        BitBudget("stft_8bit", stft_payload_bits(stft.power_db.shape, 8), frame.duration_s),
        BitBudget("semantic_packet", semantic_packet_bits(packet), frame.duration_s),
    ]

    result = {
        "frame_id": frame.frame_id,
        "n_samples": frame.n_samples,
        "duration_s": frame.duration_s,
        "sample_rate_hz": frame.sample_rate_hz,
        "truth_boxes": [box.__dict__ for box in frame.boxes],
        "pred_boxes": [box.__dict__ for box in pred_boxes],
        "best_tf_iou": best_box_iou(pred_boxes, frame.boxes),
        "payloads": [
            {"name": b.name, "bits_per_frame": b.bits_per_frame, "bitrate_bps": b.bitrate_bps}
            for b in budgets
        ],
    }

    out_dir = PROJECT_DIR / "results" / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smoke_result.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result["payloads"], indent=2, ensure_ascii=False))
    print(f"best_tf_iou={result['best_tf_iou']:.4f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

