from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from spectrum_semcom.baselines import energy_detector
from spectrum_semcom.bit_budget import BitBudget, iq_payload_bits, semantic_packet_bits, stft_payload_bits
from spectrum_semcom.data_io import load_sigmf_frame
from spectrum_semcom.metrics import best_box_iou
from spectrum_semcom.preprocessing import stft_power
from spectrum_semcom.types import SemanticPacket


def main() -> None:
    meta_path = (
        PROJECT_DIR
        / "data"
        / "raw"
        / "sigmf_5g_short"
        / "1876954_7680KSPS_srsRAN_Project_gnb_short.sigmf-meta"
    )
    frame = load_sigmf_frame(meta_path, start_sample=0, max_samples=None, normalize=True)
    stft = stft_power(frame.iq, frame.sample_rate_hz, n_fft=1024, hop_length=256)
    pred_boxes = energy_detector(stft)
    packet = SemanticPacket(uav_id=0, frame_id=frame.frame_id, boxes=pred_boxes, metadata_bits=160)

    budgets = [
        BitBudget("raw_iq_32bit_iq_from_sigmf", iq_payload_bits(frame.n_samples, 32, 32), frame.duration_s),
        BitBudget("raw_iq_12bit_iq_if_requantized", iq_payload_bits(frame.n_samples, 12, 12), frame.duration_s),
        BitBudget("stft_8bit", stft_payload_bits(stft.power_db.shape, 8), frame.duration_s),
        BitBudget("semantic_packet_energy_baseline", semantic_packet_bits(packet), frame.duration_s),
    ]

    label_counts = Counter(box.label for box in frame.boxes)
    result = {
        "source": "Daniel Estévez annotated 5G SigMF recording",
        "frame_id": frame.frame_id,
        "n_samples": frame.n_samples,
        "duration_s": frame.duration_s,
        "sample_rate_hz": frame.sample_rate_hz,
        "center_freq_hz": frame.center_freq_hz,
        "metadata": frame.metadata,
        "annotation_count": len(frame.boxes),
        "annotation_label_counts": dict(sorted(label_counts.items())),
        "stft_shape": list(stft.power_db.shape),
        "energy_pred_boxes": [box.__dict__ for box in pred_boxes],
        "best_tf_iou_against_annotations": best_box_iou(pred_boxes, frame.boxes),
        "payloads": [
            {"name": b.name, "bits_per_frame": b.bits_per_frame, "bitrate_bps": b.bitrate_bps}
            for b in budgets
        ],
    }

    out_dir = PROJECT_DIR / "results" / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "real_sigmf_smoke_result.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: result[k] for k in ["n_samples", "duration_s", "annotation_count", "stft_shape"]}, indent=2))
    print(json.dumps(result["payloads"], indent=2, ensure_ascii=False))
    print(f"best_tf_iou_against_annotations={result['best_tf_iou_against_annotations']:.4f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

