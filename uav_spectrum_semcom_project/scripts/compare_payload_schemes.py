from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
BASELINE_PATH = PROJECT_DIR / "results" / "phase1" / "frame_baseline_tuned_val_test.json"
CNN_PATH = PROJECT_DIR / "results" / "phase1" / "frame_cnn_train_val_test.json"
OUT_CSV = PROJECT_DIR / "results" / "phase1" / "payload_scheme_comparison.csv"
OUT_MD = PROJECT_DIR / "results" / "phase1" / "payload_scheme_comparison.md"

import sys

sys.path.insert(0, str(PROJECT_DIR / "src"))

from spectrum_semcom.bit_budget import (
    BitBudget,
    feature_token_packet_bits,
    hard_box_packet_bits,
    semantic_packet_bits,
    soft_box_packet_bits,
)
from spectrum_semcom.types import SemanticPacket


def _fmt_bps(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.3f} Gbps"
    if value >= 1e6:
        return f"{value / 1e6:.3f} Mbps"
    if value >= 1e3:
        return f"{value / 1e3:.3f} kbps"
    return f"{value:.3f} bps"


def _rows_from_results() -> list[dict]:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    cnn = json.loads(CNN_PATH.read_text(encoding="utf-8"))

    sources = [
        ("tuned_traditional", baseline["test_rows"]),
        ("frame_cnn", cnn["test"]["rows"]),
    ]
    rows = []
    for method, eval_rows in sources:
        for row in eval_rows:
            n_pred = int(row["n_pred"])
            duration_s = float(row["duration_s"]) if "duration_s" in row else 0.01
            semantic_bits = int(row.get("semantic_packet_bits_per_frame", semantic_packet_bits(SemanticPacket(0, row["frame_id"], []))))
            schemes = {
                "hard_box": hard_box_packet_bits(n_pred),
                "soft_box_8class": soft_box_packet_bits(n_pred, n_classes=8, prob_bits=8),
                "semantic_box_current": semantic_bits,
                "feature_token_16d_6bit": feature_token_packet_bits(n_tokens=max(1, n_pred), token_dim=16, bits_per_value=6),
                "feature_token_32d_8bit": feature_token_packet_bits(n_tokens=max(1, n_pred), token_dim=32, bits_per_value=8),
            }
            for scheme, bits in schemes.items():
                rows.append(
                    {
                        "method": method,
                        "frame_id": row["frame_id"],
                        "scheme": scheme,
                        "n_pred": n_pred,
                        "bits_per_frame": bits,
                        "bitrate_bps": BitBudget(scheme, bits, duration_s).bitrate_bps,
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "f1": row["f1"],
                    }
                )
    return rows


def _aggregate(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (row["method"], row["scheme"])
        groups.setdefault(key, []).append(row)
    out = []
    for (method, scheme), items in sorted(groups.items()):
        out.append(
            {
                "method": method,
                "scheme": scheme,
                "n_frames": len(items),
                "mean_bits_per_frame": sum(item["bits_per_frame"] for item in items) / len(items),
                "mean_bitrate_bps": sum(item["bitrate_bps"] for item in items) / len(items),
                "mean_n_pred": sum(item["n_pred"] for item in items) / len(items),
                "mean_frame_f1": sum(item["f1"] for item in items) / len(items),
            }
        )
    return out


def main() -> None:
    rows = _rows_from_results()
    agg = _aggregate(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        writer.writeheader()
        writer.writerows(agg)

    lines = [
        "# Payload Scheme Comparison",
        "",
        "## 1. 对比说明",
        "",
        "本表不改变检测结果，只比较同一组预测框在不同上报 payload 表示下的通信开销。",
        "",
        "| 方法 | Payload 方案 | 平均预测框数 | 平均 bit/frame | 平均速率 | 平均 frame F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in agg:
        lines.append(
            f"| {item['method']} | `{item['scheme']}` | {item['mean_n_pred']:.2f} | "
            f"{item['mean_bits_per_frame']:.1f} | {_fmt_bps(item['mean_bitrate_bps'])} | {item['mean_frame_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 2. 阶段性结论",
            "",
            "1. `hard_box` 是最低开销方案，但只适合地面端不需要复核、不需要类别概率的场景。",
            "2. `soft_box_8class` 增加类别概率，适合多 UAV 融合时做软信息融合。",
            "3. `feature_token` 开销高于纯候选框，但仍是 kbps 级，后续可作为语义通信主线。",
            "4. 当前结果再次说明：即便上传更丰富的语义 token，速率仍远低于 12+12 bit I/Q 的 184.32 Mbps。",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()

