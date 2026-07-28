from __future__ import annotations

from dataclasses import dataclass

from .types import SignalBox


def interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    left = max(min(a0, a1), min(b0, b1))
    right = min(max(a0, a1), max(b0, b1))
    inter = max(0.0, right - left)
    union = max(max(a0, a1), max(b0, b1)) - min(min(a0, a1), min(b0, b1))
    return inter / union if union > 0 else 0.0


def time_frequency_iou(a: SignalBox, b: SignalBox) -> float:
    t_iou = interval_iou(a.t_start_s, a.t_end_s, b.t_start_s, b.t_end_s)
    f_iou = interval_iou(a.f_low_hz, a.f_high_hz, b.f_low_hz, b.f_high_hz)
    return t_iou * f_iou


def best_box_iou(pred: list[SignalBox], truth: list[SignalBox]) -> float:
    if not pred or not truth:
        return 0.0
    return max(time_frequency_iou(p, g) for p in pred for g in truth)


@dataclass(frozen=True)
class DetectionMetrics:
    n_pred: int
    n_truth: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    mean_matched_iou: float
    best_iou: float


def match_boxes(
    pred: list[SignalBox],
    truth: list[SignalBox],
    iou_threshold: float = 0.1,
) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching between predicted and ground-truth boxes."""

    candidates: list[tuple[float, int, int]] = []
    for pred_idx, pred_box in enumerate(pred):
        for truth_idx, truth_box in enumerate(truth):
            iou = time_frequency_iou(pred_box, truth_box)
            if iou >= iou_threshold:
                candidates.append((iou, pred_idx, truth_idx))

    candidates.sort(reverse=True)
    used_pred: set[int] = set()
    used_truth: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, pred_idx, truth_idx in candidates:
        if pred_idx in used_pred or truth_idx in used_truth:
            continue
        used_pred.add(pred_idx)
        used_truth.add(truth_idx)
        matches.append((pred_idx, truth_idx, iou))
    return matches


def evaluate_detections(
    pred: list[SignalBox],
    truth: list[SignalBox],
    iou_threshold: float = 0.1,
) -> DetectionMetrics:
    matches = match_boxes(pred, truth, iou_threshold=iou_threshold)
    tp = len(matches)
    fp = max(0, len(pred) - tp)
    fn = max(0, len(truth) - tp)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    mean_iou = sum(iou for _, _, iou in matches) / tp if tp else 0.0
    return DetectionMetrics(
        n_pred=len(pred),
        n_truth=len(truth),
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_matched_iou=mean_iou,
        best_iou=best_box_iou(pred, truth),
    )
