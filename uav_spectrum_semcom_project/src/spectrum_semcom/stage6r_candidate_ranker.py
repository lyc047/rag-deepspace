"""Feature and safe selection utilities for the Stage-6R candidate ranker."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .stage6r_context_adapter import robust_spectral_shape_features


def pooled_context_features(
    power_dbm: np.ndarray, output_bins: int = 8
) -> np.ndarray:
    """Pool a normalized spectral-shape vector to a fixed feature width."""
    shape = robust_spectral_shape_features(power_dbm)
    if output_bins < 2 or output_bins > shape.size:
        raise ValueError("output_bins must be between two and channel count")
    edges = np.linspace(0, shape.size, output_bins + 1, dtype=int)
    return np.asarray(
        [np.mean(shape[edges[i] : edges[i + 1]]) for i in range(output_bins)],
        dtype=np.float64,
    )


def candidate_feature_matrix(
    *,
    calibration_regrets: np.ndarray,
    actions: Sequence[Sequence[int]],
    action_maxima: Sequence[int],
    mean_optimal_actions: Sequence[float],
    context_features: np.ndarray,
    epsilon_db: float,
) -> np.ndarray:
    """Build context-action features without reading any future label."""
    regrets = np.asarray(calibration_regrets, dtype=np.float64)
    action_array = np.asarray(actions, dtype=np.float64)
    maxima = np.asarray(action_maxima, dtype=np.float64).reshape(-1)
    mean_optimal = np.asarray(
        mean_optimal_actions, dtype=np.float64
    ).reshape(-1)
    context = np.asarray(context_features, dtype=np.float64).reshape(-1)
    if (
        regrets.ndim != 2
        or regrets.shape[0] < 1
        or regrets.shape[1] != action_array.shape[0]
        or action_array.ndim != 2
        or action_array.shape[1] != maxima.size
        or maxima.shape != mean_optimal.shape
        or context.size < 2
        or np.any(maxima <= 0.0)
        or not np.all(np.isfinite(regrets))
        or not np.all(np.isfinite(action_array))
        or not np.all(np.isfinite(context))
        or not np.isfinite(epsilon_db)
        or epsilon_db < 0.0
    ):
        raise ValueError("invalid candidate feature inputs")
    action_normalized = action_array / maxima
    optimum_normalized = mean_optimal / maxima
    delta = action_normalized - optimum_normalized
    safe = regrets <= float(epsilon_db) + 1e-12
    coverage = np.mean(safe, axis=0)
    mean_regret = np.mean(regrets, axis=0)
    std_regret = np.std(regrets, axis=0)
    maximum_regret = np.max(regrets, axis=0)
    quantile_regret = np.quantile(regrets, 0.9, axis=0)
    repeated_context = np.broadcast_to(
        context, (action_array.shape[0], context.size)
    )
    return np.column_stack(
        (
            repeated_context,
            action_normalized,
            delta,
            coverage,
            mean_regret,
            std_regret,
            maximum_regret,
            quantile_regret,
        )
    ).astype(np.float64)


def future_coverage_labels(
    future_regrets: np.ndarray, epsilon_db: float
) -> np.ndarray:
    regrets = np.asarray(future_regrets, dtype=np.float64)
    if (
        regrets.ndim != 2
        or regrets.shape[0] < 1
        or regrets.shape[1] < 1
        or not np.all(np.isfinite(regrets))
        or not np.isfinite(epsilon_db)
        or epsilon_db < 0.0
    ):
        raise ValueError("invalid future regrets")
    return np.mean(regrets <= float(epsilon_db) + 1e-12, axis=0)


def deterministic_training_subset(
    *,
    calibration_coverage: np.ndarray,
    future_coverage: np.ndarray,
    maximum_rows: int,
) -> np.ndarray:
    """Keep high-value and uniformly spread candidates deterministically."""
    calibration = np.asarray(calibration_coverage, dtype=np.float64).reshape(-1)
    future = np.asarray(future_coverage, dtype=np.float64).reshape(-1)
    if (
        calibration.size < 1
        or calibration.shape != future.shape
        or maximum_rows < 8
        or not np.all(np.isfinite(calibration))
        or not np.all(np.isfinite(future))
    ):
        raise ValueError("invalid candidate subset inputs")
    if calibration.size <= maximum_rows:
        return np.arange(calibration.size, dtype=int)
    third = maximum_rows // 3
    by_calibration = np.argsort(-calibration, kind="stable")[:third]
    by_future = np.argsort(-future, kind="stable")[:third]
    uniform = np.linspace(
        0, calibration.size - 1, maximum_rows - 2 * third, dtype=int
    )
    selected = np.unique(
        np.concatenate((by_calibration, by_future, uniform))
    )
    if selected.size < maximum_rows:
        missing = np.setdiff1d(
            np.arange(calibration.size, dtype=int),
            selected,
            assume_unique=True,
        )
        selected = np.concatenate(
            (selected, missing[: maximum_rows - selected.size])
        )
    return np.sort(selected[:maximum_rows])


def select_ranked_codebook(
    *,
    calibration_regrets: np.ndarray,
    predicted_future_coverage: np.ndarray,
    actions: Sequence[Sequence[int]],
    epsilon_db: float,
    codeword_count: int,
    prediction_weight: float = 0.5,
    redundancy_weight: float = 0.05,
) -> np.ndarray:
    """Select K candidates using calibration gain plus learned future utility."""
    regrets = np.asarray(calibration_regrets, dtype=np.float64)
    prediction = np.clip(
        np.asarray(predicted_future_coverage, dtype=np.float64).reshape(-1),
        0.0,
        1.0,
    )
    action_tuple = tuple(tuple(map(int, row)) for row in actions)
    if (
        regrets.ndim != 2
        or regrets.shape[0] < 1
        or regrets.shape[1] != prediction.size
        or prediction.size != len(action_tuple)
        or codeword_count < 1
        or codeword_count > prediction.size
        or prediction_weight < 0.0
        or redundancy_weight < 0.0
        or not np.all(np.isfinite(regrets))
        or not np.all(np.isfinite(prediction))
    ):
        raise ValueError("invalid ranked-codebook inputs")
    safe = regrets <= float(epsilon_db) + 1e-12
    remaining = np.ones(regrets.shape[0], dtype=bool)
    selected: list[int] = []
    for _ in range(codeword_count):
        best_row = None
        for index in range(prediction.size):
            if index in selected:
                continue
            new_coverage = float(np.mean(remaining & safe[:, index]))
            redundancy = 0.0
            if selected:
                overlaps = []
                for prior in selected:
                    union = np.count_nonzero(
                        safe[:, index] | safe[:, prior]
                    )
                    overlap = np.count_nonzero(
                        safe[:, index] & safe[:, prior]
                    )
                    overlaps.append(overlap / union if union else 1.0)
                redundancy = max(overlaps)
            utility = (
                new_coverage
                + prediction_weight * float(prediction[index])
                - redundancy_weight * redundancy
            )
            row = (
                -utility,
                float(np.mean(regrets[:, index])),
                action_tuple[index],
                index,
            )
            if best_row is None or row < best_row:
                best_row = row
        assert best_row is not None
        chosen = int(best_row[-1])
        selected.append(chosen)
        remaining &= ~safe[:, chosen]
    return np.asarray(selected, dtype=int)

