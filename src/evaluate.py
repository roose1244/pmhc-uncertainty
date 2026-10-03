"""Metrics on the log target and on hours."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from src.target import PSEUDOCOUNT


def hours_from_log(log_half_life: np.ndarray) -> np.ndarray:
    return np.maximum(np.power(10.0, log_half_life) - PSEUDOCOUNT, 0.0)


def regression_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    y_true_log = np.asarray(y_true_log, dtype=np.float64)
    y_pred_log = np.asarray(y_pred_log, dtype=np.float64)
    spearman = spearmanr(y_true_log, y_pred_log).statistic
    mae_hours = float(np.mean(np.abs(hours_from_log(y_pred_log) - hours_from_log(y_true_log))))
    rmse_log = float(np.sqrt(np.mean((y_pred_log - y_true_log) ** 2)))
    return {
        "n": int(len(y_true_log)),
        "spearman": float(spearman),
        "mae_hours": mae_hours,
        "rmse_log": rmse_log,
    }
