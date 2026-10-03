"""Ensemble dispersion and split-conformal intervals. Calib is never used to train."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.evaluate import hours_from_log

ALPHA = 0.10


def conformal_quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
    """Finite-sample split-conformal quantile. Level is ceil((n+1)(1-alpha))/n."""
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        raise ValueError("calibration scores are empty")
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        return float("inf")
    return float(np.quantile(scores, level, method="higher"))


def sigma_floor(val_std: np.ndarray) -> float:
    floor = float(np.quantile(np.asarray(val_std, dtype=np.float64), 0.01))
    return max(floor, 1e-6)


def apply_intervals(
    frame: pd.DataFrame,
    q_norm: float,
    q_const: float,
    floor: float,
) -> pd.DataFrame:
    out = frame.copy()
    sigma = np.maximum(out["y_std"].to_numpy(dtype=np.float64), floor)
    mu = out["y_mean"].to_numpy(dtype=np.float64)
    out["lo"] = mu - q_norm * sigma
    out["hi"] = mu + q_norm * sigma
    out["lo_const"] = mu - q_const
    out["hi_const"] = mu + q_const
    out["lo_hours"] = hours_from_log(out["lo"].to_numpy())
    out["hi_hours"] = hours_from_log(out["hi"].to_numpy())
    return out


def fit_intervals(preds: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    val = preds.loc[preds["fold"] == "val"]
    calib = preds.loc[preds["fold"] == "calib"]
    floor = sigma_floor(val["y_std"].to_numpy())
    mu = calib["y_mean"].to_numpy(dtype=np.float64)
    y = calib["y_true"].to_numpy(dtype=np.float64)
    sigma = np.maximum(calib["y_std"].to_numpy(dtype=np.float64), floor)
    q_norm = conformal_quantile(np.abs(y - mu) / sigma)
    q_const = conformal_quantile(np.abs(y - mu))
    scored = apply_intervals(preds, q_norm, q_const, floor)
    return scored, {"q_norm": q_norm, "q_const": q_const, "sigma_floor": floor}


def _covered(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = ALPHA) -> float:
    width = hi - lo
    below = np.maximum(lo - y, 0.0)
    above = np.maximum(y - hi, 0.0)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def selective_mae(y_true_log: np.ndarray, y_pred_log: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """MAE in hours as we keep the most certain predictions. Low score = more certain."""
    order = np.argsort(score, kind="mergesort")
    y = hours_from_log(y_true_log)[order]
    yhat = hours_from_log(y_pred_log)[order]
    n = len(y)
    out = {}
    for fraction in (1.0, 0.8, 0.5, 0.2):
        keep = max(int(round(n * fraction)), 1)
        out[f"mae_{int(fraction * 100)}"] = float(np.mean(np.abs(yhat[:keep] - y[:keep])))
    return out


def uncertainty_report(preds: pd.DataFrame, split: str) -> tuple[pd.DataFrame, dict]:
    scored, params = fit_intervals(preds)
    test = scored.loc[scored["fold"] == "test"]
    val = scored.loc[scored["fold"] == "val"]
    y = test["y_true"].to_numpy(dtype=np.float64)
    mu = test["y_mean"].to_numpy(dtype=np.float64)
    err = np.abs(y - mu)
    std = test["y_std"].to_numpy(dtype=np.float64)
    val_err = np.abs(val["y_true"].to_numpy() - val["y_mean"].to_numpy())
    report = {
        "split": split,
        **params,
        "spearman": float(spearmanr(y, mu).statistic),
        "err_unc_spearman_test": float(spearmanr(std, err).statistic),
        "err_unc_spearman_val": float(spearmanr(val["y_std"], val_err).statistic),
        "coverage_90": _covered(y, test["lo"].to_numpy(), test["hi"].to_numpy()),
        "coverage_90_const": _covered(y, test["lo_const"].to_numpy(), test["hi_const"].to_numpy()),
        "interval_score_log": interval_score(y, test["lo"].to_numpy(), test["hi"].to_numpy()),
        "interval_score_const": interval_score(
            y, test["lo_const"].to_numpy(), test["hi_const"].to_numpy()
        ),
        "median_width_hours": float(np.median(test["hi_hours"] - test["lo_hours"])),
        "median_width_const_hours": float(
            np.median(hours_from_log(test["hi_const"].to_numpy()) - hours_from_log(test["lo_const"].to_numpy()))
        ),
    }
    report.update(selective_mae(y, mu, std))
    oracle = selective_mae(y, mu, err)
    report["oracle_mae_50"] = oracle["mae_50"]
    rng = np.random.default_rng(0)
    report["random_mae_50"] = selective_mae(y, mu, rng.random(len(y)))["mae_50"]
    return scored, report
