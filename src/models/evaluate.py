"""Shared evaluation utilities for the model comparison table.

Target: the delay increment y_delay_increment (see src/build/features.py),
not the raw delay level. Evaluation metrics per the goal document:

- MAE and RMSE, reported separately by horizon bucket. Since the current
  feature table only implements a k=1 (next stop) target rather than
  arbitrary multi-horizon predictions, "horizon" here is interpreted as
  the scheduled run time to the next stop (`runtime_planned_next_s`),
  bucketed the same way (0-5, 5-15, 15-30, 30-60 minutes). This is stated
  explicitly because it is a real interpretive choice, not the literal
  poll-to-event horizon used in panel_predictions.
- PR-AUC for "delay exceeds 6 minutes at the target stop" (the class is
  imbalanced, see notebooks/02_descriptives.ipynb).
- Skill score against the operator benchmark: 1 - MAE_model / MAE_operator.
- Breakdown by peak vs off-peak and by a stop-density proxy for
  urban vs rural, since no direct urban/rural label exists in GTFS.

Splitting is always by time (sorted by obs_arr_ts), never randomly, per the
goal document's working agreement.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score

HORIZON_BUCKETS = [(0, 5), (5, 15), (15, 30), (30, 60), (60, None)]
DELAY_THRESHOLD_S = 360


def time_based_split(df: pl.DataFrame, time_col: str = "obs_arr_ts", train_frac: float = 0.7, val_frac: float = 0.15) -> dict[str, pl.DataFrame]:
    """Split strictly by time order, never randomly."""
    df = df.sort(time_col)
    n = df.height
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return {
        "train": df.slice(0, train_end),
        "val": df.slice(train_end, val_end - train_end),
        "test": df.slice(val_end, n - val_end),
    }


def horizon_bucket_label(minutes: float) -> str:
    for lo, hi in HORIZON_BUCKETS:
        if hi is None and minutes >= lo:
            return f">= {lo} min"
        if hi is not None and lo <= minutes < hi:
            return f"{lo}-{hi} min"
    return "unknown"


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_by_horizon_bucket(df: pl.DataFrame, y_true_col: str, y_pred_col: str, horizon_seconds_col: str = "runtime_planned_next_s") -> pl.DataFrame:
    rows = []
    horizon_min = df[horizon_seconds_col].fill_null(0) / 60
    df = df.with_columns(horizon_min.alias("_horizon_min"))
    for lo, hi in HORIZON_BUCKETS:
        if hi is None:
            mask = df["_horizon_min"] >= lo
            label = f">= {lo} min"
        else:
            mask = (df["_horizon_min"] >= lo) & (df["_horizon_min"] < hi)
            label = f"{lo}-{hi} min"
        sub = df.filter(mask)
        if sub.height == 0:
            continue
        y_true = sub[y_true_col].to_numpy()
        y_pred = sub[y_pred_col].to_numpy()
        rows.append({"horizon_bucket": label, "n": sub.height, "mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)})
    return pl.DataFrame(rows)


def pr_auc_delay_exceeds_threshold(df: pl.DataFrame, current_delay_col: str, y_pred_col: str, y_true_col: str = "y_delay_increment") -> float:
    """PR-AUC for the binary event that the target stop's absolute delay exceeds 6 minutes.

    The predicted score is the model's implied predicted absolute delay
    (current_delay + predicted increment); the label is whether the actual
    absolute delay (current_delay + true increment) exceeds the threshold.
    """
    actual_next_delay = df[current_delay_col] + df[y_true_col]
    predicted_next_delay = df[current_delay_col] + df[y_pred_col]
    labels = (actual_next_delay.abs() > DELAY_THRESHOLD_S).to_numpy().astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    scores = predicted_next_delay.abs().to_numpy()
    return float(average_precision_score(labels, scores))


def skill_score(mae_model: float, mae_operator: float) -> float:
    if mae_operator == 0:
        return float("nan")
    return 1.0 - mae_model / mae_operator


def evaluate_by_group(df: pl.DataFrame, group_col: str, y_true_col: str, y_pred_col: str) -> pl.DataFrame:
    rows = []
    for group_value in df[group_col].unique().sort().to_list():
        sub = df.filter(pl.col(group_col) == group_value)
        if sub.height == 0:
            continue
        y_true = sub[y_true_col].to_numpy()
        y_pred = sub[y_pred_col].to_numpy()
        rows.append({group_col: group_value, "n": sub.height, "mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred)})
    return pl.DataFrame(rows)


def build_comparison_table(model_results: dict[str, pl.DataFrame], operator_mae_by_bucket: dict[str, float]) -> pl.DataFrame:
    """model_results: {model_name: evaluate_by_horizon_bucket(...) output}."""
    rows = []
    for model_name, table in model_results.items():
        for row in table.iter_rows(named=True):
            op_mae = operator_mae_by_bucket.get(row["horizon_bucket"])
            skill = skill_score(row["mae"], op_mae) if op_mae else None
            rows.append({
                "model": model_name,
                "horizon_bucket": row["horizon_bucket"],
                "n": row["n"],
                "mae": row["mae"],
                "rmse": row["rmse"],
                "skill_vs_operator": skill,
            })
    return pl.DataFrame(rows)
