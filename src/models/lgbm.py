"""Tier 1 baseline: LightGBM on the tabular feature set.

Feature columns exclude anything that would leak the target or is a bare
identifier (trip_id, stop_id, service_date, obs_arr_ts). All remaining
features are computed strictly before the prediction time, per the
leakage discipline documented in src/build/features.py.
"""

from __future__ import annotations

import lightgbm as lgb
import polars as pl

FEATURE_COLUMNS = [
    "current_delay",
    "delay_prev1",
    "delay_prev2",
    "delay_prev3",
    "delay_slope",
    "stops_remaining",
    "distance_remaining_m",
    "elapsed_share",
    "dwell_planned_s",
    "runtime_planned_next_s",
    "slack_s",
    "route_recent_mean_delay",
    "route_recent_n",
    "segment_recent_mean_delay",
    "upcoming_stop_scheduled_count_5min",
    "day_of_week",
    "is_public_holiday",
    "is_school_holiday",
    "is_peak",
    "precipitation",
    "temperature_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "snowfall",
]
TARGET_COLUMN = "y_delay_increment"


def to_lgb_dataset(df: pl.DataFrame, reference: lgb.Dataset | None = None) -> lgb.Dataset:
    x = df.select(FEATURE_COLUMNS).to_pandas()
    y = df[TARGET_COLUMN].to_pandas()
    if reference is None:
        return lgb.Dataset(x, label=y, feature_name=FEATURE_COLUMNS)
    return lgb.Dataset(x, label=y, feature_name=FEATURE_COLUMNS, reference=reference)


def train_lgbm(train: pl.DataFrame, val: pl.DataFrame, seed: int = 42) -> lgb.Booster:
    train_ds = to_lgb_dataset(train)
    val_ds = to_lgb_dataset(val, reference=train_ds)
    params = {
        # regression_l1 (MAE loss), not the "regression" (L2) default: the
        # target distribution is heavy-tailed (pitfall 7) with a handful of
        # +/-86400s producer-quirk outliers (see notebooks/02_descriptives.ipynb)
        # that dominate a squared-error objective and degrade the bulk fit.
        "objective": "regression_l1",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": seed,
        "verbose": -1,
    }
    booster = lgb.train(
        params,
        train_ds,
        num_boost_round=1000,
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return booster


def predict_lgbm(booster: lgb.Booster, df: pl.DataFrame) -> pl.Series:
    x = df.select(FEATURE_COLUMNS).to_pandas()
    preds = booster.predict(x, num_iteration=booster.best_iteration)
    return pl.Series("y_pred_lgbm", preds)
