"""Tier 0 baselines for the delay increment prediction task.

- Persistence: predict a zero increment (delay stays constant). The naive
  lower bound; any useful model must beat this comfortably given the
  target is explicitly the increment, not the raw level.
- Operator's own prediction: derived from `panel_predictions`, the
  operator's live forecast history. For a given row (the vehicle currently
  at some stop with a known observed arrival time), this looks up the most
  recent operator prediction for the *next* stop that was published at or
  before that observation time, and takes its implied increment
  (predicted_arr_delay_next - current_delay). This is the real benchmark:
  "beats the operator's own live forecast" is the headline result.
- Historical mean by (route_id, stop_id, hour, day_of_week): a classical
  baseline, fit strictly on the training split and looked up (with a
  global fallback for unseen combinations) on validation/test.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]


def persistence_predict(df: pl.DataFrame) -> pl.Series:
    return pl.Series("y_pred_persistence", [0.0] * df.height)


def operator_predict(features_path: Path, realised_path: Path, predictions_path: Path) -> pl.DataFrame:
    """Return the input feature rows augmented with the operator's implied increment prediction."""
    con = duckdb.connect()
    query = f"""
    with next_seq as (
        select service_date, trip_id, stop_sequence,
               lead(stop_sequence) over (partition by service_date, trip_id order by stop_sequence) as next_stop_sequence
        from read_parquet('{realised_path}')
    ),
    base as (
        select f.*, ns.next_stop_sequence
        from read_parquet('{features_path}') f
        join next_seq ns
          on ns.service_date = f.service_date
         and ns.trip_id = f.trip_id
         and ns.stop_sequence = f.stop_sequence
    ),
    matched as (
        select
            base.service_date, base.trip_id, base.stop_sequence,
            p.predicted_arr_delay,
            row_number() over (
                partition by base.service_date, base.trip_id, base.stop_sequence
                order by p.poll_ts desc
            ) as rn
        from base
        left join read_parquet('{predictions_path}') p
          on p.service_date = base.service_date
         and p.trip_id = base.trip_id
         and p.stop_sequence = base.next_stop_sequence
         and p.poll_ts <= base.obs_arr_ts
    )
    select service_date, trip_id, stop_sequence, predicted_arr_delay
    from matched
    where rn = 1
    """
    operator_lookup = con.execute(query).pl()
    return operator_lookup


def historical_mean_fit(train: pl.DataFrame) -> pl.DataFrame:
    """Fit the (route_id, stop_id, hour, day_of_week) historical mean increment on the training split."""
    return (
        train.with_columns(pl.col("obs_arr_ts").dt.hour().alias("hour"))
        .group_by(["route_id", "stop_id", "hour", "day_of_week"])
        .agg(pl.col("y_delay_increment").mean().alias("hist_mean_increment"), pl.len().alias("hist_n"))
    )


def historical_mean_predict(df: pl.DataFrame, lookup: pl.DataFrame, global_mean: float) -> pl.Series:
    joined = (
        df.with_columns(pl.col("obs_arr_ts").dt.hour().alias("hour"))
        .join(lookup, on=["route_id", "stop_id", "hour", "day_of_week"], how="left")
        .with_columns(pl.col("hist_mean_increment").fill_null(global_mean))
    )
    return joined["hist_mean_increment"].alias("y_pred_historical_mean")
