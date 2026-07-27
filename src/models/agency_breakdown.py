"""Per-agency (company/region) breakdown: descriptive delay stats and model performance.

The primary feed is Germany-wide, aggregating ~470 distinct transit
agencies (see `data/static/<date>/extracted/agency.txt`). There is no
single "city" served, and GTFS has no structured city field, but most
large agencies are themselves named after the metro region they serve
(e.g. "Hamburger Verkehrsverbund", "Verkehrsverbund Stuttgart",
"Berliner Verkehrsbetriebe"), so agency-level grouping doubles as a
practical city/region-level breakdown for the largest operators.

Two outputs:

- `agency_delay_summary.parquet`: descriptive realised-delay stats per
  agency (mean/median delay, share exceeding 6 minutes), joined via
  `routes.txt.agency_id -> agency.txt.agency_name`. No model involved,
  cheap to compute from `panel_realised.parquet` alone.
- `model_breakdown_agency.parquet`: LightGBM MAE, operator MAE, and
  skill score per agency on the test split, requiring a fresh LightGBM
  fit (the trained booster is not persisted elsewhere in this repo).

Run as a script:
    python -m src.models.agency_breakdown --static-date 2026-07-23
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb
import polars as pl

from src.models.baselines import operator_predict
from src.models.evaluate import mae, pr_auc_delay_exceeds_threshold, time_based_split
from src.models.lgbm import predict_lgbm, train_lgbm

REPO_ROOT = Path(__file__).resolve().parents[2]
MIN_ROWS_PER_AGENCY = 3000


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("agency_breakdown")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(sh)
    return logger


def load_route_agency_map(static_date: str) -> pl.DataFrame:
    extract_dir = REPO_ROOT / "data" / "static" / static_date / "extracted"
    con = duckdb.connect()
    return con.execute(
        f"""
        select r.route_id, coalesce(a.agency_name, 'unknown') as agency_name
        from read_csv_auto('{extract_dir / "routes.txt"}') r
        left join read_csv_auto('{extract_dir / "agency.txt"}') a using (agency_id)
        """
    ).pl().with_columns(pl.col("route_id").cast(pl.Utf8))


def build_delay_summary(realised_path: Path, route_agency: pl.DataFrame, logger: logging.Logger) -> pl.DataFrame:
    realised = pl.read_parquet(realised_path).with_columns(pl.col("route_id").cast(pl.Utf8))
    joined = realised.join(route_agency, on="route_id", how="left").with_columns(pl.col("agency_name").fill_null("unknown"))
    summary = (
        joined.filter(pl.col("realised_arr_delay").is_not_null())
        .group_by("agency_name")
        .agg(
            pl.len().alias("n_stop_events"),
            pl.col("trip_id").n_unique().alias("n_trips"),
            pl.col("realised_arr_delay").mean().alias("mean_delay_s"),
            pl.col("realised_arr_delay").median().alias("median_delay_s"),
            (pl.col("realised_arr_delay").abs() > 360).mean().alias("share_over_6min"),
        )
        .filter(pl.col("n_stop_events") >= MIN_ROWS_PER_AGENCY)
        .sort("n_stop_events", descending=True)
    )
    logger.info("delay summary: %d agencies with >= %d stop-events", summary.height, MIN_ROWS_PER_AGENCY)
    return summary


def build_model_breakdown(features_path: Path, realised_path: Path, predictions_path: Path, route_agency: pl.DataFrame, logger: logging.Logger) -> pl.DataFrame:
    features = pl.read_parquet(features_path)
    splits = time_based_split(features)
    train, val, test = splits["train"], splits["val"], splits["test"]

    logger.info("training lgbm for agency breakdown")
    booster = train_lgbm(train, val)
    test = test.with_columns(predict_lgbm(booster, test))

    test_keys_path = features_path.parent / "_agency_breakdown_test_keys.parquet"
    test.write_parquet(test_keys_path)
    op_lookup = operator_predict(str(test_keys_path), str(realised_path), str(predictions_path))
    test_keys_path.unlink(missing_ok=True)

    test = test.join(op_lookup, on=["service_date", "trip_id", "stop_sequence"], how="left")
    test = test.with_columns((pl.col("predicted_arr_delay") - pl.col("current_delay")).alias("y_pred_operator"))
    test = test.with_columns(pl.col("y_pred_operator").fill_null(0.0))
    test = test.with_columns(pl.col("route_id").cast(pl.Utf8)).join(route_agency, on="route_id", how="left")
    test = test.with_columns(pl.col("agency_name").fill_null("unknown"))

    rows = []
    for (agency,), group in test.group_by(["agency_name"]):
        if group.height < MIN_ROWS_PER_AGENCY:
            continue
        y_true = group["y_delay_increment"].to_numpy()
        y_lgbm = group["y_pred_lgbm"].to_numpy()
        y_op = group["y_pred_operator"].to_numpy()
        lgbm_mae = mae(y_true, y_lgbm)
        op_mae = mae(y_true, y_op)
        rows.append({
            "agency": agency,
            "n": group.height,
            "lgbm_mae": lgbm_mae,
            "operator_mae": op_mae,
            "skill_vs_operator": (1 - lgbm_mae / op_mae) if op_mae else None,
            "pr_auc": pr_auc_delay_exceeds_threshold(group, "current_delay", "y_pred_lgbm"),
        })

    result = pl.DataFrame(rows).sort("n", descending=True)
    logger.info("model breakdown: %d agencies with >= %d test rows", result.height, MIN_ROWS_PER_AGENCY)
    return result


def main() -> None:
    logger = setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-date", required=True)
    args = parser.parse_args()

    processed_dir = REPO_ROOT / "data" / "processed"
    route_agency = load_route_agency_map(args.static_date)

    delay_summary = build_delay_summary(processed_dir / "panel_realised.parquet", route_agency, logger)
    delay_summary.write_parquet(processed_dir / "agency_delay_summary.parquet")

    model_breakdown = build_model_breakdown(
        processed_dir / "features.parquet",
        processed_dir / "panel_realised.parquet",
        processed_dir / "panel_predictions.parquet",
        route_agency,
        logger,
    )
    model_breakdown.write_parquet(processed_dir / "model_breakdown_agency.parquet")

    logger.info("wrote agency_delay_summary.parquet and model_breakdown_agency.parquet")


if __name__ == "__main__":
    main()
