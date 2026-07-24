"""Build the realisation panel by joining the static schedule against the realtime lake.

Two output tables, per GOAL_transit_delay_propagation.md section 2:

- panel_realised: one row per (service_date, trip_id, stop_sequence), carrying
  the last observed delay before the scheduled time passed.
- panel_predictions: one row per (service_date, trip_id, stop_sequence, poll_ts),
  the full prediction history keyed by horizon, used as the operator benchmark.

Scheduled absolute timestamps are computed only for the (service_date, trip_id,
stop_sequence) combinations actually observed in the realtime lake, not for
every valid service date of every trip, since materialising the latter would
be enormous. The service_date used for each observed trip instance comes
directly from GTFS-RT's own TripDescriptor.start_date field, not re-derived
from the static calendar; pitfall 1 (stale static join) is guarded by
assert_static_version_covers_dates, which fails loudly if any observed
service_date falls outside the static feed's manifest validity window.

Performance note: this first pass favours correctness over throughput. The
DST-aware local-to-UTC conversion (src.build.gtfs_time.gtfs_time_to_utc_timestamp)
runs once per distinct (service_date, seconds_since_midnight) pair rather than
per row, which is enough to keep this fast at the data volumes seen so far.
Revisit if/when the lake grows to the point this becomes a bottleneck.

Run as a script:
    python -m src.build.realisation --static-date 2026-07-23 --rt-glob "data/rt/date=*/hour=*/*.parquet"
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import polars as pl

from src.build.gtfs_time import gtfs_time_to_utc_timestamp

REPO_ROOT = Path(__file__).resolve().parents[2]
BERLIN = ZoneInfo("Europe/Berlin")


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("realisation")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(sh)
    return logger


def load_manifest(static_date: str) -> dict:
    manifest_path = REPO_ROOT / "data" / "static" / static_date / "manifest.json"
    with open(manifest_path) as f:
        return json.load(f)


def assert_static_version_covers_dates(observed_dates: list[datetime.date], manifest: dict) -> None:
    """Fail loudly if any observed service date falls outside the static feed's validity window.

    This guards against joining realtime data from one week against a stale
    static schedule version from another (pitfall 1 in the goal document).
    """
    start = datetime.datetime.strptime(manifest["feed_start_date"], "%Y%m%d").date()
    end = datetime.datetime.strptime(manifest["feed_end_date"], "%Y%m%d").date()
    bad = [d for d in observed_dates if not (start <= d <= end)]
    if bad:
        raise RuntimeError(
            f"observed service date(s) {sorted(set(bad))} fall outside the static feed's "
            f"validity window [{start}, {end}]; use a static snapshot whose validity window "
            f"contains these service dates instead"
        )


def add_sched_timestamps(df: pl.DataFrame, seconds_col: str, out_col: str) -> pl.DataFrame:
    """Convert (service_date, seconds_since_midnight) into an absolute UTC timestamp column.

    Computed once per distinct (service_date, seconds) pair to avoid a
    per-row Python conversion cost, then joined back onto the full frame.
    """
    distinct_pairs = df.select("service_date", seconds_col).unique()
    conversions = [
        gtfs_time_to_utc_timestamp(
            row["service_date"],
            seconds_to_gtfs_time_string(row[seconds_col]),
            BERLIN,
        )
        for row in distinct_pairs.iter_rows(named=True)
    ]
    distinct_pairs = distinct_pairs.with_columns(pl.Series(out_col, conversions))
    return df.join(distinct_pairs, on=["service_date", seconds_col], how="left")


def seconds_to_gtfs_time_string(seconds: int) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_schedule(con: duckdb.DuckDBPyConnection, static_date: str) -> None:
    schedule_dir = REPO_ROOT / "data" / "interim" / "schedule" / static_date
    con.execute(f"create or replace view stop_times as select * from read_parquet('{schedule_dir}/stop_times.parquet')")
    con.execute(f"create or replace view trips as select * from read_parquet('{schedule_dir}/trips.parquet')")


def build_panel_predictions(
    con: duckdb.DuckDBPyConnection, rt_glob: str, manifest: dict, logger: logging.Logger
) -> pl.DataFrame:
    raw = con.execute(
        f"""
        select
            strptime(r.start_date, '%Y%m%d')::date as service_date,
            r.trip_id,
            t.route_id,
            r.stop_sequence,
            r.poll_ts,
            st.arr_seconds,
            r.arrival_delay as predicted_arr_delay
        from read_parquet('{rt_glob}') r
        join stop_times st using (trip_id, stop_sequence)
        left join trips t using (trip_id)
        where r.schedule_relationship = 'SCHEDULED'
          and r.arrival_delay is not null
          and r.start_date is not null
        """
    ).pl()
    logger.info("panel_predictions: %s raw joined rows before timestamp conversion", raw.height)

    assert_static_version_covers_dates(raw["service_date"].unique().to_list(), manifest)

    raw = add_sched_timestamps(raw, "arr_seconds", "sched_arr_ts")
    raw = raw.with_columns(pl.col("poll_ts").dt.convert_time_zone("UTC"))
    raw = raw.with_columns(
        (pl.col("sched_arr_ts") - pl.col("poll_ts")).dt.total_seconds().alias("horizon_s")
    ).filter(pl.col("horizon_s") > 0)

    return raw.select(
        "service_date", "trip_id", "route_id", "stop_sequence", "poll_ts", "horizon_s", "predicted_arr_delay"
    )


def build_panel_realised(
    con: duckdb.DuckDBPyConnection, rt_glob: str, manifest: dict, logger: logging.Logger
) -> pl.DataFrame:
    raw = con.execute(
        f"""
        with joined as (
            select
                strptime(r.start_date, '%Y%m%d')::date as service_date,
                r.trip_id,
                t.route_id,
                r.direction_id,
                r.vehicle_id,
                r.stop_id,
                r.stop_sequence,
                r.poll_ts,
                st.arr_seconds,
                st.dep_seconds,
                st.dwell_planned_s,
                st.runtime_planned_next_s,
                st.stop_lat,
                st.stop_lon,
                r.arrival_delay,
                r.departure_delay
            from read_parquet('{rt_glob}') r
            join stop_times st using (trip_id, stop_sequence)
            left join trips t using (trip_id)
            where r.schedule_relationship = 'SCHEDULED'
              and r.start_date is not null
        )
        select * from joined
        """
    ).pl()
    logger.info("panel_realised: %s raw joined rows before timestamp conversion", raw.height)

    assert_static_version_covers_dates(raw["service_date"].unique().to_list(), manifest)

    raw = add_sched_timestamps(raw, "arr_seconds", "sched_arr_ts")
    raw = add_sched_timestamps(raw, "dep_seconds", "sched_dep_ts")
    raw = raw.with_columns(pl.col("poll_ts").dt.convert_time_zone("UTC"))

    # Last observed value strictly before the scheduled arrival time passed.
    before_sched = raw.filter(pl.col("poll_ts") <= pl.col("sched_arr_ts"))
    last_per_stop = (
        before_sched.sort("poll_ts")
        .group_by(["service_date", "trip_id", "stop_sequence"], maintain_order=True)
        .last()
    )

    return last_per_stop.select(
        "service_date",
        "trip_id",
        "route_id",
        "direction_id",
        "vehicle_id",
        "stop_id",
        "stop_sequence",
        "sched_arr_ts",
        "sched_dep_ts",
        pl.col("arrival_delay").alias("realised_arr_delay"),
        pl.col("departure_delay").alias("realised_dep_delay"),
        "dwell_planned_s",
        "runtime_planned_next_s",
        "stop_lat",
        "stop_lon",
    )


def build_realisation(static_date: str, rt_glob: str, logger: logging.Logger) -> None:
    manifest = load_manifest(static_date)

    con = duckdb.connect()
    load_schedule(con, static_date)

    out_dir = REPO_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = build_panel_predictions(con, rt_glob, manifest, logger)
    predictions.write_parquet(out_dir / "panel_predictions.parquet")
    logger.info("wrote %s rows to panel_predictions.parquet", predictions.height)

    realised = build_panel_realised(con, rt_glob, manifest, logger)
    realised.write_parquet(out_dir / "panel_realised.parquet")
    logger.info("wrote %s rows to panel_realised.parquet", realised.height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-date", required=True, help="Static feed version to join against, e.g. 2026-07-23")
    parser.add_argument(
        "--rt-glob",
        default="data/rt/date=*/hour=*/*.parquet",
        help="Glob pattern (relative to repo root) selecting realtime snapshot files",
    )
    return parser.parse_args()


def main() -> None:
    logger = setup_logging()
    args = parse_args()
    build_realisation(args.static_date, str(REPO_ROOT / args.rt_glob), logger)


if __name__ == "__main__":
    main()
