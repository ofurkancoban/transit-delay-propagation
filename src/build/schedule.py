"""Parse the archived static GTFS feed into normalised Parquet tables.

Absolute timestamps are not precomputed here: doing so would require
materialising every (trip_id, stop_sequence) row for every valid service
date, which for a Germany-wide feed with ~34M stop_times rows and roughly a
month of validity would multiply into over a billion rows. Instead this
module writes:

- stop_times: trip_id, stop_sequence, stop_id, arr_seconds, dep_seconds
  (seconds since midnight of the service date, GTFS times past 24:00:00
  preserved as-is), dwell_planned_s, runtime_planned_next_s, stop_lat,
  stop_lon.
- trips: trip_id, route_id, service_id (the free gtfs.de feed does not
  publish direction_id or block_id; the vehicle-chain channel and
  direction-aware network features are not available from this feed).
- service_dates: service_id, service_date, expanded from calendar.txt's
  weekday pattern and date range, with calendar_dates.txt exceptions
  (added/removed single dates) applied on top.

realisation.py joins service_dates against stop_times only for the
(trip_id, service_date) pairs actually observed in the realtime lake and
converts those specific rows to absolute UTC timestamps via
src.build.gtfs_time.gtfs_time_to_utc_timestamp.

Run as a script:
    python -m src.build.schedule --static-date 2026-07-23
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]

WEEKDAY_COLUMNS = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("schedule")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(sh)
    return logger


def gtfs_seconds_expr(column: str) -> str:
    """DuckDB SQL expression converting an HH:MM:SS GTFS time column to seconds since midnight."""
    return (
        f"try_cast(split_part({column}, ':', 1) as bigint) * 3600 "
        f"+ try_cast(split_part({column}, ':', 2) as bigint) * 60 "
        f"+ try_cast(split_part({column}, ':', 3) as bigint)"
    )


def build_stop_times(con: duckdb.DuckDBPyConnection, extract_dir: Path, out_dir: Path, logger: logging.Logger) -> int:
    stop_times_csv = extract_dir / "stop_times.txt"
    stops_csv = extract_dir / "stops.txt"

    arr_expr = gtfs_seconds_expr("arrival_time")
    dep_expr = gtfs_seconds_expr("departure_time")

    query = f"""
    with raw as (
        select
            trip_id,
            stop_id,
            stop_sequence,
            {arr_expr} as arr_seconds,
            {dep_expr} as dep_seconds
        from read_csv_auto('{stop_times_csv}', header=true)
    ),
    joined as (
        select
            r.trip_id,
            r.stop_sequence,
            r.stop_id,
            r.arr_seconds,
            r.dep_seconds,
            s.stop_lat,
            s.stop_lon,
            lead(r.arr_seconds) over (partition by r.trip_id order by r.stop_sequence) as next_arr_seconds
        from raw r
        left join read_csv_auto('{stops_csv}', header=true) s using (stop_id)
    )
    select
        trip_id,
        stop_sequence,
        stop_id,
        arr_seconds,
        dep_seconds,
        dep_seconds - arr_seconds as dwell_planned_s,
        next_arr_seconds - dep_seconds as runtime_planned_next_s,
        stop_lat,
        stop_lon
    from joined
    order by trip_id, stop_sequence
    """
    out_path = out_dir / "stop_times.parquet"
    con.execute(f"copy ({query}) to '{out_path}' (format parquet)")
    row_count = con.execute(f"select count(*) from read_parquet('{out_path}')").fetchone()[0]
    logger.info("wrote %s rows to %s", row_count, out_path)
    return row_count


def build_trips(con: duckdb.DuckDBPyConnection, extract_dir: Path, out_dir: Path, logger: logging.Logger) -> int:
    trips_csv = extract_dir / "trips.txt"
    out_path = out_dir / "trips.parquet"
    con.execute(
        f"copy (select * from read_csv_auto('{trips_csv}', header=true)) to '{out_path}' (format parquet)"
    )
    row_count = con.execute(f"select count(*) from read_parquet('{out_path}')").fetchone()[0]
    logger.info("wrote %s rows to %s", row_count, out_path)
    return row_count


def build_service_dates(con: duckdb.DuckDBPyConnection, extract_dir: Path, out_dir: Path, logger: logging.Logger) -> int:
    calendar_csv = extract_dir / "calendar.txt"
    calendar_dates_csv = extract_dir / "calendar_dates.txt"

    weekday_case = " or ".join(
        f"(dayofweek(d) = {i} and {col} = 1)"
        # duckdb dayofweek(): Sunday=0 .. Saturday=6
        for i, col in zip([1, 2, 3, 4, 5, 6, 0], WEEKDAY_COLUMNS)
    )

    query = f"""
    with calendar as (
        select
            service_id,
            monday, tuesday, wednesday, thursday, friday, saturday, sunday,
            strptime(cast(start_date as varchar), '%Y%m%d')::date as start_date,
            strptime(cast(end_date as varchar), '%Y%m%d')::date as end_date
        from read_csv_auto('{calendar_csv}', header=true)
    ),
    expanded as (
        select
            c.service_id,
            d as service_date
        from calendar c,
             generate_series(c.start_date, c.end_date, interval 1 day) as t(d)
        where {weekday_case}
    ),
    exceptions as (
        select
            service_id,
            strptime(cast(date as varchar), '%Y%m%d')::date as service_date,
            exception_type
        from read_csv_auto('{calendar_dates_csv}', header=true)
    ),
    added as (
        select service_id, service_date from exceptions where exception_type = 1
    ),
    removed as (
        select service_id, service_date from exceptions where exception_type = 2
    ),
    combined as (
        select service_id, service_date from expanded
        union
        select service_id, service_date from added
    )
    select c.service_id, c.service_date
    from combined c
    anti join removed r using (service_id, service_date)
    order by service_id, service_date
    """
    out_path = out_dir / "service_dates.parquet"
    con.execute(f"copy ({query}) to '{out_path}' (format parquet)")
    row_count = con.execute(f"select count(*) from read_parquet('{out_path}')").fetchone()[0]
    logger.info("wrote %s rows to %s", row_count, out_path)
    return row_count


def build_schedule(static_date: str, logger: logging.Logger) -> None:
    extract_dir = REPO_ROOT / "data" / "static" / static_date / "extracted"
    if not extract_dir.exists():
        raise FileNotFoundError(f"no archived static feed found at {extract_dir}")

    out_dir = REPO_ROOT / "data" / "interim" / "schedule" / static_date
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    build_trips(con, extract_dir, out_dir, logger)
    build_service_dates(con, extract_dir, out_dir, logger)
    build_stop_times(con, extract_dir, out_dir, logger)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-date",
        required=True,
        help="Date directory under data/static/ to normalise, e.g. 2026-07-23",
    )
    return parser.parse_args()


def main() -> None:
    logger = setup_logging()
    args = parse_args()
    build_schedule(args.static_date, logger)


if __name__ == "__main__":
    main()
