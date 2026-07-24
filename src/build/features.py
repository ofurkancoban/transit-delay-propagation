"""Build the feature table for the delay increment prediction task.

Target: y = realised_dep_delay(s+k) - realised_dep_delay(s), the delay
increment between consecutive stops, not the raw delay level (raw delay is
almost entirely autoregressive and predicting it yields a meaningless R2
above 0.95 while learning nothing beyond "delay stays the same").

Leakage discipline is the single most important requirement here. Every
feature is computed strictly from information available before the
prediction time, defined as the actual observed arrival time at the current
stop (obs_arr_ts = sched_arr_ts + realised_arr_delay), not the scheduled
time. Network-state aggregates use DuckDB window frames of the form
`RANGE BETWEEN INTERVAL 15 MINUTES PRECEDING AND INTERVAL 1 SECOND
PRECEDING`, ordered by obs_arr_ts, which by construction cannot see the
current row or anything after it. The upcoming-stop scheduling pressure
feature is computed purely from the static schedule (arr_seconds is known
in advance for every trip), so it carries no realtime leakage risk either.

Two known simplifications, both documented rather than hidden:

- `slack` (planned run time minus the historical observed run time for that
  segment) uses an expanding-window average ordered by obs_arr_ts rather
  than a true historical baseline, since only a single day of realised data
  exists so far. Revisit once enough days have accrued to hold out a
  training-only baseline window.
- The vehicle-chain feature block from the goal document is entirely
  skipped: `vehicle_id` is never populated on the primary feed (see
  README.md pitfall 4), so no vehicle-chain feature is estimable.

Volatility features (`delay_jump_1`, `delay_jump_1_abs`, `delay_jump_2`,
`delay_recent_std`, `delay_vs_route_recent_gap`) were added after a
diagnostic in notebooks/03_model_results.ipynb found LightGBM's error is
concentrated (>10x the operator's) on rows where the operator's own
prediction had just changed, i.e. a currently-unfolding delay event.
These features, all derivable from the TripUpdates already collected (no
new data source), give a small, real, but not decisive improvement
(~0.5% overall MAE, and only ~0.1% on the specific volatile-event rows
that motivated them): they help the model use its existing recent-delay
history more effectively, but do not substitute for the live vehicle
position information the operator evidently has and this feed does not
publish (confirmed empirically: a live pull of the primary feed contains
zero VehiclePosition entities).

Run as a script:
    python -m src.build.features --static-date 2026-07-23
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

import duckdb
import holidays

REPO_ROOT = Path(__file__).resolve().parents[2]
GRID_RESOLUTION_DEGREES = 0.25
PEAK_HOURS = {(6, 9), (16, 19)}


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("features")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(sh)
    return logger


def load_school_holiday_periods(path: Path) -> list[tuple[datetime.date, datetime.date]]:
    with open(path) as f:
        data = json.load(f)
    return [
        (datetime.date.fromisoformat(p["start_date"]), datetime.date.fromisoformat(p["end_date"]))
        for p in data["periods"]
    ]


def is_school_holiday(d: datetime.date, periods: list[tuple[datetime.date, datetime.date]]) -> bool:
    return any(start <= d <= end for start, end in periods)


def is_peak_hour(hour: int) -> bool:
    return any(lo <= hour < hi for lo, hi in PEAK_HOURS)


def register_calendar_table(con: duckdb.DuckDBPyConnection, service_dates: list[datetime.date], holidays_json: Path) -> None:
    """Register a small in-memory calendar lookup table keyed by service_date."""
    de_holidays = holidays.Germany(subdiv="NI")
    school_periods = load_school_holiday_periods(holidays_json)

    rows = [
        {
            "service_date": d,
            "day_of_week": d.weekday(),
            "is_public_holiday": d in de_holidays,
            "is_school_holiday": is_school_holiday(d, school_periods),
        }
        for d in service_dates
    ]
    con.execute(
        "create or replace table calendar as select * from (select unnest($rows, recursive := true))",
        {"rows": rows},
    )


def build_upcoming_stop_pressure(con: duckdb.DuckDBPyConnection, stop_times_path: Path, logger: logging.Logger) -> None:
    """Static-schedule-only feature: count of trips scheduled at a stop within a 5 minute window.

    Computed purely from arr_seconds in the static schedule, so it carries
    no realtime leakage risk regardless of when the prediction is made.
    """
    con.execute(
        f"""
        create or replace table stop_pressure as
        with st as (
            select trip_id, stop_id, arr_seconds
            from read_parquet('{stop_times_path}')
            where arr_seconds is not null
        )
        select
            a.stop_id,
            a.arr_seconds,
            count(distinct b.trip_id) as upcoming_stop_scheduled_count_5min
        from st a
        join st b
          on a.stop_id = b.stop_id
         and abs(a.arr_seconds - b.arr_seconds) <= 300
        group by a.stop_id, a.arr_seconds
        """
    )
    n = con.execute("select count(*) from stop_pressure").fetchone()[0]
    logger.info("built stop_pressure lookup with %d (stop_id, arr_seconds) keys", n)


def build_features(
    static_date: str,
    realised_path: Path,
    weather_path: Path,
    holidays_json: Path,
    logger: logging.Logger,
    k: int = 1,
) -> tuple[duckdb.DuckDBPyConnection, str]:
    schedule_dir = REPO_ROOT / "data" / "interim" / "schedule" / static_date
    stop_times_path = schedule_dir / "stop_times.parquet"

    con = duckdb.connect()
    con.execute("pragma memory_limit='6GB'")
    con.execute(f"pragma temp_directory='{REPO_ROOT / 'data' / 'tmp_duckdb'}'")
    build_upcoming_stop_pressure(con, stop_times_path, logger)

    service_dates = con.execute(
        f"select distinct service_date from read_parquet('{realised_path}')"
    ).fetchnumpy()["service_date"]
    service_dates = [d.astype("M8[D]").tolist() for d in service_dates]
    register_calendar_table(con, service_dates, holidays_json)

    weather_exists = weather_path.exists()
    weather_join = ""
    weather_cols = "null as precipitation, null as temperature_2m, null as wind_speed_10m, null as wind_gusts_10m, null as snowfall"
    if weather_exists:
        con.execute(f"create or replace table weather as select * from read_parquet('{weather_path}')")
        weather_join = """
        left join weather w
          on w.grid_lat = round(base.stop_lat / {res}) * {res}
         and w.grid_lon = round(base.stop_lon / {res}) * {res}
         and w.time = strftime(date_trunc('hour', base.obs_arr_ts), '%Y-%m-%dT%H:00')
        """.format(res=GRID_RESOLUTION_DEGREES)
        weather_cols = "w.precipitation, w.temperature_2m, w.wind_speed_10m, w.wind_gusts_10m, w.snowfall"

    query = f"""
    with sched_lead as (
        select
            cast(trip_id as varchar) as trip_id,
            stop_sequence,
            lead(stop_id) over (partition by trip_id order by stop_sequence) as next_stop_id_sched,
            lead(arr_seconds) over (partition by trip_id order by stop_sequence) as next_arr_seconds
        from read_parquet('{stop_times_path}')
    ),
    base as (
        select
            *,
            sched_arr_ts + to_seconds(coalesce(realised_arr_delay, 0)) as obs_arr_ts,
            sched_dep_ts + to_seconds(coalesce(realised_dep_delay, 0)) as obs_dep_ts,
            lead(stop_id) over trip_w as next_stop_id,
            lead(realised_dep_delay) over trip_w as next_realised_dep_delay,
            lead(sched_dep_ts + to_seconds(coalesce(realised_arr_delay, 0))) over trip_w as next_obs_arr_ts,
            lag(realised_dep_delay, 1) over trip_w as delay_prev1,
            lag(realised_dep_delay, 2) over trip_w as delay_prev2,
            lag(realised_dep_delay, 3) over trip_w as delay_prev3,
            max(stop_sequence) over (partition by service_date, trip_id) as trip_max_seq,
            min(stop_sequence) over (partition by service_date, trip_id) as trip_min_seq
        from read_parquet('{realised_path}')
        window trip_w as (partition by service_date, trip_id order by stop_sequence)
    ),
    with_geo as (
        select
            *,
            lead(stop_lat) over trip_w as next_stop_lat,
            lead(stop_lon) over trip_w as next_stop_lon,
            2 * 6371000 * asin(sqrt(
                pow(sin(radians(lead(stop_lat) over trip_w - stop_lat) / 2), 2) +
                cos(radians(stop_lat)) * cos(radians(lead(stop_lat) over trip_w)) *
                pow(sin(radians(lead(stop_lon) over trip_w - stop_lon) / 2), 2)
            )) as segment_dist_m
        from base
        window trip_w as (partition by service_date, trip_id order by stop_sequence)
    ),
    with_network as (
        select
            *,
            coalesce(next_realised_dep_delay - realised_dep_delay, null) as y_delay_increment,
            (trip_max_seq - stop_sequence) as stops_remaining,
            case when trip_max_seq > trip_min_seq
                 then (stop_sequence - trip_min_seq)::double / (trip_max_seq - trip_min_seq)
                 else null end as elapsed_share,
            sum(coalesce(segment_dist_m, 0)) over (
                partition by service_date, trip_id order by stop_sequence
                rows between current row and unbounded following
            ) as distance_remaining_m,
            (coalesce(delay_prev1, realised_dep_delay) - coalesce(delay_prev3, delay_prev1, realised_dep_delay)) / 3.0 as delay_slope,
            avg(realised_arr_delay) over (
                partition by route_id order by obs_arr_ts
                range between interval 15 minutes preceding and interval 1 second preceding
            ) as route_recent_mean_delay,
            count(*) over (
                partition by route_id order by obs_arr_ts
                range between interval 15 minutes preceding and interval 1 second preceding
            ) as route_recent_n,
            avg(realised_arr_delay) over (
                partition by stop_id, next_stop_id order by obs_arr_ts
                range between interval 15 minutes preceding and interval 1 second preceding
            ) as segment_recent_mean_delay,
            avg(date_diff('second', obs_dep_ts, next_obs_arr_ts)) over (
                partition by route_id, stop_id, next_stop_id order by obs_arr_ts
                range between unbounded preceding and interval 1 second preceding
            ) as segment_historical_avg_runtime,
            (realised_dep_delay - delay_prev1) as delay_jump_1,
            abs(realised_dep_delay - delay_prev1) as delay_jump_1_abs,
            (delay_prev1 - delay_prev2) as delay_jump_2,
            list_aggregate(
                list_filter(list_value(realised_dep_delay, delay_prev1, delay_prev2, delay_prev3), x -> x is not null),
                'stddev_pop'
            ) as delay_recent_std
        from with_geo
    )
    select
        base.service_date,
        base.trip_id,
        base.route_id,
        base.stop_id,
        base.next_stop_id,
        base.stop_sequence,
        base.obs_arr_ts,
        base.realised_dep_delay as current_delay,
        base.y_delay_increment,
        base.delay_prev1,
        base.delay_prev2,
        base.delay_prev3,
        base.delay_slope,
        base.stops_remaining,
        base.distance_remaining_m,
        base.elapsed_share,
        base.dwell_planned_s,
        base.runtime_planned_next_s,
        base.runtime_planned_next_s - base.segment_historical_avg_runtime as slack_s,
        base.route_recent_mean_delay,
        base.route_recent_n,
        base.segment_recent_mean_delay,
        base.delay_jump_1,
        base.delay_jump_1_abs,
        base.delay_jump_2,
        base.delay_recent_std,
        base.realised_dep_delay - base.route_recent_mean_delay as delay_vs_route_recent_gap,
        sp.upcoming_stop_scheduled_count_5min,
        cal.day_of_week,
        cal.is_public_holiday,
        cal.is_school_holiday,
        (hour(base.obs_arr_ts) between 6 and 8 or hour(base.obs_arr_ts) between 16 and 18) as is_peak,
        {weather_cols}
    from with_network base
    left join sched_lead sl
      on sl.trip_id = base.trip_id
     and sl.stop_sequence = base.stop_sequence
    left join stop_pressure sp
      on sp.stop_id = sl.next_stop_id_sched
     and sp.arr_seconds = sl.next_arr_seconds
    left join calendar cal
      on cal.service_date = base.service_date
    {weather_join}
    where base.y_delay_increment is not null
    """
    return con, query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-date", required=True, help="Static feed version used to build the schedule, e.g. 2026-07-23")
    parser.add_argument(
        "--realised-path",
        default=str(REPO_ROOT / "data" / "processed" / "panel_realised.parquet"),
    )
    parser.add_argument(
        "--weather-path",
        default=str(REPO_ROOT / "data" / "interim" / "weather" / "weather_archive.parquet"),
    )
    parser.add_argument(
        "--holidays-json",
        default=str(REPO_ROOT / "config" / "school_holidays_lower_saxony.json"),
    )
    parser.add_argument("--k", type=int, default=1, help="stops ahead for the delay increment target")
    return parser.parse_args()


def main() -> None:
    logger = setup_logging()
    args = parse_args()
    if args.k != 1:
        raise NotImplementedError("only k=1 (next stop) is currently implemented")

    con, query = build_features(
        args.static_date,
        Path(args.realised_path),
        Path(args.weather_path),
        Path(args.holidays_json),
        logger,
        k=args.k,
    )
    out_dir = REPO_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features.parquet"
    con.execute(f"copy ({query}) to '{out_path}' (format parquet)")
    row_count = con.execute(f"select count(*) from read_parquet('{out_path}')").fetchone()[0]
    logger.info("wrote %d rows to %s", row_count, out_path)


if __name__ == "__main__":
    main()
