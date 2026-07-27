#!/usr/bin/env python3
"""Read-only status snapshot for the live dashboard.

Parses the collector's own log file and runs one read-only DuckDB scan
over the parquet lake, then prints a single JSON object to stdout. Meant
to be invoked over a restricted SSH command by the refresh-dashboard
GitHub Actions workflow, not run interactively.

The lake scan exists because the log's own `total_written` counter is
per-process: it resets to 0 whenever the collector process restarts
(observed directly — the log's cumulative counter read ~78M while the
actual parquet lake already held ~139M rows, from runs before the most
recent process start), even when systemd's own restart counter reports
0 restarts for the *current* process. The physical row count in the lake
is the only number that is actually cumulative across the whole
collection history, so it is treated as authoritative here despite
costing a few seconds per refresh (acceptable at a 10-minute cadence).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb

REPO_ROOT = Path("/opt/transit-delay-propagation")
LOG_FILE = REPO_ROOT / "logs" / "collector.log"
RT_GLOB = str(REPO_ROOT / "data" / "rt" / "**" / "*.parquet")
ALERTS_GLOB = str(REPO_ROOT / "data" / "rt_alerts" / "**" / "*.parquet")
STATIC_DIR = REPO_ROOT / "data" / "static"
MAP_WINDOW_MINUTES = 15
MAP_MAX_STOPS = 250
ALERTS_LOG_PATTERN = re.compile(
    r"poll=(\d+) seen=(\d+) written=(\d+) missing_stop_sequence=(\d+) lag_s=(\S+) total_written=(\d+)"
    r"(?: alerts_seen=(\d+) alert_rows_written=(\d+) alert_rows_total=(\d+))?"
)
STARTED_PATTERN = re.compile(r"^(\S+ \S+) INFO collector started")


def systemctl_status(unit: str) -> dict:
    try:
        active = subprocess.run(
            ["systemctl", "is-active", unit], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        restarts = subprocess.run(
            ["systemctl", "show", unit, "-p", "NRestarts", "--value"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        since = subprocess.run(
            ["systemctl", "show", unit, "-p", "ActiveEnterTimestamp", "--value"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return {"active": active, "restarts": int(restarts or 0), "active_since": since}
    except Exception:
        return {"active": "unknown", "restarts": None, "active_since": None}


def percentile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def parse_log(log_file: Path) -> dict:
    if not log_file.exists():
        return {}
    lines = log_file.read_text(errors="replace").splitlines()

    started_at = None
    for line in lines:
        m = STARTED_PATTERN.match(line)
        if m:
            started_at = m.group(1)

    last_poll = None
    poll_count = 0
    fail_count = sum(1 for line in lines if "poll failed" in line)
    lag_values: list[float] = []
    for line in lines:
        m = ALERTS_LOG_PATTERN.search(line)
        if not m:
            continue
        poll_count += 1
        try:
            lag_values.append(float(m.group(5)))
        except (TypeError, ValueError):
            pass
        last_poll = {
            "poll": int(m.group(1)),
            "seen": int(m.group(2)),
            "written": int(m.group(3)),
            "missing_stop_sequence": int(m.group(4)),
            "lag_s": m.group(5),
            "total_written": int(m.group(6)),
            "alerts_seen": int(m.group(7)) if m.group(7) else None,
            "alert_rows_written": int(m.group(8)) if m.group(8) else None,
            "alert_rows_total": int(m.group(9)) if m.group(9) else None,
        }

    lag_values.sort()
    total_attempts = poll_count + fail_count
    poll_success_rate = (poll_count / total_attempts) if total_attempts else None

    return {
        "collector_started_at": started_at,
        "last_poll": last_poll,
        "fail_count_this_process": fail_count,
        "poll_count_this_process": poll_count,
        "poll_success_rate": poll_success_rate,
        "lag_p50": percentile(lag_values, 0.50),
        "lag_p90": percentile(lag_values, 0.90),
        "lag_max": lag_values[-1] if lag_values else None,
    }


def lake_stats() -> dict:
    """Authoritative cumulative counts from the parquet lake itself (see module docstring)."""
    con = duckdb.connect()
    try:
        row = con.execute(
            f"""
            select
                count(*) as rows,
                count(distinct trip_id) as trips,
                avg(case when abs(arrival_delay) > 360 then 1.0 else 0.0 end)
                    filter (where arrival_delay is not null) as delay_gt_6min_share
            from read_parquet('{RT_GLOB}')
            """
        ).fetchone()
        alert_row = con.execute(f"select count(*) from read_parquet('{ALERTS_GLOB}')").fetchone()
        return {
            "total_rows": row[0],
            "distinct_trips": row[1],
            "delay_gt_6min_share": row[2],
            "alert_rows_total_lake": alert_row[0],
        }
    except Exception as exc:
        return {"error": str(exc)}


def latest_static_stops_path() -> Path | None:
    """Most recently fetched static feed's stops.txt (for stop lat/lon)."""
    if not STATIC_DIR.exists():
        return None
    candidates = sorted((d for d in STATIC_DIR.iterdir() if d.is_dir()), reverse=True)
    for d in candidates:
        stops = d / "extracted" / "stops.txt"
        if stops.exists():
            return stops
    return None


def recent_rt_globs(now: datetime, window_minutes: int) -> list[str]:
    """Hour-partition globs covering [now - window_minutes, now], to avoid scanning the full lake."""
    hours = {now, now - timedelta(minutes=window_minutes)}
    globs = []
    for hour_dt in hours:
        globs.append(
            str(REPO_ROOT / "data" / "rt" / f"date={hour_dt.strftime('%Y-%m-%d')}" / f"hour={hour_dt.strftime('%H')}" / "*.parquet")
        )
    return globs


def map_snapshot() -> dict:
    """Latest known delay per stop, for the live delay map.

    Capped to the MAP_MAX_STOPS worst currently-known delays nationwide,
    not the full ~264,933-stop network: with no VehiclePosition on this
    feed (confirmed absent), "live position" is approximated by the last
    reported delay per stop_id within the trailing window, and the full
    per-poll stop set (~130k distinct stops in 20 minutes) would make the
    JSON committed by the refresh workflow every 10 minutes grow without
    bound over time. Showing the worst delays is also the more useful
    view for a "delay propagation" map than an unfiltered scatter.
    """
    stops_path = latest_static_stops_path()
    if stops_path is None:
        return {"error": "no static stops.txt found"}

    now = datetime.now(timezone.utc)
    globs = recent_rt_globs(now, MAP_WINDOW_MINUTES)
    cutoff = now - timedelta(minutes=MAP_WINDOW_MINUTES)
    con = duckdb.connect()
    try:
        rows = con.execute(
            f"""
            with recent as (
                select stop_id, trip_id, arrival_delay, poll_ts,
                    row_number() over (partition by stop_id order by poll_ts desc) as rn
                from read_parquet({globs!r})
                where poll_ts >= timestamp '{cutoff.strftime('%Y-%m-%d %H:%M:%S')}'
                    and arrival_delay is not null
            )
            select s.stop_id, s.stop_name, s.stop_lat, s.stop_lon, r.arrival_delay, r.trip_id
            from recent r
            join read_csv_auto({str(stops_path)!r}) s using (stop_id)
            where r.rn = 1
            order by abs(r.arrival_delay) desc
            limit {MAP_MAX_STOPS}
            """
        ).fetchall()
    except Exception as exc:
        return {"error": str(exc)}

    return {
        "window_minutes": MAP_WINDOW_MINUTES,
        "max_stops": MAP_MAX_STOPS,
        "stops": [
            {
                "stop_id": r[0],
                "stop_name": r[1],
                "lat": r[2],
                "lon": r[3],
                "delay_s": r[4],
                "trip_id": r[5],
            }
            for r in rows
        ],
    }


def main() -> None:
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "systemd": systemctl_status("gtfsrt-collector.service"),
        "log": parse_log(LOG_FILE),
        "lake": lake_stats(),
        "map": map_snapshot(),
    }
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
