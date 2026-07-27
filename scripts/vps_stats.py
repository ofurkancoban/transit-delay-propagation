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
from datetime import datetime, timezone
from pathlib import Path

import duckdb

REPO_ROOT = Path("/opt/transit-delay-propagation")
LOG_FILE = REPO_ROOT / "logs" / "collector.log"
RT_GLOB = str(REPO_ROOT / "data" / "rt" / "**" / "*.parquet")
ALERTS_GLOB = str(REPO_ROOT / "data" / "rt_alerts" / "**" / "*.parquet")
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


def main() -> None:
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "systemd": systemctl_status("gtfsrt-collector.service"),
        "log": parse_log(LOG_FILE),
        "lake": lake_stats(),
    }
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
