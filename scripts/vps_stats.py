#!/usr/bin/env python3
"""Read-only status snapshot for the live dashboard.

Parses the collector's own log file (never touches the parquet lake, so
this is cheap enough to run every few minutes) and prints a single JSON
object to stdout. Meant to be invoked over a restricted SSH command by
the refresh-dashboard GitHub Actions workflow, not run interactively.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE = Path("/opt/transit-delay-propagation/logs/collector.log")
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
    fail_count = sum(1 for line in lines if "poll failed" in line)
    for line in reversed(lines):
        m = ALERTS_LOG_PATTERN.search(line)
        if m:
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
            break

    return {
        "collector_started_at": started_at,
        "last_poll": last_poll,
        "fail_count_this_process": fail_count,
    }


def main() -> None:
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "systemd": systemctl_status("gtfsrt-collector.service"),
        "log": parse_log(LOG_FILE),
    }
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
