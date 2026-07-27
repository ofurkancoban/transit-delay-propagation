"""GTFS-Realtime poller with delta compression.

Polls a GTFS-RT protobuf feed on a fixed interval, extracts stop time updates
from TripUpdates, keeps only rows whose delay values changed since the previous
poll for the same (trip_id, stop_sequence), and appends the result to a
partitioned Parquet lake at data/rt/date=YYYY-MM-DD/hour=HH/.

Also extracts ServiceAlerts (added after a Phase 4 diagnosis found that
LightGBM's error concentrates almost entirely in a small "volatile" regime
where a delay is actively changing and the primary feed's TripUpdates alone
carry no advance warning of it; a live pull of the feed confirmed it has no
VehiclePosition data either, so ServiceAlerts is the one other entity type
on this feed worth exploiting). Each alert can inform multiple entities
(route/stop/trip); rows are flattened to one row per (alert, informed
entity) pair. Delta-compressed the same way as TripUpdates, keyed by
alert_id: most alerts are long-lived (a route closure lasting hours), so
without delta compression the same ~50k-entity alert list would be
rewritten in full on every 30s poll. Written to a separate partitioned
Parquet lake at data/rt_alerts/date=YYYY-MM-DD/hour=HH/.

Alert.severity_level is captured too: a live feed check found it
populated on 100% of alerts (INFO/WARNING/SEVERE), and it cleanly
separates the boilerplate DELFI attribution notices (INFO, ~99.8% of
alerts) from actual disruption alerts (WARNING/SEVERE, ~0.2%), unlike
cause/effect which are UNKNOWN_CAUSE/UNKNOWN_EFFECT for the boilerplate
rows too. StopTimeUpdate.stop_time_properties.assigned_stop_id is also
captured (rare, ~0.03% of stop time updates): it signals a platform or
track reassignment away from the scheduled stop, a genuine disruption
event not otherwise visible in the delay fields alone.

Run as a script:
    python -m src.collect.gtfsrt_collector --feed primary
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
import requests
import yaml
from dotenv import load_dotenv
from google.transit import gtfs_realtime_pb2

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("gtfsrt_collector")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(sh)
    return logger


ROW_SCHEMA = pa.schema(
    [
        ("poll_ts", pa.timestamp("s", tz="UTC")),
        ("feed_header_ts", pa.timestamp("s", tz="UTC")),
        ("trip_id", pa.string()),
        ("route_id", pa.string()),
        ("direction_id", pa.int32()),
        ("start_date", pa.string()),
        ("schedule_relationship", pa.string()),
        ("vehicle_id", pa.string()),
        ("stop_id", pa.string()),
        ("stop_sequence", pa.int32()),
        ("arrival_delay", pa.int32()),
        ("arrival_time", pa.int64()),
        ("departure_delay", pa.int32()),
        ("departure_time", pa.int64()),
        ("stu_schedule_relationship", pa.string()),
        ("assigned_stop_id", pa.string()),
    ]
)

ALERT_ROW_SCHEMA = pa.schema(
    [
        ("poll_ts", pa.timestamp("s", tz="UTC")),
        ("feed_header_ts", pa.timestamp("s", tz="UTC")),
        ("alert_id", pa.string()),
        ("cause", pa.string()),
        ("effect", pa.string()),
        ("active_period_start", pa.timestamp("s", tz="UTC")),
        ("active_period_end", pa.timestamp("s", tz="UTC")),
        ("informed_agency_id", pa.string()),
        ("informed_route_id", pa.string()),
        ("informed_route_type", pa.int32()),
        ("informed_stop_id", pa.string()),
        ("informed_trip_id", pa.string()),
        ("header_text", pa.string()),
        ("description_text", pa.string()),
        ("severity_level", pa.string()),
    ]
)


@dataclass
class PollResult:
    rows_seen: int
    rows_written: int
    rows_missing_stop_sequence: int
    feed_header_ts: Optional[int]


class DeltaCache:
    """Tracks the last written (arrival_delay, departure_delay, assigned_stop_id) per (trip_id, stop_sequence)."""

    def __init__(self) -> None:
        self._state: dict[tuple[str, int], tuple[Optional[int], Optional[int], Optional[str]]] = {}

    def changed(self, key: tuple[str, int], value: tuple[Optional[int], Optional[int], Optional[str]]) -> bool:
        prev = self._state.get(key)
        if prev == value:
            return False
        self._state[key] = value
        return True


class AlertDeltaCache:
    """Tracks the last written content hash per alert_id.

    Most alerts are long-lived (a route closure lasting hours), so without
    this the same alert would be rewritten in full on every poll.
    """

    def __init__(self) -> None:
        self._state: dict[str, tuple] = {}

    def changed(self, alert_id: str, value: tuple) -> bool:
        prev = self._state.get(alert_id)
        if prev == value:
            return False
        self._state[alert_id] = value
        return True


def fetch_feed(url: str, timeout: int) -> gtfs_realtime_pb2.FeedMessage:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(resp.content)
    return feed


def schedule_relationship_name(entity_relationship: int) -> str:
    try:
        return gtfs_realtime_pb2.TripDescriptor.ScheduleRelationship.Name(entity_relationship)
    except ValueError:
        return "UNKNOWN"


def stu_relationship_name(rel: int) -> str:
    try:
        return gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.ScheduleRelationship.Name(rel)
    except ValueError:
        return "UNKNOWN"


def extract_rows(feed: gtfs_realtime_pb2.FeedMessage, poll_ts: int, cache: DeltaCache) -> tuple[list[dict], int, int]:
    rows: list[dict] = []
    rows_seen = 0
    missing_seq = 0
    header_ts = feed.header.timestamp if feed.header.HasField("timestamp") else None

    for entity in feed.entity:
        if not entity.HasField("trip_update"):
            continue
        tu = entity.trip_update
        trip_id = tu.trip.trip_id
        route_id = tu.trip.route_id
        direction_id = tu.trip.direction_id if tu.trip.HasField("direction_id") else -1
        start_date = tu.trip.start_date
        sched_rel = schedule_relationship_name(tu.trip.schedule_relationship)
        vehicle_id = tu.vehicle.id if tu.HasField("vehicle") and tu.vehicle.id else None

        for stu in tu.stop_time_update:
            rows_seen += 1
            if not stu.HasField("stop_sequence"):
                missing_seq += 1
                continue

            arrival_delay = stu.arrival.delay if stu.HasField("arrival") and stu.arrival.HasField("delay") else None
            arrival_time = stu.arrival.time if stu.HasField("arrival") and stu.arrival.HasField("time") else None
            departure_delay = stu.departure.delay if stu.HasField("departure") and stu.departure.HasField("delay") else None
            departure_time = stu.departure.time if stu.HasField("departure") and stu.departure.HasField("time") else None
            assigned_stop_id = (
                stu.stop_time_properties.assigned_stop_id
                if stu.HasField("stop_time_properties") and stu.stop_time_properties.assigned_stop_id
                else None
            )

            key = (trip_id, stu.stop_sequence)
            value = (arrival_delay, departure_delay, assigned_stop_id)
            if not cache.changed(key, value):
                continue

            rows.append(
                {
                    "poll_ts": poll_ts,
                    "feed_header_ts": header_ts,
                    "trip_id": trip_id,
                    "route_id": route_id,
                    "direction_id": direction_id,
                    "start_date": start_date,
                    "schedule_relationship": sched_rel,
                    "vehicle_id": vehicle_id,
                    "stop_id": stu.stop_id,
                    "stop_sequence": stu.stop_sequence,
                    "arrival_delay": arrival_delay,
                    "arrival_time": arrival_time,
                    "departure_delay": departure_delay,
                    "departure_time": departure_time,
                    "stu_schedule_relationship": stu_relationship_name(stu.schedule_relationship),
                    "assigned_stop_id": assigned_stop_id,
                }
            )

    return rows, rows_seen, missing_seq


def write_rows(rows: list[dict], rt_lake_dir: Path, poll_ts: int) -> None:
    if not rows:
        return
    import datetime

    dt = datetime.datetime.fromtimestamp(poll_ts, tz=datetime.timezone.utc)
    partition_dir = rt_lake_dir / f"date={dt.strftime('%Y-%m-%d')}" / f"hour={dt.strftime('%H')}"
    partition_dir.mkdir(parents=True, exist_ok=True)

    columns = {name: [] for name in ROW_SCHEMA.names}
    for row in rows:
        for name in ROW_SCHEMA.names:
            columns[name].append(row.get(name))

    table = pa.table(
        {
            "poll_ts": pa.array(columns["poll_ts"], type=pa.int64()).cast(pa.timestamp("s", tz="UTC")),
            "feed_header_ts": pa.array(columns["feed_header_ts"], type=pa.int64()).cast(pa.timestamp("s", tz="UTC")),
            "trip_id": pa.array(columns["trip_id"], type=pa.string()),
            "route_id": pa.array(columns["route_id"], type=pa.string()),
            "direction_id": pa.array(columns["direction_id"], type=pa.int32()),
            "start_date": pa.array(columns["start_date"], type=pa.string()),
            "schedule_relationship": pa.array(columns["schedule_relationship"], type=pa.string()),
            "vehicle_id": pa.array(columns["vehicle_id"], type=pa.string()),
            "stop_id": pa.array(columns["stop_id"], type=pa.string()),
            "stop_sequence": pa.array(columns["stop_sequence"], type=pa.int32()),
            "arrival_delay": pa.array(columns["arrival_delay"], type=pa.int32()),
            "arrival_time": pa.array(columns["arrival_time"], type=pa.int64()),
            "departure_delay": pa.array(columns["departure_delay"], type=pa.int32()),
            "departure_time": pa.array(columns["departure_time"], type=pa.int64()),
            "stu_schedule_relationship": pa.array(columns["stu_schedule_relationship"], type=pa.string()),
            "assigned_stop_id": pa.array(columns["assigned_stop_id"], type=pa.string()),
        }
    )

    file_path = partition_dir / f"poll_{poll_ts}.parquet"
    pq.write_table(table, file_path, compression="zstd")


def cause_name(cause: int) -> str:
    try:
        return gtfs_realtime_pb2.Alert.Cause.Name(cause)
    except ValueError:
        return "UNKNOWN_CAUSE"


def effect_name(effect: int) -> str:
    try:
        return gtfs_realtime_pb2.Alert.Effect.Name(effect)
    except ValueError:
        return "UNKNOWN_EFFECT"


def severity_level_name(severity_level: int) -> str:
    try:
        return gtfs_realtime_pb2.Alert.SeverityLevel.Name(severity_level)
    except ValueError:
        return "UNKNOWN_SEVERITY"


def translated_text(translated_string, default: Optional[str] = None) -> Optional[str]:
    if not translated_string.translation:
        return default
    for t in translated_string.translation:
        if t.language in ("en", "de", ""):
            return t.text
    return translated_string.translation[0].text


def extract_alert_rows(
    feed: gtfs_realtime_pb2.FeedMessage, poll_ts: int, cache: AlertDeltaCache
) -> tuple[list[dict], int]:
    """Flatten each Alert to one row per informed entity. Delta-compressed by alert_id."""
    rows: list[dict] = []
    alerts_seen = 0
    header_ts = feed.header.timestamp if feed.header.HasField("timestamp") else None

    for entity in feed.entity:
        if not entity.HasField("alert"):
            continue
        alerts_seen += 1
        alert = entity.alert
        alert_id = entity.id
        cause = cause_name(alert.cause)
        effect = effect_name(alert.effect)
        severity_level = severity_level_name(alert.severity_level) if alert.HasField("severity_level") else None
        header_text = translated_text(alert.header_text)
        description_text = translated_text(alert.description_text)

        active_period_start = None
        active_period_end = None
        if alert.active_period:
            period = alert.active_period[0]
            active_period_start = period.start if period.HasField("start") else None
            active_period_end = period.end if period.HasField("end") else None

        informed_entities = alert.informed_entity or [None]

        value = (
            cause,
            effect,
            severity_level,
            active_period_start,
            active_period_end,
            header_text,
            description_text,
            tuple(
                (ie.agency_id, ie.route_id, ie.route_type if ie.HasField("route_type") else None, ie.stop_id, ie.trip.trip_id)
                for ie in alert.informed_entity
            ),
        )
        if not cache.changed(alert_id, value):
            continue

        for ie in informed_entities:
            rows.append(
                {
                    "poll_ts": poll_ts,
                    "feed_header_ts": header_ts,
                    "alert_id": alert_id,
                    "cause": cause,
                    "effect": effect,
                    "severity_level": severity_level,
                    "active_period_start": active_period_start,
                    "active_period_end": active_period_end,
                    "informed_agency_id": ie.agency_id if ie else None,
                    "informed_route_id": ie.route_id if ie and ie.route_id else None,
                    "informed_route_type": ie.route_type if ie and ie.HasField("route_type") else None,
                    "informed_stop_id": ie.stop_id if ie and ie.stop_id else None,
                    "informed_trip_id": ie.trip.trip_id if ie and ie.trip.trip_id else None,
                    "header_text": header_text,
                    "description_text": description_text,
                }
            )

    return rows, alerts_seen


def write_alert_rows(rows: list[dict], alerts_lake_dir: Path, poll_ts: int) -> None:
    if not rows:
        return
    import datetime

    dt = datetime.datetime.fromtimestamp(poll_ts, tz=datetime.timezone.utc)
    partition_dir = alerts_lake_dir / f"date={dt.strftime('%Y-%m-%d')}" / f"hour={dt.strftime('%H')}"
    partition_dir.mkdir(parents=True, exist_ok=True)

    columns = {name: [] for name in ALERT_ROW_SCHEMA.names}
    for row in rows:
        for name in ALERT_ROW_SCHEMA.names:
            columns[name].append(row.get(name))

    table = pa.table(
        {
            "poll_ts": pa.array(columns["poll_ts"], type=pa.int64()).cast(pa.timestamp("s", tz="UTC")),
            "feed_header_ts": pa.array(columns["feed_header_ts"], type=pa.int64()).cast(pa.timestamp("s", tz="UTC")),
            "alert_id": pa.array(columns["alert_id"], type=pa.string()),
            "cause": pa.array(columns["cause"], type=pa.string()),
            "effect": pa.array(columns["effect"], type=pa.string()),
            "active_period_start": pa.array(columns["active_period_start"], type=pa.int64()).cast(pa.timestamp("s", tz="UTC")),
            "active_period_end": pa.array(columns["active_period_end"], type=pa.int64()).cast(pa.timestamp("s", tz="UTC")),
            "informed_agency_id": pa.array(columns["informed_agency_id"], type=pa.string()),
            "informed_route_id": pa.array(columns["informed_route_id"], type=pa.string()),
            "informed_route_type": pa.array(columns["informed_route_type"], type=pa.int32()),
            "informed_stop_id": pa.array(columns["informed_stop_id"], type=pa.string()),
            "informed_trip_id": pa.array(columns["informed_trip_id"], type=pa.string()),
            "header_text": pa.array(columns["header_text"], type=pa.string()),
            "description_text": pa.array(columns["description_text"], type=pa.string()),
            "severity_level": pa.array(columns["severity_level"], type=pa.string()),
        }
    )

    file_path = partition_dir / f"poll_{poll_ts}.parquet"
    pq.write_table(table, file_path, compression="zstd")


def run(
    feed_url: str,
    rt_lake_dir: Path,
    alerts_lake_dir: Path,
    poll_interval: int,
    timeout: int,
    logger: logging.Logger,
) -> None:
    cache = DeltaCache()
    alert_cache = AlertDeltaCache()
    rt_lake_dir.mkdir(parents=True, exist_ok=True)
    alerts_lake_dir.mkdir(parents=True, exist_ok=True)
    logger.info("collector started, feed=%s interval=%ss", feed_url, poll_interval)

    total_rows_written = 0
    total_alert_rows_written = 0
    poll_count = 0
    fail_count = 0

    while True:
        loop_start = time.time()
        poll_ts = int(loop_start)
        try:
            feed = fetch_feed(feed_url, timeout)
            rows, seen, missing_seq = extract_rows(feed, poll_ts, cache)
            write_rows(rows, rt_lake_dir, poll_ts)
            alert_rows, alerts_seen = extract_alert_rows(feed, poll_ts, alert_cache)
            write_alert_rows(alert_rows, alerts_lake_dir, poll_ts)
            poll_count += 1
            total_rows_written += len(rows)
            total_alert_rows_written += len(alert_rows)
            header_ts = feed.header.timestamp if feed.header.HasField("timestamp") else None
            lag = (poll_ts - header_ts) if header_ts else None
            logger.info(
                "poll=%d seen=%d written=%d missing_stop_sequence=%d lag_s=%s total_written=%d "
                "alerts_seen=%d alert_rows_written=%d alert_rows_total=%d",
                poll_count,
                seen,
                len(rows),
                missing_seq,
                lag,
                total_rows_written,
                alerts_seen,
                len(alert_rows),
                total_alert_rows_written,
            )
        except Exception as exc:
            fail_count += 1
            logger.exception("poll failed (fail_count=%d): %s", fail_count, exc)

        elapsed = time.time() - loop_start
        sleep_for = max(0.0, poll_interval - elapsed)
        time.sleep(sleep_for)


def main() -> None:
    parser = argparse.ArgumentParser(description="GTFS-RT collector with delta compression")
    parser.add_argument("--feed", choices=["primary", "secondary"], default="primary")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    config = load_config()

    url_env = (
        config["collector"]["primary_feed_url_env"]
        if args.feed == "primary"
        else config["collector"]["secondary_feed_url_env"]
    )
    feed_url = os.environ.get(url_env)
    if not feed_url:
        raise SystemExit(f"environment variable {url_env} is not set")

    rt_lake_dir = REPO_ROOT / config["collector"]["rt_lake_dir"]
    alerts_lake_dir = REPO_ROOT / config["collector"].get("alerts_lake_dir", "data/rt_alerts")
    if args.feed == "secondary":
        rt_lake_dir = rt_lake_dir.parent / "rt_secondary"
        alerts_lake_dir = alerts_lake_dir.parent / "rt_alerts_secondary"

    log_file = REPO_ROOT / config["collector"]["log_file"]
    if args.feed == "secondary":
        log_file = log_file.with_name("collector_secondary.log")

    logger = setup_logging(log_file)
    run(
        feed_url=feed_url,
        rt_lake_dir=rt_lake_dir,
        alerts_lake_dir=alerts_lake_dir,
        poll_interval=config["collector"]["poll_interval_seconds"],
        timeout=config["collector"]["request_timeout_seconds"],
        logger=logger,
    )


if __name__ == "__main__":
    main()
