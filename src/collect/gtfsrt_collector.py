"""GTFS-Realtime poller with delta compression.

Polls a GTFS-RT protobuf feed on a fixed interval, extracts stop time updates
from TripUpdates, keeps only rows whose delay values changed since the previous
poll for the same (trip_id, stop_sequence), and appends the result to a
partitioned Parquet lake at data/rt/date=YYYY-MM-DD/hour=HH/.

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
    ]
)


@dataclass
class PollResult:
    rows_seen: int
    rows_written: int
    rows_missing_stop_sequence: int
    feed_header_ts: Optional[int]


class DeltaCache:
    """Tracks the last written (arrival_delay, departure_delay) per (trip_id, stop_sequence)."""

    def __init__(self) -> None:
        self._state: dict[tuple[str, int], tuple[Optional[int], Optional[int]]] = {}

    def changed(self, key: tuple[str, int], value: tuple[Optional[int], Optional[int]]) -> bool:
        prev = self._state.get(key)
        if prev == value:
            return False
        self._state[key] = value
        return True


def fetch_feed(url: str, timeout: int) -> gtfs_realtime_pb2.FeedMessage:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(resp.content)
    return feed


def schedule_relationship_name(entity_relationship: int) -> str:
    names = gtfs_realtime_pb2.TripDescriptor.ScheduleRelationship.keys()
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

            key = (trip_id, stu.stop_sequence)
            value = (arrival_delay, departure_delay)
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
        }
    )

    file_path = partition_dir / f"poll_{poll_ts}.parquet"
    pq.write_table(table, file_path, compression="zstd")


def run(feed_url: str, rt_lake_dir: Path, poll_interval: int, timeout: int, logger: logging.Logger) -> None:
    cache = DeltaCache()
    rt_lake_dir.mkdir(parents=True, exist_ok=True)
    logger.info("collector started, feed=%s interval=%ss", feed_url, poll_interval)

    total_rows_written = 0
    poll_count = 0
    fail_count = 0

    while True:
        loop_start = time.time()
        poll_ts = int(loop_start)
        try:
            feed = fetch_feed(feed_url, timeout)
            rows, seen, missing_seq = extract_rows(feed, poll_ts, cache)
            write_rows(rows, rt_lake_dir, poll_ts)
            poll_count += 1
            total_rows_written += len(rows)
            header_ts = feed.header.timestamp if feed.header.HasField("timestamp") else None
            lag = (poll_ts - header_ts) if header_ts else None
            logger.info(
                "poll=%d seen=%d written=%d missing_stop_sequence=%d lag_s=%s total_written=%d",
                poll_count,
                seen,
                len(rows),
                missing_seq,
                lag,
                total_rows_written,
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
    if args.feed == "secondary":
        rt_lake_dir = rt_lake_dir.parent / "rt_secondary"

    log_file = REPO_ROOT / config["collector"]["log_file"]
    if args.feed == "secondary":
        log_file = log_file.with_name("collector_secondary.log")

    logger = setup_logging(log_file)
    run(
        feed_url=feed_url,
        rt_lake_dir=rt_lake_dir,
        poll_interval=config["collector"]["poll_interval_seconds"],
        timeout=config["collector"]["request_timeout_seconds"],
        logger=logger,
    )


if __name__ == "__main__":
    main()
