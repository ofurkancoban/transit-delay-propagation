"""Time arithmetic helpers for GTFS static data.

GTFS stop_times.txt legitimately contains times past midnight, such as
25:30:00 for a trip that starts before midnight and continues into the next
service day. Naive parsing with datetime.strptime corrupts overnight service.
"""

from __future__ import annotations

import datetime


def parse_gtfs_time_to_seconds(value: str) -> int:
    """Parse an HH:MM:SS GTFS time string into seconds since midnight of the service date.

    HH may exceed 23 (e.g. 25:30:00 means 01:30:00 the following calendar day,
    still counted against the trip's service date).
    """
    parts = value.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"invalid GTFS time string: {value!r}")
    hours, minutes, seconds = (int(p) for p in parts)
    if minutes < 0 or minutes > 59 or seconds < 0 or seconds > 59:
        raise ValueError(f"invalid GTFS time string: {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def gtfs_time_to_utc_timestamp(
    service_date: datetime.date,
    gtfs_time: str,
    local_tz: datetime.tzinfo,
) -> datetime.datetime:
    """Convert a GTFS time string plus a service date into an absolute UTC timestamp.

    The service date stays fixed even when seconds_since_midnight rolls past
    86400 (i.e. hours >= 24); the calendar day advances but the logical
    service date used for joins does not.
    """
    seconds = parse_gtfs_time_to_seconds(gtfs_time)
    local_midnight = datetime.datetime.combine(service_date, datetime.time.min, tzinfo=local_tz)
    local_dt = local_midnight + datetime.timedelta(seconds=seconds)
    return local_dt.astimezone(datetime.timezone.utc)
