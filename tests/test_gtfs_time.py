import datetime

import pytest
from zoneinfo import ZoneInfo

from src.build.gtfs_time import gtfs_time_to_utc_timestamp, parse_gtfs_time_to_seconds

BERLIN = ZoneInfo("Europe/Berlin")


def test_parse_normal_time():
    assert parse_gtfs_time_to_seconds("08:15:00") == 8 * 3600 + 15 * 60


def test_parse_midnight_crossing_time():
    assert parse_gtfs_time_to_seconds("25:30:00") == 25 * 3600 + 30 * 60


def test_parse_far_past_midnight():
    assert parse_gtfs_time_to_seconds("27:05:10") == 27 * 3600 + 5 * 60 + 10


def test_invalid_time_raises():
    with pytest.raises(ValueError):
        parse_gtfs_time_to_seconds("08:75:00")


def test_midnight_crossing_lands_on_next_calendar_day():
    service_date = datetime.date(2026, 7, 1)
    ts = gtfs_time_to_utc_timestamp(service_date, "25:30:00", BERLIN)
    local_ts = ts.astimezone(BERLIN)
    assert local_ts.date() == datetime.date(2026, 7, 2)
    assert local_ts.hour == 1
    assert local_ts.minute == 30


def test_normal_time_stays_on_service_date():
    service_date = datetime.date(2026, 7, 1)
    ts = gtfs_time_to_utc_timestamp(service_date, "08:15:00", BERLIN)
    local_ts = ts.astimezone(BERLIN)
    assert local_ts.date() == service_date
    assert local_ts.hour == 8
    assert local_ts.minute == 15


def test_dst_spring_forward_germany_2026():
    # Europe/Berlin DST starts 2026-03-29; 02:00 to 03:00 local is skipped.
    service_date = datetime.date(2026, 3, 28)
    ts = gtfs_time_to_utc_timestamp(service_date, "26:30:00", BERLIN)
    # 26:30:00 on 2026-03-28 is 02:30 on 2026-03-29, inside the skipped hour,
    # zoneinfo normalises this to the equivalent UTC instant without raising.
    assert ts.tzinfo == datetime.timezone.utc
