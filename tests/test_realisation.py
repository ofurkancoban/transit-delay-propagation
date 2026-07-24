import datetime

import polars as pl
import pytest

from src.build.realisation import (
    add_sched_timestamps,
    assert_static_version_covers_dates,
    seconds_to_gtfs_time_string,
)


def test_seconds_to_gtfs_time_string_normal():
    assert seconds_to_gtfs_time_string(8 * 3600 + 15 * 60) == "08:15:00"


def test_seconds_to_gtfs_time_string_past_midnight():
    assert seconds_to_gtfs_time_string(25 * 3600 + 30 * 60) == "25:30:00"


def test_add_sched_timestamps_midnight_crossing():
    df = pl.DataFrame(
        {
            "service_date": [datetime.date(2026, 7, 23), datetime.date(2026, 7, 23)],
            "arr_seconds": [8 * 3600, 25 * 3600 + 30 * 60],
        }
    )
    result = add_sched_timestamps(df, "arr_seconds", "sched_arr_ts")
    local = result["sched_arr_ts"].dt.convert_time_zone("Europe/Berlin")
    assert local[0].date() == datetime.date(2026, 7, 23)
    assert local[0].hour == 8
    # 25:30:00 on 2026-07-23 lands on 2026-07-24 at 01:30 local, service_date unchanged.
    assert local[1].date() == datetime.date(2026, 7, 24)
    assert local[1].hour == 1
    assert local[1].minute == 30


def test_assert_static_version_covers_dates_passes_when_in_range():
    manifest = {"feed_start_date": "20260718", "feed_end_date": "20260817"}
    assert_static_version_covers_dates([datetime.date(2026, 7, 23)], manifest)


def test_assert_static_version_covers_dates_raises_when_stale():
    manifest = {"feed_start_date": "20260701", "feed_end_date": "20260707"}
    with pytest.raises(RuntimeError, match="validity window"):
        assert_static_version_covers_dates([datetime.date(2026, 7, 23)], manifest)
