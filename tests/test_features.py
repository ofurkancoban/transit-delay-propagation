import datetime

import duckdb
import pytest

from src.build.features import (
    build_upcoming_stop_pressure,
    is_peak_hour,
    is_school_holiday,
    load_school_holiday_periods,
)


def test_is_school_holiday_inside_period():
    periods = [(datetime.date(2026, 7, 16), datetime.date(2026, 8, 26))]
    assert is_school_holiday(datetime.date(2026, 7, 23), periods)
    assert not is_school_holiday(datetime.date(2026, 9, 1), periods)


def test_load_school_holiday_periods(tmp_path):
    import json

    payload = {
        "state": "Niedersachsen",
        "periods": [{"name": "Test", "start_date": "2026-01-01", "end_date": "2026-01-10"}],
    }
    path = tmp_path / "holidays.json"
    path.write_text(json.dumps(payload))
    periods = load_school_holiday_periods(path)
    assert periods == [(datetime.date(2026, 1, 1), datetime.date(2026, 1, 10))]


def test_is_peak_hour():
    assert is_peak_hour(7)
    assert is_peak_hour(17)
    assert not is_peak_hour(12)
    assert not is_peak_hour(23)


def test_build_upcoming_stop_pressure_counts_within_window(tmp_path):
    import polars as pl

    stop_times = pl.DataFrame(
        {
            "trip_id": ["a", "b", "c", "d"],
            "stop_id": ["S1", "S1", "S1", "S1"],
            # arrivals at 08:00:00, 08:03:00 (within 300s of a), 08:10:00 (outside), 08:04:30
            "arr_seconds": [8 * 3600, 8 * 3600 + 180, 8 * 3600 + 600, 8 * 3600 + 270],
        }
    )
    path = tmp_path / "stop_times.parquet"
    stop_times.write_parquet(path)

    con = duckdb.connect()
    logger = __import__("logging").getLogger("test")
    build_upcoming_stop_pressure(con, path, logger)
    result = con.execute(
        "select arr_seconds, upcoming_stop_scheduled_count_5min from stop_pressure order by arr_seconds"
    ).fetchall()

    counts = dict(result)
    # trip a (08:00:00): within 300s are a itself, b (08:03:00, diff 180s), d (08:04:30, diff 270s) -> 3
    assert counts[8 * 3600] == 3
    # trip c (08:10:00) is isolated, only itself within window -> 1
    assert counts[8 * 3600 + 600] == 1


def test_leakage_no_future_timestamp_in_network_window():
    """Structural check: the network-state window frames only look backward.

    RANGE BETWEEN INTERVAL 15 MINUTES PRECEDING AND INTERVAL 1 SECOND PRECEDING,
    ordered by obs_arr_ts, cannot include the current row or any row at or
    after its timestamp by construction. This test constructs a small
    synthetic example and asserts the aggregate excludes a same-timestamp
    and a future row.
    """
    con = duckdb.connect()
    con.execute(
        """
        create table t as select * from (values
            ('r1', timestamp '2026-07-23 08:00:00', 10),
            ('r2', timestamp '2026-07-23 08:05:00', 20),
            ('r3', timestamp '2026-07-23 08:05:00', 999),
            ('r4', timestamp '2026-07-23 08:20:00', 30)
        ) as t(id, ts, delay)
        """
    )
    result = con.execute(
        """
        select id, ts,
               avg(delay) over (
                   order by ts
                   range between interval 15 minutes preceding and interval 1 second preceding
               ) as recent_mean
        from t
        order by ts, id
        """
    ).fetchall()
    by_id = {row[0]: row[2] for row in result}
    # r2's window should see only r1 (10), not r3 (same timestamp, tie) or anything after.
    assert by_id["r2"] == pytest.approx(10.0)
    # r4's window (08:05 to 08:20 exclusive of 08:20 itself) should average r2 and r3.
    assert by_id["r4"] == pytest.approx((20 + 999) / 2)
