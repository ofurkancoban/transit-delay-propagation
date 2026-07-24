import datetime
import logging
from pathlib import Path

import duckdb
import polars as pl

from src.build.schedule import build_service_dates, build_stop_times, build_trips


def write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


def test_calendar_expansion_respects_weekday_and_date_range(tmp_path):
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # service_id 1 runs Mondays only, 2026-07-06 (Mon) to 2026-07-20 (Mon) inclusive.
    write_csv(
        extract_dir / "calendar.txt",
        "monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date,service_id",
        ["1,0,0,0,0,0,0,20260706,20260720,1"],
    )
    write_csv(
        extract_dir / "calendar_dates.txt",
        "service_id,date,exception_type",
        [],
    )

    build_service_dates(duckdb.connect(), extract_dir, out_dir, logging.getLogger("test"))
    result = pl.read_parquet(out_dir / "service_dates.parquet")
    dates = sorted(result["service_date"].dt.date().to_list())

    assert dates == [
        datetime.date(2026, 7, 6),
        datetime.date(2026, 7, 13),
        datetime.date(2026, 7, 20),
    ]


def test_calendar_dates_exceptions_add_and_remove(tmp_path):
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # service_id 1 runs Mondays, 2026-07-06 to 2026-07-20.
    write_csv(
        extract_dir / "calendar.txt",
        "monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date,service_id",
        ["1,0,0,0,0,0,0,20260706,20260720,1"],
    )
    # Remove 2026-07-13 (a regular Monday), add 2026-07-15 (a Wednesday, public holiday service).
    write_csv(
        extract_dir / "calendar_dates.txt",
        "service_id,date,exception_type",
        ["1,20260713,2", "1,20260715,1"],
    )

    build_service_dates(duckdb.connect(), extract_dir, out_dir, logging.getLogger("test"))
    result = pl.read_parquet(out_dir / "service_dates.parquet")
    dates = sorted(result["service_date"].dt.date().to_list())

    assert dates == [
        datetime.date(2026, 7, 6),
        datetime.date(2026, 7, 15),
        datetime.date(2026, 7, 20),
    ]


def test_stop_times_midnight_crossing_preserves_seconds_past_24h(tmp_path):
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    write_csv(
        extract_dir / "stop_times.txt",
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence,stop_headsign,pickup_type,drop_off_type",
        [
            "1,23:50:00,23:50:00,A,0,,,",
            "1,24:10:00,24:15:00,B,1,,,",
            "1,25:30:00,25:30:00,C,2,,,",
        ],
    )
    write_csv(
        extract_dir / "stops.txt",
        "stop_name,parent_station,stop_id,stop_lat,stop_lon,location_type,platform_code",
        ["A,,A,52.0,13.0,,", "B,,B,52.1,13.1,,", "C,,C,52.2,13.2,,"],
    )

    build_stop_times(duckdb.connect(), extract_dir, out_dir, logging.getLogger("test"))
    result = pl.read_parquet(out_dir / "stop_times.parquet").sort("stop_sequence")

    assert result["arr_seconds"].to_list() == [23 * 3600 + 50 * 60, 24 * 3600 + 10 * 60, 25 * 3600 + 30 * 60]
    assert result["dwell_planned_s"].to_list() == [0, 5 * 60, 0]
    # runtime to next stop: dep at stop 0 (23:50:00) -> arr at stop 1 (24:10:00) = 20 min
    assert result["runtime_planned_next_s"].to_list()[0] == 20 * 60


def test_trips_table_passthrough(tmp_path):
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    write_csv(
        extract_dir / "trips.txt",
        "route_id,service_id,trip_id",
        ["100,1,1", "100,1,2"],
    )

    build_trips(duckdb.connect(), extract_dir, out_dir, logging.getLogger("test"))
    result = pl.read_parquet(out_dir / "trips.parquet")
    assert result.shape == (2, 3)
