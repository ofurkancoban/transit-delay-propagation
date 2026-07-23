"""Fetch weather data from Open-Meteo, snapped to a coarse grid.

Stops are snapped to a 0.25 degree grid cell so that one API call covers many
stops, keeping call volume orders of magnitude lower than a per-stop query.

Run as a script:
    python -m src.collect.weather_fetch --mode archive --start 2026-07-01 --end 2026-07-07
    python -m src.collect.weather_fetch --mode forecast
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import polars as pl
import requests
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("weather_fetch")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(sh)
    return logger


def snap_to_grid(lat: float, lon: float, resolution: float) -> tuple[float, float]:
    """Snap a coordinate to the center of its grid cell."""
    grid_lat = round(lat / resolution) * resolution
    grid_lon = round(lon / resolution) * resolution
    return round(grid_lat, 4), round(grid_lon, 4)


def unique_grid_cells(stops: pl.DataFrame, resolution: float) -> pl.DataFrame:
    """Given a stops table with stop_lat, stop_lon, return unique grid cell centers."""
    cells = stops.select(
        [
            ((pl.col("stop_lat") / resolution).round(0) * resolution).round(4).alias("grid_lat"),
            ((pl.col("stop_lon") / resolution).round(0) * resolution).round(4).alias("grid_lon"),
        ]
    ).unique()
    return cells


def fetch_archive(url: str, lat: float, lon: float, start_date: str, end_date: str, variables: list[str], timeout: int = 30) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "timezone": "UTC",
    }
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_forecast(url: str, lat: float, lon: float, variables: list[str], timeout: int = 30) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(variables),
        "timezone": "UTC",
    }
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch grid-cell weather from Open-Meteo")
    parser.add_argument("--mode", choices=["archive", "forecast"], required=True)
    parser.add_argument("--start", help="start date YYYY-MM-DD, required for archive mode")
    parser.add_argument("--end", help="end date YYYY-MM-DD, required for archive mode")
    parser.add_argument(
        "--stops-parquet",
        default=None,
        help="path to a parquet file with stop_lat, stop_lon columns; defaults to skipping if absent",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    config = load_config()
    weather_cfg = config["weather"]
    logger = setup_logging()

    url_env = weather_cfg["archive_url_env"] if args.mode == "archive" else weather_cfg["forecast_url_env"]
    url = os.environ.get(url_env)
    if not url:
        raise SystemExit(f"environment variable {url_env} is not set")

    stops_path = Path(args.stops_parquet) if args.stops_parquet else None
    if stops_path is None or not stops_path.exists():
        logger.warning(
            "no stops parquet provided or found; run schedule.py first to produce normalised stops, then re-run this script"
        )
        return

    stops = pl.read_parquet(stops_path)
    cells = unique_grid_cells(stops, weather_cfg["grid_resolution_degrees"])
    logger.info("resolved %d unique grid cells from %d stops", cells.height, stops.height)

    out_dir = REPO_ROOT / "data" / "interim" / "weather"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for row in cells.iter_rows(named=True):
        lat, lon = row["grid_lat"], row["grid_lon"]
        if args.mode == "archive":
            if not args.start or not args.end:
                raise SystemExit("--start and --end are required for archive mode")
            data = fetch_archive(url, lat, lon, args.start, args.end, weather_cfg["variables"])
        else:
            data = fetch_forecast(url, lat, lon, weather_cfg["variables"])

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        for i, ts in enumerate(times):
            rec = {"grid_lat": lat, "grid_lon": lon, "time": ts}
            for var in weather_cfg["variables"]:
                rec[var] = hourly.get(var, [None] * len(times))[i]
            records.append(rec)

    if not records:
        logger.warning("no weather records fetched")
        return

    result = pl.DataFrame(records)
    out_path = out_dir / f"weather_{args.mode}.parquet"
    result.write_parquet(out_path)
    logger.info("wrote %d weather rows to %s", result.height, out_path)


if __name__ == "__main__":
    main()
