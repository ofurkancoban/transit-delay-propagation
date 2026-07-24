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
import time
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


def request_with_retry(url: str, params: dict, timeout: int, max_retries: int = 5) -> dict | list:
    """GET with exponential backoff on 429 (Open-Meteo's free tier rate limit)."""
    delay = 5
    for attempt in range(max_retries):
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 429 and attempt < max_retries - 1:
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"exhausted retries against {url}")


def fetch_archive_batch(
    url: str, lats: list[float], lons: list[float], start_date: str, end_date: str, variables: list[str], timeout: int = 60
) -> list[dict]:
    """Fetch multiple grid cells in a single request using Open-Meteo's batched coordinate lists.

    Open-Meteo accepts comma-separated latitude/longitude lists and returns a
    JSON array (one object per coordinate pair) instead of a single object,
    which cuts call volume by the batch size compared to one request per cell.
    """
    params = {
        "latitude": ",".join(str(v) for v in lats),
        "longitude": ",".join(str(v) for v in lons),
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "timezone": "UTC",
    }
    data = request_with_retry(url, params, timeout)
    return data if isinstance(data, list) else [data]


def fetch_forecast_batch(url: str, lats: list[float], lons: list[float], variables: list[str], timeout: int = 60) -> list[dict]:
    params = {
        "latitude": ",".join(str(v) for v in lats),
        "longitude": ",".join(str(v) for v in lons),
        "hourly": ",".join(variables),
        "timezone": "UTC",
    }
    data = request_with_retry(url, params, timeout)
    return data if isinstance(data, list) else [data]


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

    if args.mode == "archive" and (not args.start or not args.end):
        raise SystemExit("--start and --end are required for archive mode")

    batch_size = 100
    cell_list = [(row["grid_lat"], row["grid_lon"]) for row in cells.iter_rows(named=True)]

    records = []
    for i in range(0, len(cell_list), batch_size):
        batch = cell_list[i : i + batch_size]
        lats = [c[0] for c in batch]
        lons = [c[1] for c in batch]
        if args.mode == "archive":
            results = fetch_archive_batch(url, lats, lons, args.start, args.end, weather_cfg["variables"])
        else:
            results = fetch_forecast_batch(url, lats, lons, weather_cfg["variables"])

        for (lat, lon), data in zip(batch, results):
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            for j, ts in enumerate(times):
                rec = {"grid_lat": lat, "grid_lon": lon, "time": ts}
                for var in weather_cfg["variables"]:
                    rec[var] = hourly.get(var, [None] * len(times))[j]
                records.append(rec)
        logger.info("fetched batch %d-%d of %d grid cells", i, i + len(batch), len(cell_list))
        time.sleep(2)

    if not records:
        logger.warning("no weather records fetched")
        return

    result = pl.DataFrame(records)
    out_path = out_dir / f"weather_{args.mode}.parquet"
    result.write_parquet(out_path)
    logger.info("wrote %d weather rows to %s", result.height, out_path)


if __name__ == "__main__":
    main()
