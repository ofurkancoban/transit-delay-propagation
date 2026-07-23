"""Download and archive the current gtfs.de static feed.

The free feeds expire after about 7 days. This script fetches the current
download URL from the gtfs.de feeds page rather than hardcoding it, downloads
the zip, verifies it unzips and contains the required GTFS files, stores it
under data/static/<download-date>/, and writes a manifest.json with the feed
URL, download timestamp, checksum, and feed validity window.

Run as a script:
    python -m src.collect.static_gtfs_fetch
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_FILES = {
    "stops.txt",
    "routes.txt",
    "trips.txt",
    "stop_times.txt",
    "calendar.txt",
}


def load_config() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("static_gtfs_fetch")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(sh)
    return logger


def resolve_download_url(index_url: str, feed_name_hint: str, timeout: int = 20) -> str:
    """Resolve the current Germany-wide static feed download link from gtfs.de.

    The feeds index page links to a per-feed detail page (e.g.
    /en/feeds/de_full/) rather than exposing the .zip link directly. That
    detail page contains the actual protocol-relative download URL, which
    also serves as a stable "latest" pointer that gtfs.de keeps up to date.
    """
    resp = requests.get(index_url, timeout=timeout)
    resp.raise_for_status()
    html = resp.text

    detail_links = re.findall(r'href="(/en/feeds/[^"]+/)"', html)
    if not detail_links:
        raise RuntimeError(f"no feed detail links found on {index_url}")

    hinted = [link for link in detail_links if feed_name_hint.lower() in link.lower()]
    detail_link = hinted[0] if hinted else detail_links[0]

    base = index_url.split("/en/feeds", 1)[0]
    detail_url = f"{base}{detail_link}"

    detail_resp = requests.get(detail_url, timeout=timeout)
    detail_resp.raise_for_status()
    detail_html = detail_resp.text

    zip_links = re.findall(r'href="([^"]+\.zip)"', detail_html)
    if not zip_links:
        raise RuntimeError(f"no .zip link found on {detail_url}")
    chosen = zip_links[0]

    if chosen.startswith("//"):
        return f"https:{chosen}"
    if chosen.startswith("http"):
        return chosen
    return f"{base}/{chosen.lstrip('/')}"


def download_feed(url: str, dest_zip: Path, timeout: int = 120) -> str:
    resp = requests.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()
    sha256 = hashlib.sha256()
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_zip, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_and_extract(dest_zip: Path, extract_dir: Path) -> None:
    with zipfile.ZipFile(dest_zip) as zf:
        bad_file = zf.testzip()
        if bad_file is not None:
            raise RuntimeError(f"corrupt member in zip: {bad_file}")
        names = set(zf.namelist())
        missing = REQUIRED_FILES - names
        if missing:
            raise RuntimeError(f"static feed missing required files: {missing}")
        extract_dir.mkdir(parents=True, exist_ok=True)
        zf.extractall(extract_dir)


def parse_feed_info(extract_dir: Path) -> tuple[str | None, str | None]:
    """Return the feed's validity window as (feed_start_date, feed_end_date).

    Prefer feed_info.txt when it carries those optional fields. Some
    producers (including gtfs.de) omit them, so fall back to the min/max of
    calendar.txt's start_date/end_date columns, which are required fields.
    """
    import csv

    feed_info_path = extract_dir / "feed_info.txt"
    if feed_info_path.exists():
        with open(feed_info_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
            if row and row.get("feed_start_date") and row.get("feed_end_date"):
                return row["feed_start_date"], row["feed_end_date"]

    calendar_path = extract_dir / "calendar.txt"
    if not calendar_path.exists():
        return None, None

    start_dates: list[str] = []
    end_dates: list[str] = []
    with open(calendar_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("start_date"):
                start_dates.append(row["start_date"])
            if row.get("end_date"):
                end_dates.append(row["end_date"])

    if not start_dates or not end_dates:
        return None, None
    return min(start_dates), max(end_dates)


def write_manifest(
    archive_dir: Path,
    feed_url: str,
    download_ts: str,
    checksum: str,
    feed_start_date: str | None,
    feed_end_date: str | None,
) -> None:
    manifest = {
        "feed_url": feed_url,
        "download_timestamp_utc": download_ts,
        "sha256": checksum,
        "feed_start_date": feed_start_date,
        "feed_end_date": feed_end_date,
    }
    with open(archive_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    logger = setup_logging()
    config = load_config()
    static_cfg = config["static_feed"]

    feed_url = resolve_download_url(static_cfg["index_url"], static_cfg["feed_name_hint"])
    logger.info("resolved static feed url: %s", feed_url)

    download_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    archive_dir = REPO_ROOT / static_cfg["archive_dir"] / download_date
    dest_zip = archive_dir / "gtfs.zip"

    checksum = download_feed(feed_url, dest_zip)
    logger.info("downloaded %s, sha256=%s", dest_zip, checksum)

    extract_dir = archive_dir / "extracted"
    verify_and_extract(dest_zip, extract_dir)
    logger.info("verified and extracted to %s", extract_dir)

    feed_start_date, feed_end_date = parse_feed_info(extract_dir)
    download_ts = datetime.now(timezone.utc).isoformat()
    write_manifest(archive_dir, feed_url, download_ts, checksum, feed_start_date, feed_end_date)
    logger.info(
        "manifest written, feed_start_date=%s feed_end_date=%s",
        feed_start_date,
        feed_end_date,
    )


if __name__ == "__main__":
    main()
