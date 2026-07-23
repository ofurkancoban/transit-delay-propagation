#!/usr/bin/env bash
# Launch the GTFS-RT collector for the primary feed.
# Intended to be invoked by systemd (see scripts/systemd/gtfsrt-collector.service)
# or run directly in a foreground/background shell for local development.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

exec python -m src.collect.gtfsrt_collector --feed primary
