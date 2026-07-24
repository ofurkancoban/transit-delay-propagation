#!/usr/bin/env bash
# Build the full Phase 2 pipeline (normalised schedule + realisation panel) in one command.
#
# Usage: scripts/build_panel.sh <static-date> [rt-glob]
# Example: scripts/build_panel.sh 2026-07-23

set -euo pipefail

STATIC_DATE="${1:?usage: build_panel.sh <static-date> [rt-glob]}"
RT_GLOB="${2:-data/rt/date=*/hour=*/*.parquet}"

cd "$(dirname "$0")/.."
source .venv/bin/activate

python -m src.build.schedule --static-date "$STATIC_DATE"
python -m src.build.realisation --static-date "$STATIC_DATE" --rt-glob "$RT_GLOB"
