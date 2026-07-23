#!/usr/bin/env bash
# Restart-always wrapper for local development machines without systemd.
# Equivalent in spirit to systemd's Restart=always for gtfsrt-collector.service.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

while true; do
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) starting collector" >> logs/watchdog.log
  ./scripts/run_collector.sh >> logs/collector.stdout.log 2>> logs/collector.stderr.log
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) collector exited, restarting in 10s" >> logs/watchdog.log
  sleep 10
done
