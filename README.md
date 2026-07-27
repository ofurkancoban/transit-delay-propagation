# Transit Delay Propagation

Research pipeline that collects live GTFS-Realtime data from the German public
transport network, reconstructs a stop-level delay panel, and answers one
question: through which channels does delay propagate across a transit
network, and can a network-aware model beat the operator's own real-time
delay forecast.

The project has two halves:

- **Prediction:** forecast the delay increment at downstream stops and
  benchmark against the operator's own published prediction from the GTFS-RT
  feed.
- **Explanation:** decompose propagation into four channels (longitudinal,
  vehicle chain, transfer, shared infrastructure) with a spatial panel model.

## Important note on the target variable

The target is the **delay increment between consecutive stops**, not the raw
delay level. Raw delay is almost entirely autoregressive: predicting it
yields a meaningless R2 above 0.95 while the model learns nothing beyond
"delay stays the same." All model evaluation in this repository is reported
on the increment.

## Status

**Phase 1 (collection) is complete.** The collector ran under systemd on a
VPS with `Restart=always` for 24h 21m (2026-07-23 15:06 to 2026-07-24 15:36
CEST) with zero process restarts and zero logged errors, producing 44.4M
raw snapshot rows across 2,859 poll files. Two brief upstream feed-side
stalls (6 and 10 minutes) were observed and excluded from the realisation
panel per pitfall 6; neither involved the collector process going down.
**Phase 2 (schedule normalisation and realisation panel) is complete.**
`scripts/build_panel.sh <static-date>` builds both output tables end to end
from the archived static feed and the raw snapshot lake in a single command,
producing `panel_predictions.parquet` (30.9M rows, the operator benchmark)
and `panel_realised.parquet` (9.09M stop-events, 504,503 distinct trips).
Row counts, delay distributions and prediction-horizon coverage are sanity
checked in `notebooks/02_descriptives.ipynb`, which also verifies five
real midnight-crossing trips (raw GTFS times of `24:00:00` and later) by
hand against the resulting absolute timestamps.

**Phase 3 (feature engineering) is complete.** `src/build/features.py`
builds `data/processed/features.parquet` (7.74M rows) with the delay
increment target (`y_delay_increment`, not the raw delay level) plus trip
state, network state and exogenous feature blocks. The vehicle-chain block
is skipped entirely since `vehicle_id` is never populated (pitfall 4).
Network-state aggregates use DuckDB window frames that only look backward
from each row's actual observed arrival time, and the upcoming-stop
scheduling-pressure feature is computed purely from the static schedule,
so neither carries realtime leakage; `tests/test_features.py` includes a
structural leakage test plus unit tests for the school holiday calendar
and the stop-pressure lookup. Weather (`src/collect/weather_fetch.py`,
now batched to respect Open-Meteo's rate limit) and Lower Saxony school
holidays (`config/school_holidays_lower_saxony.json`, hand-sourced per the
working agreement) are both wired in as exogenous features. Feature
distributions are documented in `notebooks/02_descriptives.ipynb`,
including a note on a rare (<0.01% of rows) GTFS-RT producer quirk where a
handful of TripUpdates report a delay offset by exactly 24 hours.

**Phase 4 (models) is complete.** All six tiers ran end to end on the
time-ordered split (never random, `src/models/evaluate.py:time_based_split`):
persistence and the operator's own live forecast (`src/models/baselines.py`),
a historical-mean baseline, LightGBM (`src/models/lgbm.py`, `regression_l1`
objective, not the L2 default, since a squared-error objective is badly
distorted by the rare +/-86400s producer-quirk outliers), a small GRU over
each trip's stop sequence (`src/models/sequence.py`), and a minimal
spatio-temporal GNN (`src/models/stgnn.py`, hand-written message passing in
plain PyTorch rather than `torch-geometric`, see that module's docstring).
Results, the horizon-bucket comparison table, skill scores against the
operator, PR-AUC for delay>6min, peak/mode/density breakdowns and the
edge-type ablation (the headline experiment) are all in
`notebooks/03_model_results.ipynb`.

**Headline result, reported honestly:** no tier beats the operator's own
live forecast yet (operator MAE 7.99s vs LightGBM's 10.27s, the best of
this repo's own models on the full test split), though LightGBM and the
STGNN both clearly beat the naive baselines. The ablation shows the
longitudinal (`sched_adj`) channel dominates; the vehicle-chain (`block`)
channel could not be built at all (`vehicle_id` is never populated,
pitfall 4); transfer and shared-infrastructure channels show only a small
effect at the current, deliberately subsampled graph coverage
(`src/build/graph.py`, 30,000 of 264,933 nationwide stops).

**Diagnosis and fix attempt (see `notebooks/03_model_results.ipynb`):**
re-slicing the test split by how fresh the operator's own prediction was
showed the gap is not primarily a tuning or data-volume problem. Outside
rows where the operator's prediction had just changed in the last minute
("volatile" delay events, 2.2% of the test split), LightGBM already
matches or beats the operator; the entire aggregate gap concentrates in
that 2.2%, where LightGBM's error is over 10x the operator's. Five
volatility features derived purely from existing TripUpdates history
(`delay_jump_1`, `delay_jump_1_abs`, `delay_jump_2`, `delay_recent_std`,
`delay_vs_route_recent_gap`, see `src/build/features.py`) were engineered
and adopted into production: they give a small, real, uniform improvement
(~0.5% overall MAE) but essentially none (0.1%) on the volatile regime
specifically, confirming the gap there is structural rather than fixable
by feature engineering. A live pull of the primary feed confirms it
publishes **zero** `VehiclePosition` entities (only TripUpdate and
Alert), so whatever lets the operator track a delay the instant it starts
is not exposed on this feed at all. More collection days would help the
calmer 94% of rows (still warming up on a single day) but cannot close
the volatile-event gap, which is a hard ceiling of `realtime.gtfs.de`
itself, not something more tuning or feature engineering can reach.

**Breakdown by agency (company/region), `src/models/agency_breakdown.py`:**
the aggregate "operator wins" result is not uniform across the feed's
~470 agencies (GTFS has no city field, but the largest agencies are named
after the region they serve, so agency grouping doubles as a practical
city proxy). LightGBM beats the operator at real scale in several
regions, e.g. Verkehrsverbund Stuttgart (n=45,031, skill +0.20) and
S-Bahn Berlin (n=18,833, skill +0.69), while losing badly in others, e.g.
Verkehrsverbund Rhein-Neckar (n=58,202, skill -1.19). See
`notebooks/03_model_results.ipynb` for the full ranking; a production
rollout should pick operator vs. model per agency rather than globally.

**ServiceAlerts are now collected too** (`src/collect/gtfsrt_collector.py`,
deployed live to the VPS collector at 2026-07-24 21:54 CEST, confirmed
running: first poll wrote 34,960 alert rows, the very next poll 30s later
wrote only 439, confirming delta compression is working in production).
Each alert is flattened to one row per informed entity and delta-compressed
by `alert_id` (most alerts are long-lived, e.g. a multi-week construction
notice, so without this the ~35-38k-entity alert list would be rewritten
in full on every 30s poll). The live pull showed the alert stream is a mix
of boilerplate per-trip attribution notices (cause/effect both
`UNKNOWN`) and genuine disruption alerts (`CONSTRUCTION`,
`TECHNICAL_PROBLEM`, `POLICE_ACTIVITY`, `MEDICAL_EMERGENCY`, route
diversions) with real `header_text`/`description_text` and route/stop
targeting, so it is a real candidate signal for the volatile-delay blind
spot identified above. Not yet tested against the model, since it only
starts accruing history from the deployment time forward and the existing
Phase 2-4 pipeline was built on 2026-07-23's window, before this existed.
Written to a separate partitioned lake at
`data/rt_alerts/date=YYYY-MM-DD/hour=HH/`, with unit tests in
`tests/test_gtfsrt_collector.py`.

**The live dashboard now actually updates itself.** `.github/workflows/refresh-dashboard.yml`
runs every 10 minutes, connects to the VPS collector over a dedicated,
restricted SSH key (forced to run only `scripts/vps_stats.py`, read-only,
no shell, no port forwarding), and commits fresh numbers to
`docs/stats.json`. The dashboard page fetches that file client-side every
60 seconds and updates the status badge, collector uptime, and live
row/alert counts, falling back to "stats unavailable" rather than
breaking if the feed is stale. Verified end to end: a manual workflow run
produced a real commit with fresh collector numbers, and GitHub Pages
served the updated file within seconds.

Live dashboard: https://ofurkancoban.github.io/transit-delay-propagation/

**Phase 5 (econometrics) is complete.** `src/econ/weights.py` builds
row-standardised spatial weight matrices from the same stop-level graph
used for the Phase 4 GNN ablation (`sched_adj`, `shared_segment`,
`transfer`, and a combined matrix; `block` remains unbuildable since
`vehicle_id` is never populated). `src/econ/sdm.py` fits a Spatial Durbin
Model (`spreg.ML_LagFE`, `slx_lags=1`, stop fixed effects) on a balanced
1,500-stop x 24-hour delay panel, reports LM tests for spatial lag vs.
error dependence (LM-error is significant, p<0.02, across all three
weight specs; LM-lag never is, a genuine finding) and LeSage-Pace
direct/indirect/total effects. `src/econ/local_projection.py` estimates
a 25-point Jorda local projection (0-120 min in 5-minute steps, HAC
standard errors) of the impulse response of network-wide delay to a
nationwide precipitation shock, with a computed half-life of ~85
minutes. All three modules ran against real pipeline data (not
placeholders), with two real engineering fixes along the way: 89
self-loop edges in the stop graph were dropped (a genuine data-quality
bug, not present in the Phase 4 ablation since it only affects
diagonal-sensitive spatial-weight code), and a spreg version bug
(`check_constant` indexing past a short variable-name list against a
wide panel matrix) was worked around with a documented, narrowly-scoped
monkeypatch. Results, the effect decomposition table, the IRF plot with
confidence bands, and a candid section on what is and is not identified
from a single day of data are in `notebooks/04_econometrics.ipynb`.

## Repository layout

See `config.yaml` for all tunable parameters (feed URLs, poll interval,
weather grid resolution, model horizon buckets). No constants are scattered
in code.

```
src/collect/   feed polling, static schedule archival, weather fetch
src/build/     static-to-parquet normalisation, realisation panel, features, graph
src/models/    baselines, LightGBM, sequence model, spatio-temporal GNN
src/econ/      spatial weight matrices, spatial Durbin model, local projections
notebooks/     data quality, descriptives, model results, econometrics
data/          gitignored parquet lake and versioned static snapshots
```

## Running the collector

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in feed URLs if not using the defaults
python -m src.collect.gtfsrt_collector --feed primary
```

In production this runs under systemd with `Restart=always`
(`scripts/systemd/gtfsrt-collector.service`). The static schedule is
refetched weekly via `scripts/systemd/gtfs-static-fetch.timer`, since the
free gtfs.de feeds are valid for about 7 days.

## Known pitfalls this repo is designed around

1. Static feed versioning: realtime rows must be joined against the static
   snapshot whose validity window contains the service date, never a stale one.
2. GTFS times past midnight (`25:30:00`) must parse correctly.
3. Some producers omit `stop_sequence`; those rows are dropped and the loss
   is quantified, not silently absorbed.
4. `vehicle_id` can be unstable or absent per agency; the vehicle-chain
   channel is only estimated on the subset with stable identifiers.
   **Measured on the primary feed (`realtime.gtfs.de`): `vehicle.id` is never
   populated (0 of 134,795 trips observed as of 2026-07-23, confirmed against
   the raw protobuf, not a collector bug).** The vehicle-chain / block
   propagation channel is not estimable from this feed, and is dropped from
   the edge-type ablation unless the secondary VBB feed turns out to carry it
   and passes its own completeness check.

   Direction is equally unavailable from either source: the static feed's
   `trips.txt` omits `direction_id` and `block_id` entirely (header is just
   `route_id,service_id,trip_id`), and the RT feed's `TripDescriptor.direction_id`
   is present as a column but is always `-1` (the GTFS-RT "unset" sentinel,
   confirmed across 16.3M rows). "Same route and direction" network features
   therefore reduce to "same route" (joined via `trip_id` against the static
   `trips.txt`, since RT's own `route_id` field is always empty and cannot be
   used directly) unless direction is inferred from stop sequence or
   origin/destination stop.
5. Cancelled and added trips are handled via `schedule_relationship`, never
   treated as infinitely delayed trips.
6. Collector downtime windows are logged and excluded from the realisation
   panel to avoid survivorship bias.
7. Delay distributions are heavy-tailed and asymmetric; median absolute
   error is always reported alongside the mean.
8. All timestamps are stored in UTC internally and converted to
   Europe/Berlin only at the presentation layer.
