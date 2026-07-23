# GOAL: Public Transport Delay Propagation, Prediction and Spillover Estimation

## 1. Objective

Build a reproducible research pipeline that collects live GTFS-Realtime data from the
German public transport network, reconstructs a stop-level delay panel, and answers
one question:

> Through which channels does delay propagate across a transit network, and can a
> network-aware model beat the operator's own real-time delay forecast?

The project has two halves that must both work:

- **Prediction half (machine learning):** forecast the delay increment at downstream
  stops given the current network state, and benchmark against the operator's own
  published prediction.
- **Explanation half (econometrics):** decompose propagation into distinct channels
  and estimate direct versus indirect (spillover) effects with a spatial panel model
  on a network weight matrix.

The deliverable is a working repository plus a results notebook, not a deployed
product. Treat this as a research codebase: correctness, reproducibility and clear
data lineage matter more than throughput or UI.

## 2. Why the design looks the way it does

Four propagation channels are hypothesised, and separating them is the actual
scientific contribution. Most published work only models the first one:

1. **Longitudinal:** delay carries to later stops of the same trip.
2. **Vehicle chain (block):** a late vehicle starts its next trip late when the
   turnaround buffer is too small.
3. **Transfer:** at hubs, a delayed feeder line pushes delay onto connecting lines.
4. **Shared infrastructure:** unrelated lines using the same road segment or platform
   correlate through congestion.

The GTFS-RT feed publishes the operator's own predicted delay for future stops.
That prediction is a free, strong, honest benchmark. "Our model beats the agency's
own live forecast at a 20 minute horizon" is the headline result to aim for.

## 3. Critical constraint: data accrues in real time

The feed exposes only the present state. There is no historical archive. Nothing
downstream can be validated until data has accumulated. Therefore:

**Phase 1 (the collector) must be finished, verified and running in the background
before any modelling work begins.** Get it collecting on day one, then build the rest
while data accrues. Do not spend the first session on model architecture.

## 4. Data sources

### 4.1 Realtime (primary)

- URL: `https://realtime.gtfs.de/realtime-free.pb`
- Format: GTFS-Realtime protobuf, TripUpdates and ServiceAlerts
- Auth: none
- Licence: CC BY-SA 4.0
- Poll interval: 30 seconds

### 4.2 Realtime (secondary, for cross-region comparison only)

- URL: `https://production.gtfsrt.vbb.de/data` (Berlin-Brandenburg)
- Auth: none, rate limit 60 requests per minute, CC BY 4.0
- **Known issue:** this feed has been publishing incomplete data since 2026-06-04
  with no announced fix. Verify data completeness before relying on it. Do not make
  it the primary source.

### 4.3 Static schedule

- gtfs.de free feeds, Germany-wide, generated daily from the DELFI dataset
- The free feeds have a validity of about 7 days, so they must be downloaded and
  archived weekly under a versioned directory
- Fetch the current download URL from `https://gtfs.de/en/feeds/` at implementation
  time rather than hardcoding a guessed path

### 4.4 Weather

- Open-Meteo, no API key. Use the historical/archive endpoint for backfill and the
  forecast endpoint for live features.
- Variables: precipitation, temperature_2m, wind_speed_10m, wind_gusts_10m, snowfall
- Do not query per stop. Snap stops to a coarse grid (0.25 degrees is enough) and
  query one series per grid cell, then join. This reduces call volume by orders of
  magnitude and stays within polite usage.

### 4.5 Calendar

- German public holidays and Lower Saxony school holidays. Use the `holidays` Python
  package for public holidays; source school holidays from a static JSON you commit
  to the repo rather than a live API.

## 5. Repository layout

```
transit-delay-propagation/
  README.md
  pyproject.toml
  .env.example
  src/
    collect/
      gtfsrt_collector.py       # already written, integrate as is
      static_gtfs_fetch.py      # weekly versioned static feed download
      weather_fetch.py
    build/
      schedule.py               # static GTFS to normalised parquet
      realisation.py            # snapshots to realised delay panel
      features.py               # feature engineering
      graph.py                  # network graph construction
    models/
      baselines.py
      lgbm.py
      sequence.py
      stgnn.py
      evaluate.py
    econ/
      weights.py                # spatial weight matrices
      sdm.py                    # spatial Durbin estimation
      local_projection.py       # impulse response
  notebooks/
    01_data_quality.ipynb
    02_descriptives.ipynb
    03_model_results.ipynb
    04_econometrics.ipynb
  data/                         # gitignored
    rt/                         # partitioned parquet lake
    static/<YYYY-MM-DD>/        # versioned schedule snapshots
    interim/
    processed/
  tests/
  scripts/
    run_collector.sh
    systemd/gtfsrt-collector.service
```

## 6. Tech stack

- Python 3.11+
- `gtfs-realtime-bindings`, `requests`, `pyarrow`, `duckdb`, `polars`
- `lightgbm`, `scikit-learn`, `torch`, `torch-geometric`
- `libpysal` and `spreg` for spatial econometrics, `statsmodels` for local projections
- `pytest` for tests, `ruff` for linting

No database server. DuckDB queries the Parquet lake directly.

## 7. Phases

### Phase 1: Collection (blocking, do first)

The collector `gtfsrt_collector.py` is already written and unit tested. It polls the
feed, applies delta compression (only rows whose delay values changed since the last
poll are written) and appends to a partitioned Parquet lake at
`data/rt/date=YYYY-MM-DD/hour=HH/`. Integrate it without rewriting it.

Tasks:
- Add `static_gtfs_fetch.py`: download the current static feed, verify it unzips and
  passes basic integrity checks, store under `data/static/<download-date>/`, and write
  a `manifest.json` recording feed URL, download timestamp, checksum, and the
  `feed_start_date` / `feed_end_date` from `feed_info.txt`.
- Add a systemd unit with `Restart=always` and a weekly timer for the static fetch.
- Add `notebooks/01_data_quality.ipynb`: poll success rate, feed lag distribution
  (poll timestamp minus feed header timestamp), active trips per hour, share of trips
  with a usable `vehicle_id`, share of stop_time_updates carrying `stop_sequence`.

**Acceptance:** collector running under systemd, at least 24 hours of continuous data
in the lake, static feed archived with a manifest, data quality notebook produced.

### Phase 2: Schedule normalisation and realisation panel

- `schedule.py`: parse static GTFS into normalised Parquet tables. Compute for each
  `(trip_id, stop_sequence)` the scheduled arrival and departure as absolute
  timestamps, handling GTFS times past 24:00:00 correctly. Compute planned dwell and
  planned run time to the next stop. Attach stop coordinates.
- `realisation.py`: from the snapshot lake, extract the realised delay per
  `(service_date, trip_id, stop_sequence)` as the last observed value before the
  scheduled time passed, plus the full prediction history keyed by horizon.

Two output tables:

`panel_realised`
```
service_date, trip_id, route_id, direction_id, vehicle_id, stop_id,
stop_sequence, sched_arr_ts, sched_dep_ts, realised_arr_delay,
realised_dep_delay, dwell_planned, runtime_planned_next, stop_lat, stop_lon
```

`panel_predictions`
```
service_date, trip_id, stop_sequence, poll_ts, horizon_s, predicted_arr_delay
```
where `horizon_s = sched_arr_ts - poll_ts`. This table is the operator benchmark.

**Acceptance:** both tables build end to end from raw inputs with a single command,
row counts and delay distributions sanity checked in a notebook, midnight-crossing
trips verified by hand on at least five examples.

### Phase 3: Feature engineering

Target variable: **the delay increment between consecutive stops**, not the raw delay.

```
y = realised_dep_delay(s+k) - realised_dep_delay(s)
```

Raw delay is almost entirely autoregressive; predicting it yields a meaningless R2
above 0.95 while learning nothing. State this explicitly in the README.

Feature blocks:

*Trip state*
current delay, delay at the previous 3 stops, delay slope, stops remaining, distance
remaining, elapsed share of trip, planned dwell, planned run time, **slack** defined
as planned run time minus the historical median observed run time for that segment
and time band.

*Network state (all computed as of prediction time, no leakage)*
mean delay of other vehicles on the same route and direction in the last 15 minutes,
mean delay of all trips traversing the same stop-pair segment in the last 15 minutes,
delay level at the next hub on the route, count of trips currently scheduled at the
upcoming stop within a 5 minute window.

*Vehicle chain*
arrival delay of the same `vehicle_id` on its previous trip, scheduled turnaround
buffer, indicator for whether the buffer is below the historical 25th percentile.

*Exogenous*
weather at the grid cell of the current stop, hour, day of week, public holiday,
school holiday, peak indicator.

**Leakage discipline is the single most important requirement in this phase.**
Every feature must be computable from information available strictly before
`poll_ts`. Write a test that, for a random sample of rows, asserts no input
timestamp exceeds the prediction timestamp. Split train/validation/test by time,
never randomly.

**Acceptance:** feature table builds reproducibly, leakage test passes, feature
distributions documented.

### Phase 4: Models

Implement in this order and report all of them in one comparison table:

| Tier | Model | Purpose |
|---|---|---|
| 0 | Persistence, delay stays constant | Lower bound |
| 0 | **Operator's own prediction from `panel_predictions`** | The real benchmark |
| 0 | Historical mean by (route, stop, hour, day of week) | Classical baseline |
| 1 | LightGBM on the tabular feature set | Main workhorse |
| 2 | GRU or small Transformer over the trip's stop sequence | Longitudinal structure |
| 3 | Spatio-temporal GNN | Network propagation |

Evaluation:
- MAE and RMSE on the delay increment, reported **separately by horizon bucket**
  (0 to 5, 5 to 15, 15 to 30, 30 to 60 minutes). Aggregate metrics hide everything.
- Classification metric: probability that delay exceeds 6 minutes at the target stop,
  reported as PR-AUC since the class is imbalanced.
- Skill score against the operator benchmark: `1 - MAE_model / MAE_operator`.
- Breakdown by mode (bus, tram, regional rail), by peak versus off-peak, and by
  urban versus rural stop density.

For the GNN, build a graph where nodes are stops (or route-stop segments) and edges
carry a type label:
- `sched_adj`: consecutive stops on a trip
- `transfer`: stop pairs within 300 metres with a feasible connection in the schedule
- `block`: linked by shared vehicle assignment
- `shared_segment`: distinct routes traversing the same stop pair

**The headline experiment is an edge-type ablation.** Retrain the GNN with each edge
type removed and report the performance drop. That table quantifies which propagation
channel actually carries delay, which is the paper's contribution.

**Acceptance:** single comparison table across all tiers and horizon buckets,
ablation table produced, all runs seeded and reproducible.

### Phase 5: Econometrics

- `weights.py`: build row-standardised spatial weight matrices from each edge type.
  Provide a combined matrix and per-channel matrices.
- `sdm.py`: estimate a Spatial Durbin Model on the stop-time panel with time and
  stop fixed effects. Report direct, indirect and total effects with correct
  standard errors. Compare across weight specifications and run a Lagrange multiplier
  specification test (SAR versus SEM versus SDM).
- `local_projection.py`: Jorda local projections. Estimate the impulse response of
  network-wide delay to a delay shock at a hub, horizons 0 to 24 in 5 minute steps.
  Report the half-life of a shock. Use ServiceAlerts and weather shocks as plausibly
  exogenous variation and discuss identification honestly rather than overclaiming.

**Acceptance:** effect decomposition table, impulse response plots with confidence
bands, an explicit and candid section on what is and is not identified.

## 8. Known pitfalls, address them proactively

1. **Static feed versioning.** The free static feeds expire after about 7 days. If
   realtime data from week 3 is joined against the week 1 schedule, every delay
   computation silently breaks. Always join realtime to the static version whose
   validity window contains the service date. Assert this in code and fail loudly
   if no matching version exists.
2. **GTFS times past midnight.** `stop_times.txt` legitimately contains values such
   as `25:30:00`. Naive time parsing corrupts the entire night service. Test this.
3. **Missing `stop_sequence`.** Some producers omit it. Those rows cannot be keyed
   reliably and are dropped by the collector. Quantify how many are lost.
4. **`vehicle_id` instability.** Some agencies rotate identifiers daily or omit them.
   The block channel is only estimable on the subset where identifiers are stable.
   Measure that subset before building on it.
5. **Cancelled and added trips.** Honour `schedule_relationship`. Cancelled trips are
   not infinitely delayed trips; exclude them from the regression sample and analyse
   them separately.
6. **Survivorship in the realisation panel.** If the collector was down, trips have
   truncated histories. Record collector uptime and exclude affected windows.
7. **Delay distributions are heavy-tailed and asymmetric.** Vehicles cannot depart
   early by much but can be arbitrarily late. Consider quantile loss or an asymmetric
   objective, and always report median absolute error alongside the mean.
8. **Timezones.** The feed uses Unix timestamps, GTFS uses local time. Standardise on
   UTC internally and convert to Europe/Berlin only at the presentation layer. Watch
   the daylight saving transition.

## 9. Working agreements

- All code, comments, docstrings, variable names, commit messages and documentation
  in **English**.
- **Never use em dashes** in any generated text, code comment, or documentation.
- **Never add Claude, Anthropic or any AI assistant as a co-author, contributor or
  trailer in commits.** Commit messages contain no `Co-Authored-By` line and no
  generated-with attribution.
- Conventional commit prefixes: `feat:`, `fix:`, `data:`, `docs:`, `test:`, `refactor:`.
- Every pipeline stage is a callable module with a CLI entry point, so any stage can
  be rerun independently.
- Configuration lives in a single `config.yaml`, not scattered constants.
- Write tests for every data transformation that involves time arithmetic. Those are
  where the bugs will be.
- Keep `data/` out of version control. Commit manifests and schemas, never payloads.
- Log every long-running job to a file with counts in and counts out, so silent row
  loss is visible.

## 10. Suggested first session

1. Set up the repo skeleton, `pyproject.toml`, `config.yaml`, `.gitignore`.
2. Drop in the existing collector, add the systemd unit, start it, confirm Parquet
   files appearing with sensible row counts.
3. Implement `static_gtfs_fetch.py` and archive the first schedule version.
4. Write the data quality notebook against the first few hours of data.
5. Stop there. Do not start modelling until at least a week of data exists.

Report at the end of the session: rows collected, feed lag distribution, share of
trips with usable `vehicle_id` and `stop_sequence`, and any surprises in the feed
contents that would change the plan.
