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
Phase 2's static-side work (`src/build/schedule.py`) and the realisation
panel (`src/build/realisation.py`) are both implemented and have been run
against the full collection window, producing `panel_predictions.parquet`
(30.9M rows) and `panel_realised.parquet` (9.09M stop-events, 504,503
distinct trips). Live dashboard: https://ofurkancoban.github.io/transit-delay-propagation/

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
