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

Phase 1 (collection) is running. Nothing downstream can be validated until
enough realtime data has accumulated, since the feed exposes only the
present state and there is no historical archive.

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
5. Cancelled and added trips are handled via `schedule_relationship`, never
   treated as infinitely delayed trips.
6. Collector downtime windows are logged and excluded from the realisation
   panel to avoid survivorship bias.
7. Delay distributions are heavy-tailed and asymmetric; median absolute
   error is always reported alongside the mean.
8. All timestamps are stored in UTC internally and converted to
   Europe/Berlin only at the presentation layer.
