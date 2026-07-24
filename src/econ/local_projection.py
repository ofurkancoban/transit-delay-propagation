"""Jorda local projections: impulse response of network-wide delay to a shock at a hub.

Per the goal document, the shock must be plausibly exogenous, not the
delay itself (which is endogenous to the whole network by construction).
Weather is used as the instrument: a precipitation shock at the busiest
stop in the network (the "hub", the stop with the most realised
stop-events in `panel_realised.parquet`) at time t, and the impulse
response traces network-wide mean delay (excluding the hub itself, so
the response is not mechanically driven by including the shocked unit)
at horizons t, t+5min, ..., t+120min (25 steps of 5 minutes, matching
"0 to 24" horizons in the goal document).

For each horizon h, a separate OLS regression is estimated (the Jorda
1995 local projection approach, one model per horizon rather than a
single VAR extrapolated forward):

    y_{t+h} = alpha_h + beta_h * shock_t + gamma_h * y_{t-1} + delta_h * hour_of_day + e_{t+h}

`beta_h` traced across h is the impulse response; its standard error
(Newey-West / HAC, since residuals are serially correlated by
construction from overlapping horizons) gives the confidence band.

Identification, discussed honestly rather than overclaimed: precipitation
at the hub's grid cell is plausibly exogenous to *that specific stop's*
delay-generating process (weather is not caused by transit delay), but it
is a single grid cell's reading over a single day, with only a handful of
genuinely wet 5-minute bins in the whole window (see
notebooks/02_descriptives.ipynb's weather summary). The resulting
confidence bands are wide and this should be read as a proof of the
pipeline working correctly on real data, not as a precise, publishable
estimate of a weather-to-delay elasticity. A longer collection window
with more independent rain events is the direct fix.

Run as a script:
    python -m src.econ.local_projection
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[2]
BIN_MINUTES = 5
MAX_HORIZON_STEPS = 24
PRECIP_SHOCK_QUANTILE = 0.90


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("local_projection")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(sh)
    return logger


def pick_hub(realised: pl.DataFrame, logger: logging.Logger) -> str:
    top = realised.group_by("stop_id").agg(pl.len().alias("n")).sort("n", descending=True).head(1)
    hub = top["stop_id"][0]
    logger.info("hub stop_id=%s (%d realised stop-events, busiest in the network)", hub, top["n"][0])
    return hub


def build_network_series(realised: pl.DataFrame, hub: str, logger: logging.Logger) -> pl.DataFrame:
    """5-minute-binned network-wide mean delay, excluding the hub, plus the hub's own series."""
    binned = realised.with_columns(
        pl.col("sched_arr_ts").dt.truncate(f"{BIN_MINUTES}m").alias("bin_ts")
    )
    network = (
        binned.filter(pl.col("stop_id") != hub)
        .group_by("bin_ts")
        .agg(pl.col("realised_arr_delay").mean().alias("network_mean_delay"), pl.len().alias("network_n"))
    )
    hub_series = (
        binned.filter(pl.col("stop_id") == hub)
        .group_by("bin_ts")
        .agg(
            pl.col("realised_arr_delay").mean().alias("hub_mean_delay"),
            pl.col("stop_lat").first().alias("hub_lat"),
            pl.col("stop_lon").first().alias("hub_lon"),
        )
    )
    series = network.join(hub_series, on="bin_ts", how="inner").sort("bin_ts")
    logger.info("network series: %d 5-minute bins from %s to %s", series.height, series["bin_ts"].min(), series["bin_ts"].max())
    return series


def attach_weather_shock(series: pl.DataFrame, weather_path: Path, grid_resolution: float, logger: logging.Logger) -> pl.DataFrame:
    """Nationwide max precipitation across all grid cells, not the hub's own
    cell: the hub happened to see zero rain for the entire collection
    window (confirmed empirically), which would make a hub-local shock
    series degenerate (no variation to identify beta_h from at all). Using
    the nationwide max keeps the shock plausibly exogenous to the hub's own
    delay-generating process (it is still weather, not transit operations)
    while actually varying during this window.
    """
    weather = pl.read_parquet(weather_path)
    nationwide = (
        weather.with_columns(pl.col("time").str.to_datetime().dt.replace_time_zone("UTC").alias("hour_ts"))
        .group_by("hour_ts")
        .agg(pl.col("precipitation").max().alias("precipitation"))
    )
    if nationwide.height == 0 or nationwide["precipitation"].max() == 0:
        logger.warning("no nationwide precipitation variation in this window either; shock will be all-zero")

    series = series.with_columns(pl.col("bin_ts").dt.truncate("1h").alias("hour_ts"))
    series = series.join(nationwide, on="hour_ts", how="left").with_columns(pl.col("precipitation").fill_null(0.0))

    threshold = series["precipitation"].quantile(PRECIP_SHOCK_QUANTILE)
    threshold = threshold if threshold and threshold > 0 else 0.01
    series = series.with_columns((pl.col("precipitation") >= threshold).cast(pl.Int32).alias("shock"))
    logger.info(
        "precipitation shock: threshold=%.3fmm (p%d), %d of %d bins shocked (%.1f%%)",
        threshold, int(PRECIP_SHOCK_QUANTILE * 100), series["shock"].sum(), series.height, 100 * series["shock"].mean(),
    )
    return series


def run_local_projections(series: pl.DataFrame, logger: logging.Logger) -> pl.DataFrame:
    df = series.sort("bin_ts").with_columns(
        pl.col("network_mean_delay").fill_null(strategy="forward").fill_null(0.0),
        pl.col("bin_ts").dt.convert_time_zone("Europe/Berlin").dt.hour().alias("hour_of_day"),
    )
    y = df["network_mean_delay"].to_numpy()
    shock = df["shock"].to_numpy().astype(float)
    y_lag1 = np.roll(y, 1)
    y_lag1[0] = y[0]
    hour_of_day = df["hour_of_day"].to_numpy()

    rows = []
    n = len(y)
    for h in range(0, MAX_HORIZON_STEPS + 1):
        y_lead = np.roll(y, -h)
        valid = np.arange(n) < (n - h)
        valid &= np.arange(n) >= 1

        x = np.column_stack([shock[valid], y_lag1[valid], hour_of_day[valid]])
        x = sm.add_constant(x)
        model = sm.OLS(y_lead[valid], x).fit(cov_type="HAC", cov_kwds={"maxlags": max(1, h)})

        beta = model.params[1]
        se = model.bse[1]
        rows.append({
            "horizon_minutes": h * BIN_MINUTES,
            "beta": beta,
            "se": se,
            "ci_low": beta - 1.96 * se,
            "ci_high": beta + 1.96 * se,
            "n_obs": int(valid.sum()),
        })
    result = pl.DataFrame(rows)
    logger.info("estimated %d local projections (0 to %d minutes, %d-minute steps)", len(rows), MAX_HORIZON_STEPS * BIN_MINUTES, BIN_MINUTES)
    return result


def half_life(irf: pl.DataFrame) -> float | None:
    """First horizon *after* the peak absolute response at which the
    response has decayed to less than half its peak magnitude."""
    abs_beta = irf["beta"].abs()
    peak_idx = abs_beta.arg_max()
    peak = abs_beta[peak_idx]
    if peak is None or peak == 0:
        return None
    after_peak = irf.slice(peak_idx, irf.height - peak_idx)
    below = after_peak.filter(after_peak["beta"].abs() < peak / 2)
    if below.height == 0:
        return None
    return float(below["horizon_minutes"][0])


def main() -> None:
    logger = setup_logging()
    processed_dir = REPO_ROOT / "data" / "processed"
    realised = pl.read_parquet(processed_dir / "panel_realised.parquet")

    hub = pick_hub(realised, logger)
    series = build_network_series(realised, hub, logger)
    series = attach_weather_shock(series, REPO_ROOT / "data" / "interim" / "weather" / "weather_archive.parquet", 0.25, logger)

    irf = run_local_projections(series, logger)
    irf.write_parquet(processed_dir / "local_projection_irf.parquet")

    hl = half_life(irf)
    logger.info("half-life of the shock's effect: %s minutes", hl if hl is not None else "not reached within 120min window")
    print(irf)
    logger.info("wrote local_projection_irf.parquet")


if __name__ == "__main__":
    main()
