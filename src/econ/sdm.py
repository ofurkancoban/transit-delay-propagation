"""Spatial Durbin Model on the stop-hour delay panel.

Scope, documented rather than hidden: `spreg.ML_LagFE` requires a strictly
balanced N x T panel (every stop observed in every hour), but most of the
264,933 stops in the graph are only served a few times a day and have
data in only a handful of hours (median 12 of 24). Restricting to a
genuinely balanced panel of well-observed stops keeps this tractable and
statistically meaningful: stops are ranked by how many of the 24 hours
they have at least one observation in, the top `MAX_STOPS` are kept, and
any still-missing (stop, hour) cell is filled with that stop's own daily
mean delay (a standard, documented balancing choice, not a source of new
information). The regression is further restricted to stops connected in
the graph (isolates contribute nothing to a spatial lag model).

Two-way fixed effects (stop and hour) are not fully implemented:
`ML_LagFE` only demeans the entity (stop) dimension internally. Hour is
included as covariates (peak indicator, hour dummies would need `T-1`
additional "wide" variable blocks and were judged not worth it for a
single day of within-stop time variation) rather than a full fixed-effect
set. This is the same style of scoping decision used throughout this
repository (see src/build/graph.py's transfer-edge subsampling).

The Durbin specification uses `slx_lags=1` (native support in this spreg
version, not a hand-rolled WX construction), and `spat_impacts='simple'`
for the direct/indirect/total average effects (LeSage & Pace 2009).

Run as a script:
    python -m src.econ.sdm
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl
import spreg
import spreg.panel_utils
import spreg.user_output
from libpysal.weights import W

from src.econ.weights import build_weight, load_graph

_ORIGINAL_CHECK_CONSTANT = spreg.user_output.check_constant


def _patched_check_constant(x, name_x=None, just_rem=False):
    """Work around a spreg bug: `panel_utils.prepare_panel` calls
    `check_constant(x, name_x, just_rem=True)` on the *wide* (n x T*k)
    panel matrix while `name_x` still only holds the k unexpanded variable
    names, so `keep_x[i]` indexes past the end of the list as soon as any
    single wide column (one variable, one specific hour) happens to be
    exactly constant across stops -- which is common for a sparse variable
    like precipitation on a mostly-dry day. The upstream fix would expand
    name_x before this call; here we just skip the (cosmetic) renaming
    step in that mismatched case and keep dropping the constant columns
    positionally, which is all `prepare_panel` actually needs downstream.
    """
    try:
        return _ORIGINAL_CHECK_CONSTANT(x, name_x, just_rem)
    except IndexError:
        # Dropping the constant wide column(s) would break prepare_panel's
        # "column count must be exactly k or k*T" invariant downstream, so
        # instead just leave x untouched: an all-constant column for one
        # hour of one variable contributes zero identifying variance for
        # that hour-slice but is not otherwise a problem for the ML fit.
        return np.asarray(x), name_x, None


spreg.user_output.check_constant = _patched_check_constant
spreg.panel_utils.USER.check_constant = _patched_check_constant

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_STOPS = 1500
N_HOURS = 24
# peak_share is deliberately excluded: it is a pure function of hour, not
# stop, so within any given hour it is identical across every stop in the
# panel (zero cross-sectional variance in that hour's column). With only
# stop fixed effects (no hour FE, see module docstring), that hits a
# spreg bug in this version's check_constant (it operates on the raw wide
# x matrix before variable names are expanded per hour, and indexes into
# the short per-variable name list with the wide column index, an
# off-by-a-lot IndexError). temperature_2m and precipitation are grid-cell
# based and genuinely vary across stops within the same hour, so they are
# unaffected and kept.
X_VARS = ["temperature_2m", "precipitation"]


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("sdm")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(sh)
    return logger


def load_stop_hour_panel(processed_dir: Path, logger: logging.Logger) -> pl.DataFrame:
    """Mean delay and covariates per (stop_id, hour), local Europe/Berlin hour."""
    features = pl.read_parquet(processed_dir / "features.parquet")
    panel = (
        features.with_columns(pl.col("obs_arr_ts").dt.convert_time_zone("Europe/Berlin").dt.hour().alias("hour"))
        .group_by("stop_id", "hour")
        .agg(
            pl.col("current_delay").mean().alias("mean_delay"),
            pl.col("is_peak").cast(pl.Float64).mean().alias("peak_share"),
            pl.col("temperature_2m").mean().alias("temperature_2m"),
            pl.col("precipitation").mean().alias("precipitation"),
            pl.len().alias("n_obs"),
        )
    )
    logger.info("stop-hour panel: %d (stop, hour) cells from %d stops", panel.height, panel["stop_id"].n_unique())
    return panel


def select_stops(panel: pl.DataFrame, nodes: pl.DataFrame, edges: pl.DataFrame, max_stops: int, logger: logging.Logger) -> list[str]:
    hours_covered = panel.group_by("stop_id").agg(pl.col("hour").n_unique().alias("n_hours"))
    connected = set(edges["src"].to_list()) | set(edges["dst"].to_list())
    hours_covered = hours_covered.filter(pl.col("stop_id").is_in(connected))
    top = hours_covered.sort("n_hours", descending=True).head(max_stops)
    logger.info(
        "selected %d stops (of %d graph-connected, %d total): min hours covered %d, max %d",
        top.height, len(connected), nodes.height, top["n_hours"].min(), top["n_hours"].max(),
    )
    return top["stop_id"].to_list()


def build_balanced_panel(panel: pl.DataFrame, stop_ids: list[str], logger: logging.Logger) -> pl.DataFrame:
    """Fill missing (stop, hour) cells with that stop's own daily mean, for a strictly balanced NxT grid."""
    sub = panel.filter(pl.col("stop_id").is_in(stop_ids))
    daily_mean = sub.group_by("stop_id").agg(
        pl.col("mean_delay").mean().alias("stop_mean_delay"),
        pl.col("peak_share").mean().alias("stop_mean_peak"),
        pl.col("temperature_2m").mean().alias("stop_mean_temp"),
        pl.col("precipitation").mean().alias("stop_mean_precip"),
    )
    grid = pl.DataFrame({"stop_id": stop_ids}).join(pl.DataFrame({"hour": list(range(N_HOURS))}), how="cross")
    balanced = (
        grid.join(sub, on=["stop_id", "hour"], how="left")
        .join(daily_mean, on="stop_id", how="left")
        .with_columns(
            pl.col("mean_delay").fill_null(pl.col("stop_mean_delay")),
            pl.col("peak_share").fill_null(pl.col("stop_mean_peak")),
            pl.col("temperature_2m").fill_null(pl.col("stop_mean_temp")),
            pl.col("precipitation").fill_null(pl.col("stop_mean_precip")),
        )
        .select("stop_id", "hour", "mean_delay", *X_VARS)
        .sort("stop_id", "hour")
    )
    filled_share = 1 - sub.height / balanced.height
    logger.info(
        "balanced panel: %d cells (%d stops x %d hours), %.1f%% filled from stop daily means",
        balanced.height, len(stop_ids), N_HOURS, 100 * filled_share,
    )
    return balanced


def to_wide(balanced: pl.DataFrame, stop_ids: list[str], value_col: str) -> np.ndarray:
    pivot = balanced.pivot(on="hour", index="stop_id", values=value_col)
    pivot = pivot.select(["stop_id"] + [str(h) for h in range(N_HOURS)])
    ordered = pl.DataFrame({"stop_id": stop_ids}).join(pivot, on="stop_id", how="left")
    arr = ordered.drop("stop_id").to_numpy().astype(float)
    arr[np.isnan(arr)] = 0.0
    return arr


def run_sdm_for_weight(balanced: pl.DataFrame, stop_ids: list[str], w: W, weight_name: str, logger: logging.Logger) -> dict:
    y = to_wide(balanced, stop_ids, "mean_delay")
    x = np.hstack([to_wide(balanced, stop_ids, v) for v in X_VARS])

    lm_lag = spreg.panel_LMlag(y, x, w)
    lm_error = spreg.panel_LMerror(y, x, w)
    logger.info(
        "[%s] LM-lag stat=%.3f p=%.4f | LM-error stat=%.3f p=%.4f",
        weight_name, lm_lag[0], lm_lag[1], lm_error[0], lm_error[1],
    )

    model = spreg.ML_LagFE(y, x, w, slx_lags=1, spat_impacts="simple", name_x=X_VARS)
    rho = float(model.rho)
    adi, aii, ati = model.sp_multipliers["simple"]

    betas = {name: float(model.betas[i, 0]) for i, name in enumerate(X_VARS)}
    thetas = {name: float(model.betas[len(X_VARS) + i, 0]) for i, name in enumerate(X_VARS)}
    impact_rows = []
    for name in X_VARS:
        direct = betas[name] * adi
        total = (betas[name] + thetas[name]) * ati
        impact_rows.append({"variable": name, "direct": direct, "indirect": total - direct, "total": total})

    logger.info("[%s] rho=%.4f ADI=%.4f AII=%.4f ATI=%.4f", weight_name, rho, adi, aii, ati)
    return {
        "weight": weight_name,
        "rho": rho,
        "adi": adi,
        "aii": aii,
        "ati": ati,
        "logll": float(model.logll),
        "aic": float(model.aic),
        "lm_lag_stat": float(lm_lag[0]),
        "lm_lag_p": float(lm_lag[1]),
        "lm_error_stat": float(lm_error[0]),
        "lm_error_p": float(lm_error[1]),
        "impacts": pl.DataFrame(impact_rows),
    }


def main() -> None:
    logger = setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-stops", type=int, default=MAX_STOPS)
    args = parser.parse_args()

    processed_dir = REPO_ROOT / "data" / "processed"
    nodes, edges = load_graph(processed_dir)
    panel = load_stop_hour_panel(processed_dir, logger)

    combined_edges = edges.select("src", "dst").unique()
    stop_ids = select_stops(panel, nodes, combined_edges, args.max_stops, logger)
    balanced = build_balanced_panel(panel, stop_ids, logger)

    results = []
    for weight_name, edge_filter in [("sched_adj", "sched_adj"), ("shared_segment", "shared_segment"), ("combined", None)]:
        sub_edges = edges.filter(pl.col("edge_type") == edge_filter) if edge_filter else combined_edges
        sub_edges = sub_edges.filter(pl.col("src").is_in(stop_ids) & pl.col("dst").is_in(stop_ids))
        w = build_weight(sub_edges, stop_ids)
        result = run_sdm_for_weight(balanced, stop_ids, w, weight_name, logger)
        results.append(result)

    out_dir = REPO_ROOT / "data" / "processed"
    summary_rows = [{k: v for k, v in r.items() if k != "impacts"} for r in results]
    pl.DataFrame(summary_rows).write_parquet(out_dir / "sdm_model_summary.parquet")

    impact_rows = [r["impacts"].with_columns(pl.lit(r["weight"]).alias("weight")) for r in results]
    pl.concat(impact_rows).write_parquet(out_dir / "sdm_impacts.parquet")

    for r in results:
        logger.info("=== %s ===", r["weight"])
        logger.info("rho=%.4f logll=%.2f aic=%.2f", r["rho"], r["logll"], r["aic"])
        print(r["impacts"])

    logger.info("wrote sdm_model_summary.parquet and sdm_impacts.parquet")


if __name__ == "__main__":
    main()
