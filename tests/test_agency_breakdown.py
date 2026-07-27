import polars as pl

from src.models.agency_breakdown import MIN_ROWS_PER_AGENCY, build_delay_summary


def _fake_route_agency() -> pl.DataFrame:
    return pl.DataFrame({
        "route_id": ["r1", "r2", "r3"],
        "agency_name": ["Big Agency", "Big Agency", "Small Agency"],
    })


def test_build_delay_summary_aggregates_and_filters_by_min_rows(tmp_path):
    n_big = MIN_ROWS_PER_AGENCY + 10
    n_small = MIN_ROWS_PER_AGENCY - 10

    realised = pl.concat([
        pl.DataFrame({
            "route_id": ["r1"] * n_big,
            "trip_id": [f"t{i}" for i in range(n_big)],
            "realised_arr_delay": [10.0] * n_big,
        }),
        pl.DataFrame({
            "route_id": ["r3"] * n_small,
            "trip_id": [f"t{i}" for i in range(n_small)],
            "realised_arr_delay": [500.0] * n_small,
        }),
    ])
    realised_path = tmp_path / "panel_realised.parquet"
    realised.write_parquet(realised_path)

    summary = build_delay_summary(realised_path, _fake_route_agency(), _fake_logger())

    assert summary["agency_name"].to_list() == ["Big Agency"]
    row = summary.row(0, named=True)
    assert row["n_stop_events"] == n_big
    assert row["mean_delay_s"] == 10.0
    assert row["share_over_6min"] == 0.0


def _fake_logger():
    import logging
    logger = logging.getLogger("test_agency_breakdown")
    logger.addHandler(logging.NullHandler())
    return logger
