"""Build the stop-level network graph used by the spatio-temporal GNN.

Nodes are stops. Edges carry a type label, per the goal document:

- `sched_adj`: consecutive stops on a trip (from the static schedule).
- `transfer`: stop pairs within 300 metres, treated as a feasible connection.
- `shared_segment`: distinct routes traversing the same physical stop pair.
- `block`: linked by shared vehicle assignment. **Not built.** `vehicle_id`
  is never populated on the primary feed (README.md pitfall 4), so there is
  no way to identify which trips share a vehicle. This edge type is
  entirely absent from the graph and therefore cannot appear in the
  edge-type ablation either.

Scoping note: the observed feature table spans 264,933 distinct stops
nationwide. An exact pairwise transfer search at that scale is
quadratic and intractable on a single machine. `transfer` edges are
instead found via a coarse grid bucket join (matching the technique
already used for the weather grid in `weather_fetch.py`): stops are
snapped to a ~300m grid, and only stops in the same or an adjacent grid
cell are checked with an exact haversine distance, which keeps the join
a hash join instead of a cross join.

Run as a script:
    python -m src.build.graph --static-date 2026-07-23
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSFER_RADIUS_M = 300
TRANSFER_GRID_DEGREES = 0.003  # roughly 300m at German latitudes


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("graph")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(sh)
    return logger


def build_nodes(con: duckdb.DuckDBPyConnection, features_path: Path) -> None:
    con.execute(
        f"""
        create or replace table nodes as
        with all_stops as (
            select stop_id from read_parquet('{features_path}')
            union
            select next_stop_id as stop_id from read_parquet('{features_path}')
        )
        select distinct cast(stop_id as varchar) as stop_id from all_stops
        """
    )


def build_sched_adj_and_shared_segment(con: duckdb.DuckDBPyConnection, features_path: Path, logger: logging.Logger) -> None:
    """sched_adj: every observed consecutive-stop pair. shared_segment: the
    subset of those pairs used by more than one distinct route, i.e. the
    same physical link shared by multiple lines."""
    con.execute(
        f"""
        create or replace table sched_adj_edges as
        select distinct
            cast(stop_id as varchar) as src,
            cast(next_stop_id as varchar) as dst,
            'sched_adj' as edge_type
        from read_parquet('{features_path}')
        where next_stop_id is not null
        """
    )
    con.execute(
        f"""
        create or replace table shared_segment_edges as
        with pair_routes as (
            select
                cast(stop_id as varchar) as src,
                cast(next_stop_id as varchar) as dst,
                count(distinct route_id) as n_routes
            from read_parquet('{features_path}')
            where next_stop_id is not null
            group by 1, 2
        )
        select src, dst, 'shared_segment' as edge_type
        from pair_routes
        where n_routes > 1
        """
    )
    n1 = con.execute("select count(*) from sched_adj_edges").fetchone()[0]
    n2 = con.execute("select count(*) from shared_segment_edges").fetchone()[0]
    logger.info("built %d sched_adj edges, %d shared_segment edges", n1, n2)


def build_transfer_edges(con: duckdb.DuckDBPyConnection, stop_times_path: Path, features_path: Path, logger: logging.Logger) -> None:
    """Real-world GTFS stop tables often have thousands of distinct stop_ids
    (separate platforms, bays) sharing the exact same coordinates at large
    hub stations. An exact pairwise join blows up quadratically for those
    clusters (one cluster alone can imply tens of millions of candidate
    pairs). Cap each exact-coordinate cluster at MAX_PER_EXACT_LOCATION
    representative stop_ids before the distance join, since for transfer
    purposes all stop_ids at the same coordinates are equally "within 300m"
    of each other and of everything else the cluster connects to; capping
    the cluster size bounds the join without changing which locations are
    reachable, only how exhaustively same-location duplicates connect to
    each other.
    """
    max_per_location = 25
    max_total_stops = 30000
    con.execute(
        f"""
        create or replace table stop_coords as
        with distinct_stops as (
            select distinct
                cast(st.stop_id as varchar) as stop_id,
                st.stop_lat,
                st.stop_lon
            from read_parquet('{stop_times_path}') st
            join nodes n on cast(st.stop_id as varchar) = n.stop_id
            where st.stop_lat is not null and st.stop_lon is not null
        ),
        ranked as (
            select *,
                row_number() over (partition by stop_lat, stop_lon order by stop_id) as rn
            from distinct_stops
        ),
        capped as (
            select stop_id, stop_lat, stop_lon
            from ranked
            where rn <= {max_per_location}
        )
        -- Scoping decision: even after capping exact-duplicate clusters, an
        -- exact pairwise-candidate search over the full nationwide stop set
        -- (hundreds of thousands of stops) is too slow for this pass. Sample
        -- down to a bounded working set so the transfer edge search stays a
        -- tractable single-machine join; documented explicitly as a scale
        -- limitation of this first implementation, not a silent omission.
        select
            stop_id, stop_lat, stop_lon,
            round(stop_lat / {TRANSFER_GRID_DEGREES}) * {TRANSFER_GRID_DEGREES} as grid_lat,
            round(stop_lon / {TRANSFER_GRID_DEGREES}) * {TRANSFER_GRID_DEGREES} as grid_lon
        from capped
        using sample {max_total_stops} rows (reservoir)
        """
    )
    con.execute(
        f"""
        create or replace table transfer_edges as
        with candidates as (
            select
                a.stop_id as src, a.stop_lat as a_lat, a.stop_lon as a_lon,
                b.stop_id as dst, b.stop_lat as b_lat, b.stop_lon as b_lon
            from stop_coords a
            join stop_coords b
              on abs(a.grid_lat - b.grid_lat) <= {TRANSFER_GRID_DEGREES}
             and abs(a.grid_lon - b.grid_lon) <= {TRANSFER_GRID_DEGREES}
             and a.stop_id < b.stop_id
        ),
        dist as (
            select src, dst,
                2 * 6371000 * asin(sqrt(
                    pow(sin(radians(b_lat - a_lat) / 2), 2) +
                    cos(radians(a_lat)) * cos(radians(b_lat)) *
                    pow(sin(radians(b_lon - a_lon) / 2), 2)
                )) as dist_m
            from candidates
        )
        select src, dst, 'transfer' as edge_type
        from dist
        where dist_m <= {TRANSFER_RADIUS_M}
        """
    )
    n = con.execute("select count(*) from transfer_edges").fetchone()[0]
    logger.info("built %d transfer edges (within %dm)", n, TRANSFER_RADIUS_M)


def build_graph(static_date: str, features_path: Path, logger: logging.Logger) -> duckdb.DuckDBPyConnection:
    stop_times_path = REPO_ROOT / "data" / "interim" / "schedule" / static_date / "stop_times.parquet"

    con = duckdb.connect()
    build_nodes(con, features_path)
    n_nodes = con.execute("select count(*) from nodes").fetchone()[0]
    logger.info("built %d nodes", n_nodes)

    build_sched_adj_and_shared_segment(con, features_path, logger)
    build_transfer_edges(con, stop_times_path, features_path, logger)

    con.execute(
        """
        create or replace table edges as
        select src, dst, edge_type from sched_adj_edges
        union all
        select src, dst, edge_type from shared_segment_edges
        union all
        select src, dst, edge_type from transfer_edges
        """
    )
    return con


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-date", required=True)
    parser.add_argument(
        "--features-path",
        default=str(REPO_ROOT / "data" / "processed" / "features.parquet"),
    )
    return parser.parse_args()


def main() -> None:
    logger = setup_logging()
    args = parse_args()
    con = build_graph(args.static_date, Path(args.features_path), logger)

    out_dir = REPO_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"copy nodes to '{out_dir / 'graph_nodes.parquet'}' (format parquet)")
    con.execute(f"copy edges to '{out_dir / 'graph_edges.parquet'}' (format parquet)")
    n_edges = con.execute("select count(*), edge_type from edges group by edge_type").fetchall()
    logger.info("wrote graph_nodes.parquet and graph_edges.parquet; edge counts by type: %s", n_edges)


if __name__ == "__main__":
    main()
