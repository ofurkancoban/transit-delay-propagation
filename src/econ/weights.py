"""Build row-standardised spatial weight matrices from the stop-level graph.

Reuses the graph already built for the STGNN ablation
(`src/build/graph.py`, `data/processed/graph_nodes.parquet` and
`graph_edges.parquet`) rather than re-deriving stop adjacency: the edge
types are identical to the ones the goal document asks for here
(`sched_adj`, `transfer`, `shared_segment`), and `block` is absent for the
same reason noted throughout this repo: `vehicle_id` is never populated
on the primary feed (README.md pitfall 4), so no vehicle-chain edges
exist to build a weight matrix from.

Each edge type gets its own row-standardised `libpysal.weights.W`, plus a
combined matrix (union of all edge types, row-standardised after
combining, not a sum of the three row-standardised matrices, since that
would not itself row-sum to one).

Run as a script:
    python -m src.econ.weights
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import polars as pl
from libpysal.weights import W

REPO_ROOT = Path(__file__).resolve().parents[2]
EDGE_TYPES = ["sched_adj", "transfer", "shared_segment"]


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("weights")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(sh)
    return logger


def load_graph(processed_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    nodes = pl.read_parquet(processed_dir / "graph_nodes.parquet")
    edges = pl.read_parquet(processed_dir / "graph_edges.parquet")
    return nodes, edges


def edges_to_neighbors(edges: pl.DataFrame, all_stop_ids: list[str]) -> dict[str, list[str]]:
    """Undirected adjacency dict, every node present (isolates get an empty neighbor list).

    Self-loops (src == dst) are dropped: 89 exist in graph_edges.parquet,
    almost certainly from stop_times rows where a trip's stop_sequence
    repeats the same stop_id consecutively. A self-loop would put a
    nonzero on the weight matrix diagonal, which every downstream spatial
    regression in this module rejects outright (spreg.check_weights
    requires an exactly-zero diagonal), so this is a real data-quality
    fix, not a numerical workaround.
    """
    neighbors: dict[str, set[str]] = {s: set() for s in all_stop_ids}
    for src, dst in edges.select("src", "dst").iter_rows():
        if src == dst:
            continue
        neighbors[src].add(dst)
        neighbors[dst].add(src)
    return {k: sorted(v) for k, v in neighbors.items()}


def build_weight(edges: pl.DataFrame, all_stop_ids: list[str]) -> W:
    """Row-standardised W. Islands (no neighbors) are kept with zero weight rows,
    consistent with libpysal's own handling, and reported so they are not silently
    dropped from downstream regressions."""
    neighbors = edges_to_neighbors(edges, all_stop_ids)
    w = W(neighbors)
    w.transform = "r"
    return w


def build_all_weights(processed_dir: Path, logger: logging.Logger) -> dict[str, W]:
    nodes, edges = load_graph(processed_dir)
    all_stop_ids = nodes["stop_id"].to_list()

    weights = {}
    for edge_type in EDGE_TYPES:
        sub = edges.filter(pl.col("edge_type") == edge_type)
        w = build_weight(sub, all_stop_ids)
        n_islands = len(w.islands)
        logger.info(
            "%s: %d edges, %d nodes, %d islands (%.1f%%)",
            edge_type, sub.height, w.n, n_islands, 100 * n_islands / w.n,
        )
        weights[edge_type] = w

    combined = edges.select("src", "dst").unique()
    w_combined = build_weight(combined, all_stop_ids)
    logger.info(
        "combined: %d unique edges, %d nodes, %d islands (%.1f%%)",
        combined.height, w_combined.n, len(w_combined.islands), 100 * len(w_combined.islands) / w_combined.n,
    )
    weights["combined"] = w_combined
    return weights


def main() -> None:
    logger = setup_logging()
    processed_dir = REPO_ROOT / "data" / "processed"
    weights = build_all_weights(processed_dir, logger)
    logger.info("built %d weight matrices: %s", len(weights), list(weights.keys()))


if __name__ == "__main__":
    main()
