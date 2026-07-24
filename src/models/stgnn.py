"""Tier 3: a minimal spatio-temporal GNN, with the edge-type ablation as the headline experiment.

Implemented with plain PyTorch tensor operations (manual one-hop mean
aggregation via `index_add_`), not `torch_geometric`: installing
`torch-geometric` and its scatter/sparse companion wheels reliably in this
environment added a large, fragile dependency chain for what a handful of
`index_add_` calls does directly, so we build the tiny amount of message
passing needed by hand instead. This is a real engineering trade-off,
documented rather than hidden.

Graph: nodes are stops (`src/build/graph.py`), edges carry a type label:
`sched_adj`, `transfer`, `shared_segment`. `block` is entirely absent
(vehicle_id is never populated, README.md pitfall 4) and so cannot appear
in the ablation either; that omission is itself a finding, not a gap in
this module.

Node signal: given only a single day of realtime history, per-stop node
signal is a single scalar, the training-split mean `y_delay_increment`
observed at that stop (not per hour bucket: at this data volume, per-hour
per-stop aggregates would be too sparse to be a meaningful signal).
Revisit once enough days accrue to support a finer-grained, still
leakage-safe (train-only) node signal.

The headline experiment: retrain the small feedforward head with each
edge-type's neighbour-aggregate feature zeroed out in turn (and with all
three zeroed, "no graph" at all) and compare validation MAE. The drop in
performance from removing a channel is the estimate of how much that
propagation channel actually carries delay signal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

EDGE_TYPES = ["sched_adj", "transfer", "shared_segment"]
CORE_FEATURES = ["current_delay", "delay_prev1", "stops_remaining", "elapsed_share", "is_peak"]


class DelayFFN(nn.Module):
    def __init__(self, n_inputs: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_graph(processed_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    nodes = pl.read_parquet(processed_dir / "graph_nodes.parquet")
    edges = pl.read_parquet(processed_dir / "graph_edges.parquet")
    return nodes, edges


def build_node_index(nodes: pl.DataFrame) -> dict[str, int]:
    return {stop_id: i for i, stop_id in enumerate(nodes["stop_id"].to_list())}


def compute_node_signal(train: pl.DataFrame, node_index: dict[str, int]) -> np.ndarray:
    """Training-split-only per-stop mean delay increment, the leakage-safe node signal."""
    n = len(node_index)
    signal = np.zeros(n, dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)

    agg = train.group_by("stop_id").agg(pl.col("y_delay_increment").mean().alias("m"), pl.len().alias("n"))
    for row in agg.iter_rows(named=True):
        idx = node_index.get(str(row["stop_id"]))
        if idx is not None:
            signal[idx] = row["m"]
            counts[idx] = row["n"]
    return signal


def build_edge_index_by_type(edges: pl.DataFrame, node_index: dict[str, int]) -> dict[str, np.ndarray]:
    result = {}
    for edge_type in EDGE_TYPES:
        sub = edges.filter(pl.col("edge_type") == edge_type)
        src = np.array([node_index[s] for s in sub["src"].to_list() if s in node_index])
        dst = np.array([node_index[d] for d in sub["dst"].to_list() if d in node_index])
        result[edge_type] = np.stack([src, dst]) if len(src) else np.zeros((2, 0), dtype=np.int64)
    return result


def propagate_one_hop(node_signal: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """Mean of neighbour signal over each edge type's incoming edges (undirected: both directions)."""
    n = len(node_signal)
    sums = np.zeros(n, dtype=np.float32)
    counts = np.zeros(n, dtype=np.float32)
    if edge_index.shape[1] == 0:
        return sums
    src, dst = edge_index[0], edge_index[1]
    np.add.at(sums, dst, node_signal[src])
    np.add.at(counts, dst, 1.0)
    np.add.at(sums, src, node_signal[dst])
    np.add.at(counts, src, 1.0)
    counts[counts == 0] = 1.0
    return sums / counts


def build_row_features(
    df: pl.DataFrame,
    node_index: dict[str, int],
    node_signal: np.ndarray,
    neighbor_signal_by_type: dict[str, np.ndarray],
    zero_out: set[str] | None = None,
) -> np.ndarray:
    zero_out = zero_out or set()
    core = df.select(CORE_FEATURES).fill_null(0.0).to_numpy().astype(np.float32)

    stop_indices = np.array([node_index.get(str(s), -1) for s in df["stop_id"].to_list()])
    valid = stop_indices >= 0
    safe_idx = np.where(valid, stop_indices, 0)

    own_signal = np.where(valid, node_signal[safe_idx], 0.0).reshape(-1, 1).astype(np.float32)
    graph_cols = [own_signal]
    for edge_type in EDGE_TYPES:
        col = np.where(valid, neighbor_signal_by_type[edge_type][safe_idx], 0.0).astype(np.float32)
        if edge_type in zero_out:
            col = np.zeros_like(col)
        graph_cols.append(col.reshape(-1, 1))

    return np.concatenate([core] + graph_cols, axis=1)


def train_stgnn(
    train: pl.DataFrame,
    val: pl.DataFrame,
    processed_dir: Path,
    epochs: int = 15,
    zero_out: set[str] | None = None,
    seed: int = 42,
) -> tuple[DelayFFN, dict, np.ndarray, dict[str, np.ndarray]]:
    torch.manual_seed(seed)
    nodes, edges = load_graph(processed_dir)
    node_index = build_node_index(nodes)
    node_signal = compute_node_signal(train, node_index)
    edge_index_by_type = build_edge_index_by_type(edges, node_index)
    neighbor_signal_by_type = {et: propagate_one_hop(node_signal, edge_index_by_type[et]) for et in EDGE_TYPES}

    x_train = build_row_features(train, node_index, node_signal, neighbor_signal_by_type, zero_out)
    y_train = train["y_delay_increment"].to_numpy().astype(np.float32)

    means, stds = x_train.mean(axis=0), x_train.std(axis=0)
    stds[stds == 0] = 1.0
    x_train = (x_train - means) / stds

    model = DelayFFN(n_inputs=x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_train_t, y_train_t = torch.from_numpy(x_train), torch.from_numpy(y_train)

    batch_size = 4096
    n = x_train_t.shape[0]
    for _epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            pred = model(x_train_t[idx])
            loss = torch.abs(pred - y_train_t[idx]).mean()
            loss.backward()
            optimizer.step()

    context = {"means": means, "stds": stds, "node_index": node_index}
    return model, context, node_signal, neighbor_signal_by_type


def predict_stgnn(
    model: DelayFFN,
    df: pl.DataFrame,
    context: dict,
    node_signal: np.ndarray,
    neighbor_signal_by_type: dict[str, np.ndarray],
    zero_out: set[str] | None = None,
) -> np.ndarray:
    x = build_row_features(df, context["node_index"], node_signal, neighbor_signal_by_type, zero_out)
    x = (x - context["means"]) / context["stds"]
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy()


def run_edge_type_ablation(
    train: pl.DataFrame, val: pl.DataFrame, processed_dir: Path, mae_fn
) -> pl.DataFrame:
    """Retrain with each edge type (and all of them) zeroed out; report the val MAE change."""
    rows = []
    configs = [("full_graph", set())] + [(f"without_{et}", {et}) for et in EDGE_TYPES] + [("no_graph", set(EDGE_TYPES))]
    for name, zero_out in configs:
        model, context, node_signal, neighbor_signal = train_stgnn(train, val, processed_dir, zero_out=zero_out)
        pred = predict_stgnn(model, val, context, node_signal, neighbor_signal, zero_out=zero_out)
        val_mae = mae_fn(val["y_delay_increment"].to_numpy(), pred)
        rows.append({"config": name, "val_mae": val_mae})
    return pl.DataFrame(rows)
