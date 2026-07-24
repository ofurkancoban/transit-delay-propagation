"""Tier 2: a small GRU over each trip's stop sequence.

Captures the longitudinal channel directly: the recurrent state carries
information about how delay has evolved over the whole trip so far, rather
than the last 3 stops used as explicit lag features for LightGBM.

Given only a single day of realtime history, this is a modest model
(one GRU layer, hidden size 32) trained on a fraction of the available
trips for tractability on CPU; the point is a correct, honestly reported
first pass, not a tuned production model. Padding uses a mask so the loss
ignores padded steps, and trips are grouped by (service_date, trip_id) and
ordered by stop_sequence, consistent with the feature table's own ordering.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
from torch import nn

from src.models.lgbm import FEATURE_COLUMNS, TARGET_COLUMN

MAX_TRIP_LEN = 40


class DelayGRU(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out).squeeze(-1)


def build_sequences(df: pl.DataFrame, feature_means: np.ndarray, feature_stds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group rows into per-trip sequences, standardise, and pad/truncate to MAX_TRIP_LEN."""
    grouped = df.sort(["service_date", "trip_id", "stop_sequence"]).group_by(["service_date", "trip_id"], maintain_order=True)

    x_list, y_list, mask_list = [], [], []
    for _, group in grouped:
        feats = group.select(FEATURE_COLUMNS).fill_null(0.0).to_numpy().astype(np.float32)
        feats = (feats - feature_means) / feature_stds
        target = group[TARGET_COLUMN].to_numpy().astype(np.float32)

        n = min(len(feats), MAX_TRIP_LEN)
        x_pad = np.zeros((MAX_TRIP_LEN, feats.shape[1]), dtype=np.float32)
        y_pad = np.zeros(MAX_TRIP_LEN, dtype=np.float32)
        mask = np.zeros(MAX_TRIP_LEN, dtype=np.float32)
        x_pad[:n] = feats[:n]
        y_pad[:n] = target[:n]
        mask[:n] = 1.0

        x_list.append(x_pad)
        y_list.append(y_pad)
        mask_list.append(mask)

    return np.stack(x_list), np.stack(y_list), np.stack(mask_list)


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(pred - target) * mask
    return diff.sum() / mask.sum().clamp_min(1.0)


def _sample_trips(df: pl.DataFrame, max_trips: int, seed: int) -> pl.DataFrame:
    trip_ids = df.select("service_date", "trip_id").unique()
    if trip_ids.height > max_trips:
        trip_ids = trip_ids.sample(n=max_trips, seed=seed)
        df = df.join(trip_ids, on=["service_date", "trip_id"], how="inner")
    return df


def train_gru(
    train: pl.DataFrame,
    val: pl.DataFrame,
    max_trips: int = 20000,
    max_val_trips: int = 4000,
    epochs: int = 8,
    seed: int = 42,
) -> tuple[DelayGRU, np.ndarray, np.ndarray]:
    """max_val_trips is capped much lower than max_trips: building sequences
    iterates per-trip in Python, and the validation split only needs to be
    large enough to track training progress each epoch, not the full split."""
    torch.manual_seed(seed)

    train = _sample_trips(train, max_trips, seed)
    val = _sample_trips(val, max_val_trips, seed)

    numeric = train.select(FEATURE_COLUMNS).fill_null(0.0).to_numpy().astype(np.float32)
    means = numeric.mean(axis=0)
    stds = numeric.std(axis=0)
    stds[stds == 0] = 1.0

    x_train, y_train, m_train = build_sequences(train, means, stds)
    x_val, y_val, m_val = build_sequences(val, means, stds)

    model = DelayGRU(n_features=len(FEATURE_COLUMNS))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    m_train_t = torch.from_numpy(m_train)

    batch_size = 256
    n = x_train_t.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            pred = model(x_train_t[idx])
            loss = masked_l1_loss(pred, y_train_t[idx], m_train_t[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        with torch.no_grad():
            val_pred = model(torch.from_numpy(x_val))
            val_loss = masked_l1_loss(val_pred, torch.from_numpy(y_val), torch.from_numpy(m_val))
        print(f"epoch {epoch}: train_loss={total_loss / n:.3f} val_loss={val_loss.item():.3f}")

    return model, means, stds


def predict_gru(model: DelayGRU, df: pl.DataFrame, means: np.ndarray, stds: np.ndarray) -> pl.DataFrame:
    """Predict for every row, returning a (service_date, trip_id, stop_sequence, y_pred_gru) table."""
    x, _, mask = build_sequences(df, means, stds)
    with torch.no_grad():
        pred = model(torch.from_numpy(x)).numpy()

    keys = df.sort(["service_date", "trip_id", "stop_sequence"]).select("service_date", "trip_id", "stop_sequence").group_by(
        ["service_date", "trip_id"], maintain_order=True
    ).agg(pl.col("stop_sequence"))

    rows = []
    for i, key_row in enumerate(keys.iter_rows(named=True)):
        seqs = key_row["stop_sequence"]
        n = min(len(seqs), MAX_TRIP_LEN)
        for j in range(n):
            rows.append({
                "service_date": key_row["service_date"],
                "trip_id": key_row["trip_id"],
                "stop_sequence": seqs[j],
                "y_pred_gru": float(pred[i, j]),
            })
    return pl.DataFrame(rows)
