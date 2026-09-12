"""LSTM baseline, added alongside the three tree-based models Lucas et al.
(2020) actually compare, for a wider view since LSTM is a common EPF
benchmark elsewhere in the literature (e.g. Deng et al. 2023 for GB
imbalance prices). Not part of the paper's own comparison, so no paper
number to reproduce here, this is reported next to the tree models as an
additional data point, not a reproduction target.

Sequence model: a lookback window of SEQ_LEN settlement periods over the
same feature columns used for the tree models, predicting balancing_price
at the next period. Same chronological 80/20 split, same feature set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

TARGET = "balancing_price"
DROP_COLS = {"settlementDate", "settlementPeriod", TARGET, "systemBuyPrice", "systemSellPrice"}

SEQ_LEN = 48  # 24 hours of settlement periods
EPOCHS = 20
BATCH_SIZE = 128
HIDDEN_SIZE = 64


class PriceLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(X) - seq_len
    Xs = np.stack([X[i:i + seq_len] for i in range(n)])
    ys = y[seq_len:]
    return Xs.astype(np.float32), ys.astype(np.float32)


def run(features_path: Path, suffix: str) -> None:
    df = pd.read_parquet(features_path)
    df = df.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    split_idx = int(len(df) * 0.8)
    X_raw = df[feature_cols].to_numpy()
    y_raw = df[TARGET].to_numpy()

    scaler_X = StandardScaler().fit(X_raw[:split_idx])
    X_scaled = scaler_X.transform(X_raw)

    X_train_seq, y_train_seq = make_sequences(X_scaled[:split_idx], y_raw[:split_idx], SEQ_LEN)
    X_test_seq, y_test_seq = make_sequences(X_scaled[split_idx - SEQ_LEN:], y_raw[split_idx - SEQ_LEN:], SEQ_LEN)

    print(f"Train sequences: {len(X_train_seq)}, test sequences: {len(X_test_seq)}, "
          f"seq_len={SEQ_LEN}, n_features={len(feature_cols)}")

    device = torch.device("cpu")
    model = PriceLSTM(n_features=len(feature_cols)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_train_t = torch.from_numpy(X_train_seq)
    y_train_t = torch.from_numpy(y_train_seq)
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        avg_loss = total_loss / len(dataset)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:2d}/{EPOCHS}  train MSE {avg_loss:.2f}")

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test_seq)).numpy()

    mae = mean_absolute_error(y_test_seq, preds)
    mse = mean_squared_error(y_test_seq, preds)
    r2 = r2_score(y_test_seq, preds) * 100

    print(f"\nLSTM")
    print(f"  MAE: {mae:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  R2:  {r2:.1f}%")

    result = {"model": "LSTM", "MAE": round(mae, 2), "MSE": round(mse, 2), "R2_pct": round(r2, 1)}

    comparison_path = RESULTS / f"model_comparison{suffix}.csv"
    if comparison_path.exists():
        comp = pd.read_csv(comparison_path)
        comp = comp[comp["model"] != "LSTM"]
        comp = pd.concat([comp, pd.DataFrame([result])], ignore_index=True)
    else:
        comp = pd.DataFrame([result])
    comp.to_csv(comparison_path, index=False)
    print(f"\nAppended LSTM row to {comparison_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(PROC / "features.parquet"))
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    torch.manual_seed(42)
    run(Path(args.features), args.suffix)
