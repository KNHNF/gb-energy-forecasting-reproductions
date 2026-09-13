"""Seasonal-attention BiLSTM reproduction of Deng et al. (2023), rebuilt
2026-09-12 against the confirmed Algorithm 1 from the source PDF (read
directly from Zotero storage, not an earlier degraded extraction).

The confirmed architecture is materially different from this reproduction's
first attempt, and is implemented faithfully here:

- Univariate only. Input x = [x1, ..., x48] is 48 lags of the price series
  itself. No demand, wind, generation-mix, or calendar features are fed to
  the network anywhere in the paper's own algorithm.
- The "seasonal attention" is not attention over BiLSTM hidden states. It is
  an elementwise gate applied to the raw input lags before the BiLSTM sees
  them: f_A is a learnable vector of 48 weights, initialised as [1, ..., 48],
  and the gated input is x_hat = x * f_A (Equations 11-13, Algorithm 1).
  f_A is trained jointly with the BiLSTM's own weights via Adam, same loss.
- Training: batch_size=2048, epochs=20, lr=0.00001, Adam, MSE loss (stated
  explicitly in Algorithm 1's caption, not estimated).
- Evaluation: the paper does not use a chronological 80/20 split. It reserves
  the final 2000 points of the ~52,560-point sample as the test set (Section
  5.1) and trains on everything before that.
- One-step-ahead only is implemented here (k=1). The paper reports k=1..4;
  reproducing all four horizons is a mechanical extension of the same
  training loop, left for a follow-up run rather than done in this pass.

This is still a reproduction attempt, not a replication: the exact random
seed, weight initialisation, and any gradient-clipping or early-stopping the
paper may have used beyond what Algorithm 1 states are not given in the
source text.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 48          # Algorithm 1: x = [x1, ..., x48]
N_TEST = 2000         # Section 5.1: final 2,000 points held out
BATCH_SIZE = 2048     # Algorithm 1 caption
EPOCHS = 20           # Algorithm 1 caption
LR = 0.00001          # Algorithm 1 caption
HIDDEN_SIZE = 64      # not stated in the source text, kept from this repo's earlier pass
EXTREME_QUANTILE = 0.05  # bottom/top 5%, matching the paper's "extreme events" framing


class BiLSTMBaseline(nn.Module):
    """Plain BiLSTM, no seasonal gate. f_A fixed at 1 for every lag."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class SeasonalAttentionBiLSTM(nn.Module):
    """Algorithm 1: x_hat = x * f_A, f_A a learnable per-lag-position weight
    vector initialised as [1, ..., 48], trained jointly with the BiLSTM."""

    def __init__(self, seq_len: int = SEQ_LEN, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.f_A = nn.Parameter(torch.arange(1, seq_len + 1, dtype=torch.float32))
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        gated = x * self.f_A.view(1, -1, 1)          # Equation 13: Attention(x, f_A) = x * f_A
        out, _ = self.lstm(gated)
        return self.head(out[:, -1, :]).squeeze(-1)


def make_sequences(prices: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(prices) - seq_len
    Xs = np.stack([prices[i:i + seq_len] for i in range(n)])
    ys = prices[seq_len:]
    return Xs.astype(np.float32), ys.astype(np.float32)


def train_model(model, X_train_t, y_train_t, epochs: int = EPOCHS) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        if epoch % 5 == 0 or epoch == 1:
            print(f"    epoch {epoch:2d}/{epochs}  train MSE {total_loss / len(dataset):.4f}")


def evaluate(model, X_test_t, y_test_norm: np.ndarray, price_min: float, price_max: float) -> tuple[np.ndarray, dict]:
    model.eval()
    with torch.no_grad():
        preds_norm = model(X_test_t).numpy()
    preds = preds_norm * (price_max - price_min) + price_min
    y_test = y_test_norm * (price_max - price_min) + price_min
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds) * 100
    return preds, {"MAE": round(float(mae), 2), "RMSE": round(float(rmse), 2), "R2_pct": round(float(r2), 1)}


def extreme_event_mae(y_test: np.ndarray, preds: np.ndarray, q: float = EXTREME_QUANTILE) -> dict:
    lo, hi = np.quantile(y_test, [q, 1 - q])
    mask = (y_test <= lo) | (y_test >= hi)
    mae_extreme = mean_absolute_error(y_test[mask], preds[mask])
    mae_normal = mean_absolute_error(y_test[~mask], preds[~mask])
    return {
        "n_extreme": int(mask.sum()),
        "MAE_extreme": round(float(mae_extreme), 2),
        "MAE_normal": round(float(mae_normal), 2),
    }


def run() -> None:
    df = pd.read_parquet(PROC / "features.parquet")
    df = df.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)
    prices = df["imbalance_price"].to_numpy()
    print(f"Loaded {len(prices)} price points, {df['settlementDate'].min()} to {df['settlementDate'].max()}")

    # Algorithm 1, line 1: min-max normalisation, fit on train only.
    split_idx = len(prices) - N_TEST - SEQ_LEN
    price_min, price_max = prices[:split_idx].min(), prices[:split_idx].max()
    prices_norm = (prices - price_min) / (price_max - price_min)

    X_all, y_all = make_sequences(prices_norm, SEQ_LEN)
    X_train, y_train = X_all[:split_idx], y_all[:split_idx]
    X_test, y_test = X_all[split_idx:split_idx + N_TEST], y_all[split_idx:split_idx + N_TEST]
    print(f"Train sequences: {len(X_train)}, test sequences: {len(X_test)} (final {N_TEST} points, per Section 5.1)")

    X_train_t = torch.from_numpy(X_train).unsqueeze(-1)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test).unsqueeze(-1)

    results = {}
    preds_by_model = {}

    print("\nTraining plain BiLSTM baseline...")
    baseline = BiLSTMBaseline()
    train_model(baseline, X_train_t, y_train_t)
    preds_base, metrics_base = evaluate(baseline, X_test_t, y_test, price_min, price_max)
    print(f"  BiLSTM baseline: {metrics_base}")
    results["BiLSTM_baseline"] = metrics_base
    preds_by_model["BiLSTM_baseline"] = preds_base

    print("\nTraining seasonal-attention BiLSTM (x_hat = x * f_A)...")
    attn_model = SeasonalAttentionBiLSTM()
    train_model(attn_model, X_train_t, y_train_t)
    preds_attn, metrics_attn = evaluate(attn_model, X_test_t, y_test, price_min, price_max)
    print(f"  Seasonal-Attention BiLSTM: {metrics_attn}")
    results["SeasonalAttentionBiLSTM"] = metrics_attn
    preds_by_model["SeasonalAttentionBiLSTM"] = preds_attn

    y_test_gbp = y_test * (price_max - price_min) + price_min
    extreme_base = extreme_event_mae(y_test_gbp, preds_base)
    extreme_attn = extreme_event_mae(y_test_gbp, preds_attn)
    pct_improvement = 100 * (extreme_base["MAE_extreme"] - extreme_attn["MAE_extreme"]) / extreme_base["MAE_extreme"]

    print(f"\nExtreme-event MAE (bottom/top {int(EXTREME_QUANTILE*100)}%, n={extreme_base['n_extreme']}):")
    print(f"  BiLSTM baseline:           {extreme_base['MAE_extreme']}")
    print(f"  Seasonal-Attention BiLSTM: {extreme_attn['MAE_extreme']}")
    print(f"  Improvement: {pct_improvement:.1f}%")
    print(f"  (paper reports RMSE 13.035 for SA-BiLSTM, 1-step-ahead, full-sample window, Table 1)")

    summary = {
        "architecture_note": "univariate, elementwise seasonal gate x*f_A, per confirmed Algorithm 1 (2026-09-12 re-read)",
        "point_forecast_metrics": results,
        "paper_reported_rmse_1step_full_sample": 13.035,
        "extreme_event_comparison": {
            "quantile": EXTREME_QUANTILE,
            "BiLSTM_baseline": extreme_base,
            "SeasonalAttentionBiLSTM": extreme_attn,
            "pct_improvement_attention_over_baseline": round(pct_improvement, 1),
            "paper_reported_range_pct": [25, 37],
        },
    }
    with open(RESULTS / "model_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame([
        {"model": k, **v} for k, v in results.items()
    ]).to_csv(RESULTS / "model_comparison.csv", index=False)

    print(f"\nSaved results/model_comparison.json, results/model_comparison.csv")


if __name__ == "__main__":
    torch.manual_seed(42)
    run()
