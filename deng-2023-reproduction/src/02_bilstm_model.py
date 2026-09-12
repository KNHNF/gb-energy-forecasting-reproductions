"""Seasonal-attention BiLSTM reproduction of Deng et al. (2023)'s headline
architecture, plus a plain BiLSTM ablation baseline.

Architecture, stated plainly since exact hyperparameters could not be
confirmed with high confidence from the source PDF: a bidirectional LSTM
over a lookback window of settlement periods, followed by an additive
(Bahdanau-style) attention layer over the BiLSTM's per-timestep outputs, a
linear head. "Seasonal" here means the calendar features (hour, day_of_week,
month, season) already in the feature table are part of the model's input
vector at every timestep, not a separate seasonal sub-model, again because
the paper's exact seasonal-conditioning mechanism was not confirmed from the
extracted text.

Evaluation follows the paper's own framing: report overall MAE/RMSE, then
MAE specifically on extreme price events (bottom 5% / top 5% of the test
period's true prices) and the % improvement of the attention model over the
plain-BiLSTM baseline on that extreme subset, matching Deng et al. (2023)'s
headline claim of a 25-37% improvement on extreme events. The exact number
is not expected to match; the comparison structure is.
"""
from __future__ import annotations

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

TARGET = "imbalance_price"
DROP_COLS = {"settlementDate", "settlementPeriod", TARGET, "systemBuyPrice"}

SEQ_LEN = 48
EPOCHS = 25
BATCH_SIZE = 128
HIDDEN_SIZE = 64
EXTREME_QUANTILE = 0.05  # bottom/top 5%, matching the paper's "extreme events" framing


class BiLSTMBaseline(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class SeasonalAttentionBiLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True, bidirectional=True)
        self.attn_score = nn.Linear(hidden_size * 2, 1)
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)                       # (batch, seq, 2*hidden)
        scores = self.attn_score(out)                # (batch, seq, 1)
        weights = torch.softmax(scores, dim=1)        # attention over time steps
        context = (weights * out).sum(dim=1)          # (batch, 2*hidden)
        return self.head(context).squeeze(-1)


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(X) - seq_len
    Xs = np.stack([X[i:i + seq_len] for i in range(n)])
    ys = y[seq_len:]
    return Xs.astype(np.float32), ys.astype(np.float32)


def train_model(model, X_train_t, y_train_t, epochs: int = EPOCHS) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
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
            print(f"    epoch {epoch:2d}/{epochs}  train MSE {total_loss / len(dataset):.2f}")


def evaluate(model, X_test_t, y_test: np.ndarray) -> tuple[np.ndarray, dict]:
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).numpy()
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds) * 100
    return preds, {"MAE": round(float(mae), 2), "MSE": round(float(mse), 2), "R2_pct": round(float(r2), 1)}


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
    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    split_idx = int(len(df) * 0.8)
    X_raw = df[feature_cols].to_numpy()
    y_raw = df[TARGET].to_numpy()

    scaler = StandardScaler().fit(X_raw[:split_idx])
    X_scaled = scaler.transform(X_raw)

    X_train_seq, y_train_seq = make_sequences(X_scaled[:split_idx], y_raw[:split_idx], SEQ_LEN)
    X_test_seq, y_test_seq = make_sequences(X_scaled[split_idx - SEQ_LEN:], y_raw[split_idx - SEQ_LEN:], SEQ_LEN)
    print(f"Train sequences: {len(X_train_seq)}, test sequences: {len(X_test_seq)}, "
          f"seq_len={SEQ_LEN}, n_features={len(feature_cols)}")

    X_train_t = torch.from_numpy(X_train_seq)
    y_train_t = torch.from_numpy(y_train_seq)
    X_test_t = torch.from_numpy(X_test_seq)

    results = {}
    preds_by_model = {}

    print("\nTraining plain BiLSTM baseline...")
    baseline = BiLSTMBaseline(n_features=len(feature_cols))
    train_model(baseline, X_train_t, y_train_t)
    preds_base, metrics_base = evaluate(baseline, X_test_t, y_test_seq)
    print(f"  BiLSTM baseline: {metrics_base}")
    results["BiLSTM_baseline"] = metrics_base
    preds_by_model["BiLSTM_baseline"] = preds_base

    print("\nTraining seasonal-attention BiLSTM...")
    attn_model = SeasonalAttentionBiLSTM(n_features=len(feature_cols))
    train_model(attn_model, X_train_t, y_train_t)
    preds_attn, metrics_attn = evaluate(attn_model, X_test_t, y_test_seq)
    print(f"  Seasonal-Attention BiLSTM: {metrics_attn}")
    results["SeasonalAttentionBiLSTM"] = metrics_attn
    preds_by_model["SeasonalAttentionBiLSTM"] = preds_attn

    extreme_base = extreme_event_mae(y_test_seq, preds_base)
    extreme_attn = extreme_event_mae(y_test_seq, preds_attn)
    pct_improvement = 100 * (extreme_base["MAE_extreme"] - extreme_attn["MAE_extreme"]) / extreme_base["MAE_extreme"]

    print(f"\nExtreme-event MAE (bottom/top {int(EXTREME_QUANTILE*100)}%, n={extreme_base['n_extreme']}):")
    print(f"  BiLSTM baseline:           {extreme_base['MAE_extreme']}")
    print(f"  Seasonal-Attention BiLSTM: {extreme_attn['MAE_extreme']}")
    print(f"  Improvement: {pct_improvement:.1f}%")
    print(f"  (paper reports 25-37% improvement on extreme events)")

    summary = {
        "point_forecast_metrics": results,
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
