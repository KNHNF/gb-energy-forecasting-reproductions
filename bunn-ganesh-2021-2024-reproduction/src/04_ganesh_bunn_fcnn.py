"""Reproduce Ganesh & Bunn (2024)'s point-forecast FCNN and quantile
(pinball-loss) density forecasting for GB balancing market price, on the
same dataset and reduced feature set as 03_bunn_regime_switching.py (see
02_build_features.py for why only price lags and NIV lag2 survive from the
paper's 7-8 explanatory variables).

Point-forecast architecture: their reported best model is a fully-connected
network with two hidden layers (128, 8 units), which this reproduces.
Confidence note: hidden-layer sizes are recalled from the source PDF at
moderate confidence, not independently re-verified against the typeset
paper a second time; if wrong, they are wrong in the direction of following
the paper's own description as read, not an invented architecture.

Density forecasting: separate quantile models (shared architecture, pinball
loss) for quantiles 0.01, 0.05, 0.50, 0.95, 0.99, matching the tail
quantiles the paper reports. No paper pinball-loss numbers are reproduced
here (not confidently recalled from the source), only empirical interval
coverage, which is reported as a real, verifiable evaluation on this
reproduction's own predictions rather than a claimed match to unknown paper
figures.
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

FEATURE_COLS = ["price_lag1", "price_lag2", "niv_lag2", "nonbm_stor_lag2", "hour", "day_of_week", "month", "is_weekend"]
TARGET = "price"
QUANTILES = [0.01, 0.05, 0.50, 0.95, 0.99]

PAPER_POINT_RESULT = {"RMSE": 14.5, "MAE": 11.4, "R2": 0.66}
PAPER_REGIME_SWITCHING_BENCHMARK_RMSE = 14.7  # paper's own reported RS benchmark, from Bunn et al. 2021


class FCNN(nn.Module):
    def __init__(self, n_features: int, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(),
            nn.Linear(128, 8), nn.ReLU(),
            nn.Linear(8, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def pinball_loss(preds: torch.Tensor, target: torch.Tensor, quantiles: list[float]) -> torch.Tensor:
    losses = []
    for i, q in enumerate(quantiles):
        err = target - preds[:, i]
        losses.append(torch.max((q - 1) * err, q * err))
    return torch.stack(losses, dim=1).mean()


def train(model, X_train, y_train, loss_fn, epochs: int = 100, batch_size: int = 256) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        if epoch % 20 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}/{epochs}  loss {total / len(dataset):.3f}")


def run() -> None:
    df = pd.read_parquet(PROC / "features.parquet")
    df = df.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    X_raw = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_raw = df[TARGET].to_numpy(dtype=np.float32)

    scaler = StandardScaler().fit(X_raw[:split_idx])
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    X_train = torch.from_numpy(X_scaled[:split_idx])
    y_train = torch.from_numpy(y_raw[:split_idx])
    X_test = torch.from_numpy(X_scaled[split_idx:])
    y_test = y_raw[split_idx:]

    print(f"Train: {len(X_train)}, Test: {len(X_test)}, features: {FEATURE_COLS}")

    # --- point forecast FCNN ---
    print("\nTraining point-forecast FCNN (128, 8 hidden units)...")
    point_model = FCNN(n_features=len(FEATURE_COLS), out_dim=1)
    train(point_model, X_train, y_train.unsqueeze(1), nn.MSELoss())

    point_model.eval()
    with torch.no_grad():
        point_preds = point_model(X_test).squeeze(1).numpy()

    point_rmse = float(np.sqrt(mean_squared_error(y_test, point_preds)))
    point_mae = float(mean_absolute_error(y_test, point_preds))
    point_r2 = float(r2_score(y_test, point_preds))
    print(f"  RMSE: {point_rmse:.2f}  MAE: {point_mae:.2f}  R2: {point_r2:.2f}")

    # --- quantile / density forecast FCNN ---
    print(f"\nTraining quantile FCNN for quantiles {QUANTILES}...")
    quantile_model = FCNN(n_features=len(FEATURE_COLS), out_dim=len(QUANTILES))
    loss_fn = lambda preds, target: pinball_loss(preds, target.squeeze(1), QUANTILES)
    train(quantile_model, X_train, y_train.unsqueeze(1), loss_fn)

    quantile_model.eval()
    with torch.no_grad():
        quantile_preds = quantile_model(X_test).numpy()

    # empirical coverage: does the [5th, 95th] predicted interval actually
    # contain the true price the stated % of the time?
    lo_idx, hi_idx = QUANTILES.index(0.05), QUANTILES.index(0.95)
    covered_90 = np.mean((y_test >= quantile_preds[:, lo_idx]) & (y_test <= quantile_preds[:, hi_idx]))
    lo_idx2, hi_idx2 = QUANTILES.index(0.01), QUANTILES.index(0.99)
    covered_98 = np.mean((y_test >= quantile_preds[:, lo_idx2]) & (y_test <= quantile_preds[:, hi_idx2]))

    pinball_by_q = {}
    for i, q in enumerate(QUANTILES):
        err = y_test - quantile_preds[:, i]
        pb = np.mean(np.maximum((q - 1) * err, q * err))
        pinball_by_q[str(q)] = round(float(pb), 3)

    print(f"  empirical 90% interval coverage (target 0.90): {covered_90:.3f}")
    print(f"  empirical 98% interval coverage (target 0.98): {covered_98:.3f}")
    print(f"  pinball loss by quantile: {pinball_by_q}")

    results = {
        "point_forecast": {
            "this_reproduction": {"RMSE": round(point_rmse, 2), "MAE": round(point_mae, 2), "R2": round(point_r2, 2)},
            "paper_FCNN": PAPER_POINT_RESULT,
            "paper_regime_switching_benchmark_RMSE": PAPER_REGIME_SWITCHING_BENCHMARK_RMSE,
        },
        "density_forecast": {
            "quantiles": QUANTILES,
            "empirical_coverage_90pct_interval": round(float(covered_90), 3),
            "empirical_coverage_98pct_interval": round(float(covered_98), 3),
            "pinball_loss_by_quantile": pinball_by_q,
            "note": "no paper pinball-loss numbers reproduced here (not confidently recalled "
                    "from the source PDF); coverage and pinball loss are reported as found on "
                    "this reproduction's own predictions.",
        },
    }
    with open(RESULTS / "ganesh_bunn_2024_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("PAPER COMPARISON (point forecast)")
    print("=" * 60)
    print(f"{'Metric':<8}{'Paper FCNN':<14}{'This reproduction':<20}")
    print(f"{'RMSE':<8}{PAPER_POINT_RESULT['RMSE']:<14}{round(point_rmse, 2):<20}")
    print(f"{'MAE':<8}{PAPER_POINT_RESULT['MAE']:<14}{round(point_mae, 2):<20}")
    print(f"{'R2':<8}{PAPER_POINT_RESULT['R2']:<14}{round(point_r2, 2):<20}")

    print(f"\nSaved results/ganesh_bunn_2024_comparison.json")


if __name__ == "__main__":
    torch.manual_seed(42)
    run()
