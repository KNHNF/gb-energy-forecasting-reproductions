"""Train Gradient Boosting, Random Forest, and XGBoost on balancing_price and
report MAE, R2, MSE against Lucas et al. (2020)'s reported XGBoost result
(MAE 7.89, R2 76.8%, MSE 124.74 -- currency symbol corrupted in the PDF
extraction, almost certainly GBP given this is the GB market, not fully
confirmed against the original typesetting).

Split: chronological 80/20 (train on the earlier period, test on the most
recent 20%), consistent with the paper's out-of-sample evaluation approach
and standard EPF practice (no shuffling -- this is a time series).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

TARGET = "balancing_price"
DROP_COLS = {
    "settlementDate", "settlementPeriod", TARGET,
    "systemBuyPrice", "systemSellPrice",  # target leakage: SBP is the target itself, SSP tracks it near-exactly
}

PAPER_RESULT = {"MAE": 7.89, "R2_pct": 76.8, "MSE": 124.74}


def load_split(features_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_parquet(features_path)
    df = df.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols]
    y = df[TARGET]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train: {len(X_train)} rows ({df['settlementDate'].iloc[0].date()} to "
          f"{df['settlementDate'].iloc[split_idx-1].date()})")
    print(f"Test:  {len(X_test)} rows ({df['settlementDate'].iloc[split_idx].date()} to "
          f"{df['settlementDate'].iloc[-1].date()})")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    return X_train, X_test, y_train, y_test


def evaluate(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds) * 100
    print(f"\n{name}")
    print(f"  MAE: {mae:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  R2:  {r2:.1f}%")
    return {"model": name, "MAE": round(mae, 2), "MSE": round(mse, 2), "R2_pct": round(r2, 1)}, model


def feature_importance(model, feature_cols: list[str], name: str) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        return pd.DataFrame()
    fi = pd.DataFrame({"feature": feature_cols, "importance": imp})
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    fi["model"] = name
    return fi


def run(features_path: Path, prefix: str) -> None:
    X_train, X_test, y_train, y_test = load_split(features_path)
    feature_cols = list(X_train.columns)

    results = []
    importances = []

    gb = GradientBoostingRegressor(random_state=42)
    r, m = evaluate("GradientBoosting", gb, X_train, X_test, y_train, y_test)
    results.append(r)
    importances.append(feature_importance(m, feature_cols, "GradientBoosting"))

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    r, m = evaluate("RandomForest", rf, X_train, X_test, y_train, y_test)
    results.append(r)
    importances.append(feature_importance(m, feature_cols, "RandomForest"))

    xgb_model = xgb.XGBRegressor(random_state=42, n_estimators=300, n_jobs=-1)
    r, m = evaluate("XGBoost", xgb_model, X_train, X_test, y_train, y_test)
    results.append(r)
    importances.append(feature_importance(m, feature_cols, "XGBoost"))

    comparison_name = f"model_comparison{prefix}.csv"
    fi_name = f"feature_importance{prefix}.csv"
    pc_name = f"paper_comparison{prefix}.json"

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS / comparison_name, index=False)

    fi_df = pd.concat(importances, ignore_index=True)
    fi_df.to_csv(RESULTS / fi_name, index=False)

    xgb_result = next(r for r in results if r["model"] == "XGBoost")
    comparison = {"paper_xgboost": PAPER_RESULT, "this_reproduction_xgboost": xgb_result}
    with open(RESULTS / pc_name, "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 60)
    print("PAPER COMPARISON (XGBoost, best model in both)")
    print("=" * 60)
    print(f"{'Metric':<10}{'Paper (Lucas 2020)':<22}{'This reproduction':<20}")
    print(f"{'MAE':<10}{PAPER_RESULT['MAE']:<22}{xgb_result['MAE']:<20}")
    print(f"{'R2 (%)':<10}{PAPER_RESULT['R2_pct']:<22}{xgb_result['R2_pct']:<20}")
    print(f"{'MSE':<10}{PAPER_RESULT['MSE']:<22}{xgb_result['MSE']:<20}")

    print(f"\nSaved results/{comparison_name}, results/{fi_name}, results/{pc_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(PROC / "features.parquet"))
    parser.add_argument("--suffix", default="", help="appended to output filenames, e.g. _paper_window")
    args = parser.parse_args()
    run(Path(args.features), args.suffix)
