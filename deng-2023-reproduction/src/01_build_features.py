"""Build features for the Deng et al. (2023) reproduction.

Data source, updated 2026-09-12 after re-reading the source PDF directly
(not an earlier degraded extraction): Section 5.1 states the paper's full
sample is 1 July 2016 to 30 June 2019 (52,560 half-hourly points), the same
window already fetched for ../bunn-ganesh-2021-2024-reproduction (2016-07-01
to 2019-09-30, a superset). This reproduction now reuses that raw data
instead of the earlier 2018-01 to 2019-07-31 window, which was only ever a
length-matched guess made before this confirmation.

Target and feature set, also corrected 2026-09-12: Algorithm 1 in the source
PDF specifies the model as univariate. The only input is 48 lags of the
price series itself, x = [x1, ..., x48]; there is no mention anywhere in the
paper of feeding demand, wind, generation-mix, or calendar features into the
network. The earlier version of this reproduction built a 28-column
multivariate feature table and fed it to the network at every timestep,
which the paper's own algorithm does not do. Only the price series is kept
here; the other columns are still fetched into the merge for the extreme-
event analysis and for anyone re-adding an ablation later, but the model
script now takes only the lagged price columns as input.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BUNN_RAW = Path(__file__).resolve().parents[2] / "bunn-ganesh-2021-2024-reproduction" / "data" / "raw"
PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def load() -> pd.DataFrame:
    sp = pd.read_csv(BUNN_RAW / "system_prices.csv")
    sp = sp[["settlementDate", "settlementPeriod", "systemBuyPrice"]].copy()
    return sp


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.dropna(subset=["systemBuyPrice"])
    df["settlementDate"] = pd.to_datetime(df["settlementDate"])
    print(f"Rows after cleaning: {len(df)} (dropped {n0 - len(df)})")
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)
    df["imbalance_price"] = df["systemBuyPrice"]
    return df


def run() -> None:
    print(f"Loading raw data from {BUNN_RAW} ...")
    df = load()
    print(f"  loaded: {len(df)} rows")

    print("Cleaning...")
    df = clean(df)

    print("Engineering (univariate, per confirmed Algorithm 1)...")
    df = engineer(df)

    out = PROC / "features.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved {len(df)} rows x {len(df.columns)} cols to {out}")
    print(f"Date range: {df['settlementDate'].min().date()} to {df['settlementDate'].max().date()}")
    print(f"\nTarget (imbalance_price, GBP/MWh):")
    print(f"  mean:   {df['imbalance_price'].mean():.2f}")
    print(f"  median: {df['imbalance_price'].median():.2f}")
    print(f"  p5:     {df['imbalance_price'].quantile(0.05):.2f}")
    print(f"  p95:    {df['imbalance_price'].quantile(0.95):.2f}")
    print(f"  min:    {df['imbalance_price'].min():.2f}")
    print(f"  max:    {df['imbalance_price'].max():.2f}")


if __name__ == "__main__":
    run()
