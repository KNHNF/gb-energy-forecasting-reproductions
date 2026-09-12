"""Build features for the Deng et al. (2023) reproduction.

Data reuse, stated plainly: this project does not run its own fetch. It
reuses the raw CSVs already pulled by ../lucas-2020-reproduction/src/01b_
fetch_data_paper_window.py (BMRS v2 + Carbon Intensity API, 2018-01-01 to
2019-07-31, via the same gb-bm-data client). Re-fetching an overlapping
window for this project would just be the same rate-limited ~20-40 min pull
again for no new data. The window itself was chosen to be comparable in
length to Lucas et al. (2020)'s ~19 months, not to match Deng et al. (2023)'s
own sample; Deng et al.'s exact date range was not independently confirmed
from the source PDF with high confidence, so no claim is made that this
window matches theirs, only that it is a real, recent-enough GB BM sample
of comparable scale.

Target and feature set: reused from lucas-2020-reproduction/src/02_build_
features.py (balancing_price = systemBuyPrice, NIV, demand, wind/solar/gas/
nuclear mix, calendar, short lags). Deng et al. (2023)'s specific predictor
list was not confirmed with high confidence either, so this reproduction
targets their headline methodological contribution, a Seasonal-Attention
BiLSTM evaluated for extreme-event accuracy, on a defensible general GB BM
feature set, rather than claiming a feature-for-feature match this project
cannot verify.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LUCAS_RAW = Path(__file__).resolve().parents[2] / "lucas-2020-reproduction" / "data" / "raw_paper_window"
PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def load() -> pd.DataFrame:
    sp = pd.read_csv(LUCAS_RAW / "system_prices.csv")
    demand = pd.read_csv(LUCAS_RAW / "demand_outturn.csv")
    mix = pd.read_csv(LUCAS_RAW / "generation_mix.csv")

    sp = sp[[
        "settlementDate", "settlementPeriod",
        "systemBuyPrice", "netImbalanceVolume",
        "totalAcceptedOfferVolume", "totalAcceptedBidVolume",
    ]].copy()
    demand = demand[["settlementDate", "settlementPeriod", "demand_mw"]].copy()
    mix = mix[[
        "settlementDate", "settlementPeriod",
        "wind_pct", "solar_pct", "gas_pct", "nuclear_pct", "imports_pct",
    ]].copy()

    df = sp.merge(demand, on=["settlementDate", "settlementPeriod"], how="left")
    df = df.merge(mix, on=["settlementDate", "settlementPeriod"], how="left")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.dropna(subset=["systemBuyPrice"])
    df["settlementDate"] = pd.to_datetime(df["settlementDate"])
    df = df[df["demand_mw"].notna() & (df["demand_mw"] >= 10_000)]
    df = df[df["solar_pct"].isna() | (df["solar_pct"] <= 50)]
    print(f"Rows after cleaning: {len(df)} (dropped {n0 - len(df)})")
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)
    df["imbalance_price"] = df["systemBuyPrice"]

    df["hour"] = ((df["settlementPeriod"] - 1) // 2).astype(int)
    df["half_hour"] = (df["settlementPeriod"] % 2 == 0).astype(int)
    df["day_of_week"] = df["settlementDate"].dt.dayofweek
    df["month"] = df["settlementDate"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["season"] = df["month"].map({
        12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
        6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3,
    })

    df["wind_mw"] = df["wind_pct"] / 100.0 * df["demand_mw"]
    df["solar_mw"] = df["solar_pct"] / 100.0 * df["demand_mw"]

    for lag in (1, 2, 48):
        df[f"netImbalanceVolume_lag{lag}"] = df["netImbalanceVolume"].shift(lag)
    for lag in (1, 2):
        df[f"imbalance_price_lag{lag}"] = df["imbalance_price"].shift(lag)
        df[f"demand_mw_lag{lag}"] = df["demand_mw"].shift(lag)

    lag_cols = [c for c in df.columns if "_lag" in c]
    n_before = len(df)
    df = df.dropna(subset=lag_cols)
    print(f"Dropped {n_before - len(df)} rows with NaN from lagging")

    mix_cols = ["wind_pct", "solar_pct", "gas_pct", "nuclear_pct", "imports_pct"]
    n_before = len(df)
    df = df.dropna(subset=mix_cols)
    print(f"Dropped {n_before - len(df)} rows with missing generation-mix data")

    return df


def run() -> None:
    print(f"Loading raw data from {LUCAS_RAW} ...")
    df = load()
    print(f"  merged: {len(df)} rows")

    print("Cleaning...")
    df = clean(df)

    print("Engineering features...")
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
