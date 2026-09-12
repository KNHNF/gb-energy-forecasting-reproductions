"""Build features for the Bunn et al. (2021) / Ganesh & Bunn (2024)
reproductions, which share one underlying dataset and lineage: Ganesh &
Bunn (2024) directly benchmarks its neural-network models against Bunn et
al. (2021)'s regime-switching model on the same GB balancing-market sample.

Of the paper's own explanatory variables, only two are reproducible through
gb-bm-data:

  Paper variable (both papers)      This reproduction        Status
  ---------------------------------------------------------------------
  System/Imbalance Price (lagged)    systemBuyPrice, lagged   Direct match
  NIV (lag 2)                        netImbalanceVolume,      Direct match
                                      lag 2
  De-rated Margin (DRM, lag 2)       NOT REPRODUCED            BMRSClient.get_forecast
                                                                ("FOU2T14D") raises
                                                                LiveOnlyEndpointError by
                                                                design; no historical
                                                                margin/availability data
                                                                recoverable through this
                                                                client.
  Wind/Solar/Demand forecast error   NOT REPRODUCED            Requires day-ahead
  (lag 2, each)                                                forecasts vs actuals;
                                                                BMRS v2's forecast
                                                                datasets are live-only
                                                                (same LiveOnlyEndpointError
                                                                guard). gb-bm-data has
                                                                actuals only, no
                                                                historical forecast to
                                                                diff against.
  NONBM (non-BM STOR volumes)        NOT REPRODUCED            No endpoint for this in
                                                                gb-bm-data; would need a
                                                                separate BMRS dataset this
                                                                client does not wrap.
  Inter Delta (interconnector flow   NOT REPRODUCED            No interconnector-flow
  change, lag 2)                                               endpoint in gb-bm-data.
  LOLP (Ganesh & Bunn only,          netImbalanceVolume        PROXY, same limitation as
  sparse binary dummy)               (as in the other          documented in
                                      reproductions in this     lucas-2020-reproduction:
                                      data layer)               no LOLP endpoint in BMRS
                                                                v2.

This is a more severe data gap than Lucas et al. (2020): of 7-8 explanatory
variables across the two papers, only System Price and NIV survive intact.
Reported honestly rather than worked around by inventing proxies for
variables gb-bm-data genuinely cannot supply.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def load() -> pd.DataFrame:
    sp = pd.read_csv(RAW / "system_prices.csv")
    demand = pd.read_csv(RAW / "demand_outturn.csv")
    mix = pd.read_csv(RAW / "generation_mix.csv")

    sp = sp[["settlementDate", "settlementPeriod", "systemBuyPrice", "netImbalanceVolume"]].copy()
    demand = demand[["settlementDate", "settlementPeriod", "demand_mw"]].copy()
    mix = mix[["settlementDate", "settlementPeriod", "wind_pct", "solar_pct"]].copy()

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
    df["price"] = df["systemBuyPrice"]

    # paper regressors: price AR lags, NIV lag 2 (the two survivors)
    for lag in (1, 2):
        df[f"price_lag{lag}"] = df["price"].shift(lag)
    df["niv_lag2"] = df["netImbalanceVolume"].shift(2)

    # calendar, used by the density-forecast (Ganesh & Bunn) feature set only
    df["hour"] = ((df["settlementPeriod"] - 1) // 2).astype(int)
    df["day_of_week"] = df["settlementDate"].dt.dayofweek
    df["month"] = df["settlementDate"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    lag_cols = ["price_lag1", "price_lag2", "niv_lag2"]
    n_before = len(df)
    df = df.dropna(subset=lag_cols)
    print(f"Dropped {n_before - len(df)} rows with NaN from lagging")

    return df


def run() -> None:
    print("Loading raw data...")
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
    print(f"\nTarget (price, GBP/MWh):")
    print(f"  mean:   {df['price'].mean():.2f}")
    print(f"  median: {df['price'].median():.2f}")
    print(f"  min:    {df['price'].min():.2f}")
    print(f"  max:    {df['price'].max():.2f}")


if __name__ == "__main__":
    run()
