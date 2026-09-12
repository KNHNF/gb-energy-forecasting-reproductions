"""Build the feature table for the Lucas et al. (2020) reproduction.

Target: `balancing_price` = systemBuyPrice (GBP/MWh). GB has used a single
cash-out imbalance price since the Nov 2015 P305 reform, so systemBuyPrice
and systemSellPrice are effectively the same series; SBP is used here as the
canonical "GB balancing market price" the paper forecasts. This is NOT the
same target as the dissertation (which forecasts aggregate BM *cost*,
GBP/settlement period, a different and novel target) -- this script targets
*price* specifically to match Lucas et al. (2020).

Feature groups mapped from the paper (Section 3.2) to what gb-bm-data can
actually supply:

  Paper feature                  -> This reproduction                 -> Note
  ------------------------------------------------------------------------------
  NIV                             -> netImbalanceVolume                direct match
  LOLP (5 variants, agg + ahead)  -> NIV + its lags (1, 2, 48 SP)       PROXY. No LOLP
                                                                        endpoint exists in
                                                                        BMRS v2 (confirmed).
                                                                        Only one proxy family,
                                                                        not five variants.
  Base production                 -> (gas_pct + nuclear_pct) * demand  PROXY. Paper likely used
                                                                        plant-level baseload MW;
                                                                        this is a mix-derived
                                                                        estimate.
  System load                     -> demand_mw                        direct match
  Solar generation                -> solar_pct * demand_mw             direct match (derived)
  Wind generation                 -> wind_pct * demand_mw              direct match (derived)
  Seasonality                     -> hour, half_hour, day_of_week,     direct match
                                      month, season, is_weekend
  Day-ahead price                 -> systemBuyPrice_lag48 (same SP,    PROXY. No day-ahead BM
                                      previous day)                    price feed available via
                                                                        this client; BMRSClient
                                                                        deliberately raises
                                                                        LiveOnlyEndpointError for
                                                                        forecast datasets.
  De-rated margins                -> NOT REPRODUCED                   BMRSClient.get_forecast()
                                                                        raises LiveOnlyEndpointError
                                                                        for FOU2T14D (the margin/
                                                                        availability dataset) by
                                                                        design -- confirmed
                                                                        live-only, no historical
                                                                        recovery through this
                                                                        client. Excluded rather
                                                                        than faked.
  Imbalance volume contributions  -> totalAcceptedOfferVolume,        direct match
                                      totalAcceptedBidVolume
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def load(raw_dir: Path) -> pd.DataFrame:
    sp = pd.read_csv(raw_dir / "system_prices.csv")
    demand = pd.read_csv(raw_dir / "demand_outturn.csv")
    mix = pd.read_csv(raw_dir / "generation_mix.csv")

    sp = sp[[
        "settlementDate", "settlementPeriod",
        "systemBuyPrice", "systemSellPrice", "netImbalanceVolume",
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

    # target
    df["balancing_price"] = df["systemBuyPrice"]

    # seasonality
    df["hour"] = ((df["settlementPeriod"] - 1) // 2).astype(int)
    df["half_hour"] = (df["settlementPeriod"] % 2 == 0).astype(int)
    df["day_of_week"] = df["settlementDate"].dt.dayofweek
    df["month"] = df["settlementDate"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["season"] = df["month"].map({
        12: 0, 1: 0, 2: 0,
        3: 1, 4: 1, 5: 1,
        6: 2, 7: 2, 8: 2,
        9: 3, 10: 3, 11: 3,
    })

    # generation
    df["wind_mw"] = df["wind_pct"] / 100.0 * df["demand_mw"]
    df["solar_mw"] = df["solar_pct"] / 100.0 * df["demand_mw"]
    df["base_production_mw"] = (df["gas_pct"] + df["nuclear_pct"]) / 100.0 * df["demand_mw"]

    # LOLP proxy: NIV and its lags (documented substitute, not true LOLP)
    for lag in (1, 2, 48):
        df[f"netImbalanceVolume_lag{lag}"] = df["netImbalanceVolume"].shift(lag)

    # day-ahead price proxy: same SP, previous day (48 SPs back)
    df["day_ahead_price_proxy"] = df["systemBuyPrice"].shift(48)

    # a couple of short lags on price and demand, standard EPF practice
    for lag in (1, 2):
        df[f"balancing_price_lag{lag}"] = df["balancing_price"].shift(lag)
        df[f"demand_mw_lag{lag}"] = df["demand_mw"].shift(lag)

    lag_cols = [c for c in df.columns if "_lag" in c or c == "day_ahead_price_proxy"]
    n_before = len(df)
    df = df.dropna(subset=lag_cols)
    print(f"Dropped {n_before - len(df)} rows with NaN from lagging")

    # generation mix has ~6% missing settlement periods (Carbon Intensity API
    # coverage gaps, same issue documented in the dissertation pipeline).
    # Dropped rather than imputed to keep every remaining row's features real.
    mix_cols = ["wind_pct", "solar_pct", "gas_pct", "nuclear_pct", "imports_pct"]
    n_before = len(df)
    df = df.dropna(subset=mix_cols)
    print(f"Dropped {n_before - len(df)} rows with missing generation-mix data")

    return df


def run(raw_dir: Path, out_name: str) -> None:
    print("Loading raw data...")
    df = load(raw_dir)
    print(f"  merged: {len(df)} rows")

    print("Cleaning...")
    df = clean(df)

    print("Engineering features...")
    df = engineer(df)

    out = PROC / out_name
    df.to_parquet(out, index=False)
    print(f"\nSaved {len(df)} rows x {len(df.columns)} cols to {out}")
    print(f"Date range: {df['settlementDate'].min().date()} to {df['settlementDate'].max().date()}")
    print(f"\nTarget (balancing_price, GBP/MWh):")
    print(f"  mean:   {df['balancing_price'].mean():.2f}")
    print(f"  median: {df['balancing_price'].median():.2f}")
    print(f"  min:    {df['balancing_price'].min():.2f}")
    print(f"  max:    {df['balancing_price'].max():.2f}")
    print(f"\nColumns ({len(df.columns)}): {list(df.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--out", default="features.parquet")
    args = parser.parse_args()
    run(Path(args.raw_dir), args.out)
