"""Fetch raw data for the Lucas et al. (2020) reproduction using the gb-bm-data
package as the data layer. Saves one CSV per source to data/raw/.

Date range: 2023-07-01 to 2025-06-30 (2 years). The dissertation pipeline
pulled 2019-2026, but a fresh pull through BMRSClient is one HTTP call per
day per endpoint with a built-in rate-limit sleep, so a shorter window keeps
this runnable in one sitting while still giving ~35,000 settlement periods.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from gb_bm_data import BMRSClient, CarbonIntensityClient

START = date(2023, 7, 1)
END = date(2025, 6, 30)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    bmrs = BMRSClient()
    carbon = CarbonIntensityClient()

    print(f"Fetching system prices {START} to {END} ...")
    prices = bmrs.get_system_prices(START, END)
    prices.to_csv(RAW_DIR / "system_prices.csv", index=False)
    print(f"  {len(prices)} rows -> data/raw/system_prices.csv")

    print(f"Fetching demand outturn {START} to {END} ...")
    demand = bmrs.get_demand_outturn(START, END)
    demand.to_csv(RAW_DIR / "demand_outturn.csv", index=False)
    print(f"  {len(demand)} rows -> data/raw/demand_outturn.csv")

    print(f"Fetching generation mix {START} to {END} ...")
    mix = carbon.get_generation_mix(START, END)
    mix.to_csv(RAW_DIR / "generation_mix.csv", index=False)
    print(f"  {len(mix)} rows -> data/raw/generation_mix.csv")

    print("Done.")


if __name__ == "__main__":
    main()
