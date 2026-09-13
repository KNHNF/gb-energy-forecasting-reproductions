"""Fetch data for the Bunn et al. (2021) and Ganesh & Bunn (2024)
reproductions, both papers analysing the same GB balancing market dataset:
1 July 2016 to 30 June 2019 for estimation, extended to 30 September 2019
for Ganesh & Bunn's neural-network hold-out set. Fetched fresh via the
gb-bm-data package, not copied from any other project's archive.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from gb_bm_data import BMRSClient, CarbonIntensityClient

START = date(2016, 7, 1)
END = date(2019, 9, 30)

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

    # get_interconnector_flows() is not called here: confirmed 2026-09-12 to be
    # live-only (ignores from/to entirely), see gb-bm-data's client.py and this
    # repo's own README for the correction history.

    print(f"Fetching non-BM STOR volumes {START} to {END} ...")
    nonbm = bmrs.get_nonbm_stor(START, END)
    nonbm.to_csv(RAW_DIR / "nonbm_stor.csv", index=False)
    print(f"  {len(nonbm)} rows -> data/raw/nonbm_stor.csv"
          + ("  (empty on every window tried so far, see gb-bm-data client docstring)" if nonbm.empty else ""))

    print("Done.")


if __name__ == "__main__":
    main()
