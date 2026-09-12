"""Second fetch, closer to the paper's own data window.

The main run (01_fetch_data.py) uses 2023-2025 for practicality. This script
instead pulls 2018-01-01 to 2019-07-31 (19 months), a length comparable to
what Lucas et al. (2020) describe, so results/model_comparison can be
compared across two different periods, not just presented once.

Caveat, stated plainly: the paper's exact start/end dates are not confirmed
against the original PDF from this project, only the approximate window
length. This is a comparable-length historical run, not a dated replication
of the paper's exact sample.

BMRS v2 does serve data this far back (confirmed by direct test,
2026-08-14: 2018-06-01 returned 48 settlement periods), even though the
separate dissertation pipeline this data layer was built for only pulled
from 2019-01-01 onward as a scoping choice, not an API limit.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from gb_bm_data import BMRSClient, CarbonIntensityClient

START = date(2018, 1, 1)
END = date(2019, 7, 31)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_paper_window"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    bmrs = BMRSClient()
    carbon = CarbonIntensityClient()

    print(f"Fetching system prices {START} to {END} ...")
    prices = bmrs.get_system_prices(START, END)
    prices.to_csv(RAW_DIR / "system_prices.csv", index=False)
    print(f"  {len(prices)} rows -> data/raw_paper_window/system_prices.csv")

    print(f"Fetching demand outturn {START} to {END} ...")
    demand = bmrs.get_demand_outturn(START, END)
    demand.to_csv(RAW_DIR / "demand_outturn.csv", index=False)
    print(f"  {len(demand)} rows -> data/raw_paper_window/demand_outturn.csv")

    print(f"Fetching generation mix {START} to {END} ...")
    mix = carbon.get_generation_mix(START, END)
    mix.to_csv(RAW_DIR / "generation_mix.csv", index=False)
    print(f"  {len(mix)} rows -> data/raw_paper_window/generation_mix.csv")

    print("Done.")


if __name__ == "__main__":
    main()
