# Lucas et al. (2020) reproduction

An independent reproduction of the methodology in:

> Lucas, A., Pegios, K., Kotsakis, E. and Clarke, D. (2020) 'Price Forecasting
> for the Balancing Energy Market Using Machine-Learning Regression',
> *Energies*, 13(20), p. 5420. https://doi.org/10.3390/en13205420

No public code repository exists for this paper (checked 2026-08-14). This
project rebuilds the paper's feature set and model comparison from scratch,
using [gb-bm-data](https://github.com/KNHNF/gb-bm-data) (a client for the Elexon BMRS API
and the Carbon Intensity API, built for a separate dissertation project) as
the data layer.

**This is a reproduction attempt, not a replication.** The result differs
from the paper's, honestly, and the reasons why are documented below rather
than adjusted away.

## What this predicts

Target: `balancing_price` = System Buy Price (GBP/MWh) per settlement
period. GB has used a single cash-out imbalance price since the Nov 2015
P305 reform, so System Buy Price and System Sell Price are effectively the
same series; SBP is used here as "the GB balancing market price" the paper
forecasts.

This is a different target from the separate GB BM cost-forecasting
dissertation this data layer was originally built for, which forecasts
*aggregate BM cost* (GBP per settlement period, summed across all accepted
bids/offers), a novel target with no prior published ML paper. This project
instead targets *price*, to match what Lucas et al. (2020) actually forecast.

## Data

Fetched fresh via `gb_bm_data.BMRSClient` and `gb_bm_data.CarbonIntensityClient`,
not copied from the dissertation's own archive. Two windows are fetched and
run through the full pipeline independently:

| | Main run | Paper-parity run |
|---|---|---|
| Script | `src/01_fetch_data.py` | `src/01b_fetch_data_paper_window.py` |
| Range | 2023-07-01 to 2025-06-30 (2 years) | 2018-01-01 to 2019-07-31 (19 months) |
| Why | Recent data, practical to fetch in one sitting | Length comparable to the paper's own window |
| Final feature rows | 32,706 | 19,853 |

The paper's exact start/end dates are not confirmed against the original
PDF, only the approximate window length ("2018 to mid-2019" in the source
material used here); the paper-parity run is a comparable-length historical
sample, not a dated match to the paper's exact data.

- **Sources (both runs):** BMRS `/balancing/settlement/system-prices`
  (SBP, SSP, NIV, accepted offer/bid volumes), BMRS `/generation/outturn`
  (system demand), Carbon Intensity API `/generation` (wind/solar/gas/nuclear/
  imports mix).
- **Gaps:** the Carbon Intensity API has real coverage gaps (same issue
  documented in the dissertation pipeline), worse further back in time. Main
  run: 2,311 of 35,065 rows (6.6%) dropped for missing generation-mix data.
  Paper-parity run: 7,626 of 27,527 rows (27.7%) dropped, and the resulting
  feature table only starts 2018-05-10 even though the fetch requested from
  2018-01-01, because the Carbon Intensity API's own coverage is thinner that
  early. Both runs drop rather than impute, so every feature in the final
  tables is a real measurement, not a filled value.

Neither `data/processed/*.parquet` file is committed, regenerate with the
fetch script for the window you want, then `src/02_build_features.py`.

## Feature mapping: paper vs. this reproduction

| Paper feature (Section 3.2) | This reproduction | Status |
|---|---|---|
| NIV | `netImbalanceVolume` | Direct match |
| LOLP (5 variants, aggregated + time-ahead) | `netImbalanceVolume` and its lags (1, 2, 48 SP) | **Proxy.** No LOLP endpoint exists in BMRS (confirmed by direct testing: `/system/lolp`, `/balancing/lolp`, `/forecast/surplus/daily` all return empty or 404). NIV is a documented partial correlate of LOLP, but this reproduction has one proxy family, not five variants. |
| Base production | `(gas_pct + nuclear_pct) / 100 * demand_mw` | **Proxy.** Derived from generation-mix percentages, not plant-level baseload MW. |
| System load | `demand_mw` | Direct match |
| Solar generation | `solar_pct / 100 * demand_mw` | Direct match (derived) |
| Wind generation | `wind_pct / 100 * demand_mw` | Direct match (derived) |
| Seasonality | `hour`, `half_hour`, `day_of_week`, `month`, `season`, `is_weekend` | Direct match |
| Day-ahead price | `balancing_price` shifted 48 SPs (same period, previous day) | **Proxy.** No day-ahead BM price feed is available through this client; `BMRSClient.get_forecast()` deliberately raises `LiveOnlyEndpointError` for forecast datasets rather than silently returning wrong data. |
| De-rated margins (aggregated + contributions) | **Not reproduced** | `BMRSClient.get_forecast("FOU2T14D")` raises `LiveOnlyEndpointError` by design: this dataset was confirmed live-only (ignores historical date params) during real dissertation data collection, August 2026. Excluded rather than faked. |
| Imbalance volume contributions | `totalAcceptedOfferVolume`, `totalAcceptedBidVolume` | Direct match |

Two feature groups the paper doesn't have but this reproduction adds:
`balancing_price_lag1`/`lag2` (short autoregressive lags of the target
itself, standard practice in electricity price forecasting) and
`day_ahead_price_proxy`. These are the two dominant features in the trained
XGBoost model (see below), so they materially affect the result and are
flagged rather than hidden.

## Models

Gradient Boosting, Random Forest, and XGBoost (`src/03_train_evaluate.py`),
the three the paper compares: scikit-learn / XGBoost defaults with
`n_estimators=300` for RF and XGBoost. Chronological 80/20 split (train on
the earlier 26,164 rows, test on the most recent 6,542), no shuffling, since
this is a time series and the paper uses an out-of-sample evaluation scheme.

An LSTM (`src/04_lstm_model.py`, PyTorch) is added as a fourth model on the
same split and feature set, a lookback window of 48 settlement periods (24
hours) feeding a single-layer LSTM plus a linear head, 20 epochs. This is
**not** one of the paper's own models; it is included because LSTM is a
common benchmark elsewhere in GB/EPF literature (e.g. Deng et al. 2023 for
GB imbalance prices), so there is no paper number to reproduce for it, it is
reported alongside the tree models as an extra data point.

`systemBuyPrice` and `systemSellPrice` are excluded from the feature set:
`systemBuyPrice` is the target itself, and `systemSellPrice` tracks it
almost exactly under the single-price system, so including it would be
target leakage.

## Results

**Main run, 2023-2025:**

| Model | MAE | MSE | R2 |
|---|---|---|---|
| Gradient Boosting | 10.78 | 251.11 | 87.3% |
| Random Forest | 10.74 | 261.61 | 86.8% |
| XGBoost | 11.85 | 305.62 | 84.6% |
| LSTM | 20.87 | 775.17 | 60.9% |

**Paper-parity run, 2018-2019:**

| Model | MAE | MSE | R2 |
|---|---|---|---|
| Gradient Boosting | 7.78 | 151.95 | 68.0% |
| XGBoost | 7.88 | 161.86 | 65.9% |
| Random Forest | 8.66 | 170.89 | 64.0% |
| LSTM | 11.35 | 296.89 | 37.5% |

The LSTM underperforms all three tree models on both runs, by a wide margin.
This matches a recurring finding elsewhere in the GB/EPF literature
(O'Connor et al. 2024 found LEAR beats deep learning in the Irish balancing
market): tree ensembles tend to win on tabular electricity-price features
unless the deep model gets considerably more tuning and data than a
20-epoch single-layer LSTM gets here. No attempt was made to tune the LSTM
competitive; it is reported as run.

**Comparison against the paper (XGBoost, the paper's best model):**

| Metric | Paper (Lucas et al. 2020) | Main run (2023-2025) | Paper-parity run (2018-2019) |
|---|---|---|---|
| MAE | 7.89 | 11.85 | 7.88 |
| R2 | 76.8% | 84.6% | 65.9% |
| MSE | 124.74 | 305.62 | 161.86 |

The paper's currency symbol is corrupted in the PDF text extraction used to
source these numbers; MAE 7.89 is almost certainly GBP/MWh given this is the
GB market, but that has not been independently confirmed against the
original typeset PDF.

**The paper-parity run's XGBoost MAE (7.88) lands almost exactly on the
paper's reported 7.89.** This is a genuinely striking result and is reported
as found, with no tuning applied to reach it. It should not be read as
proof the reproduction matches the paper: R2 (65.9% vs 76.8%) and MSE
(161.86 vs 124.74) both diverge in the same direction as the main run, and
Gradient Boosting, not XGBoost, is the best model on this window too (MAE
7.78). One matching metric out of three, on a model that wasn't even the
best performer in this run, is closer to a striking coincidence on typical
error magnitude than a validated match, especially given the acknowledged
gaps in LOLP and de-rated margins.

**No methodology choice here was tuned to force either result.** Real,
honest differences from the paper, present on both runs:

1. **XGBoost was not the best model on either run.** Gradient Boosting beat
   it on both the main run (87.3% vs 84.6% R2) and the paper-parity run
   (68.0% vs 65.9% R2), the opposite ranking from the paper. All models were
   run with library defaults plus `n_estimators=300`, no per-model tuning
   in either direction.
2. **R2 is consistently lower on the paper-parity run than the main run**,
   despite the closer MAE match, likely because GB balancing prices were
   calmer and more range-bound in 2018-2019 (mean 51.68, max 375.00 GBP/MWh)
   than in the main run's more volatile 2023-2025 window (mean 75.66, max
   669.21 GBP/MWh): a calmer target is harder to explain proportionally even
   when the absolute error is smaller.
3. **Missing LOLP and de-rated margins matter on both runs.** The paper
   ranks LOLP (aggregated) as its second most important feature at 27.5%
   and de-rated margins at 8.9%, together over a third of total feature
   importance. Neither is reproducible through gb-bm-data (see feature
   mapping table above). In the trained XGBoost model on the main run,
   `netImbalanceVolume` (33.7%) and `balancing_price_lag1` (32.1%) dominate
   instead, with `base_production_mw` third (9.0%), a plausible
   redistribution of importance onto the features that remain, but not
   evidence the model is equivalent to the paper's.

Full feature importance: `results/feature_importance.csv` (main run),
`results/feature_importance_paper_window.csv` (paper-parity run). Raw
numbers: `results/model_comparison*.csv`, `results/paper_comparison*.json`.

## What this is and isn't

This reproduces the paper's model comparison methodology and feature
categories as closely as a fully public, no-auth, historical GB dataset
allows, using a data client that fails loudly (raises, doesn't silently
substitute) on the two feature groups it cannot serve. It is not a
replication: even the paper-parity run's date range is an estimate of the
paper's window length, not a confirmed match to its exact dates; two of the
paper's feature groups (LOLP, de-rated margins) are approximated or absent
on both runs; and XGBoost, the paper's best model, is not the best model in
either reproduction run here. That XGBoost's MAE lands close to the paper's
on one run while its R2 and MSE, and its model ranking, still diverge, is
the actual finding worth reporting, not a bug to be tuned away.

## Reproducing this

```bash
pip install -e ../gb-bm-data
pip install -r requirements.txt

# main run (2023-2025)
python src/01_fetch_data.py                            # ~20-40 min, rate-limited API calls
python src/02_build_features.py
python src/03_train_evaluate.py                         # Gradient Boosting, Random Forest, XGBoost
python src/04_lstm_model.py                              # LSTM, appends to the same results table

# paper-parity run (2018-2019)
python src/01b_fetch_data_paper_window.py                # ~20-40 min
python src/02_build_features.py --raw-dir data/raw_paper_window --out features_paper_window.parquet
python src/03_train_evaluate.py --features data/processed/features_paper_window.parquet --suffix _paper_window
python src/04_lstm_model.py --features data/processed/features_paper_window.parquet --suffix _paper_window
```

## Citation

If referencing the original paper:

```
Lucas, A., Pegios, K., Kotsakis, E. and Clarke, D. (2020) 'Price Forecasting
for the Balancing Energy Market Using Machine-Learning Regression',
Energies, 13(20), p. 5420.
```

## Author

Karan Homayounfar ([KNHNF](https://github.com/KNHNF)). Built as an
independent reproduction exercise alongside a separate MSc dissertation on
GB balancing market cost forecasting, which [gb-bm-data](https://github.com/KNHNF/gb-bm-data) was
originally extracted from.

## Licence

MIT, see [LICENSE](LICENSE).
