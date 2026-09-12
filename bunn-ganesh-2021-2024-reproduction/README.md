# Bunn et al. (2021) and Ganesh & Bunn (2024) reproduction

An independent reproduction of two related papers analysing the same GB
balancing market dataset, the second directly benchmarking against the
first:

> Bunn, D., Andresen, A., Chen, D. and Westgaard, S. (2021) 'Analysis of the
> Fundamental Predictability of Prices in the British Balancing Market',
> *IEEE Transactions on Power Systems*.

> Ganesh, S. and Bunn, D. (2024) 'Forecasting Imbalance Price Densities With
> Statistical Methods and Neural Networks', *IEEE ...*.

Both PDFs were read directly from Zotero storage for this reproduction
(2026-08-14/15), not sourced from the dissertation lit review's one-line
summaries alone.

**This is a reproduction attempt, not a replication**, and a more
data-constrained one than [lucas-2020-reproduction](../lucas-2020-reproduction):
of the two papers' 7-8 explanatory variables, only 2 are reproducible
through gb-bm-data. See the feature mapping in `src/02_build_features.py`.

## Data

Fetched fresh via `gb_bm_data.BMRSClient` and `gb_bm_data.CarbonIntensityClient`
(`src/01_fetch_data.py`): **2016-07-01 to 2019-09-30**, matching both papers'
own estimation window (1 July 2016 to 30 June 2019) extended to Ganesh &
Bunn's neural-network hold-out (30 September 2019). 56,970 raw rows, 56,513
after cleaning and lag construction.

Generation-mix coverage is thin this far back (23,376 of 56,970 raw rows,
~41%), worse than either of `lucas-2020-reproduction`'s two windows,
consistent with the Carbon Intensity API's own coverage improving over
time. This does not affect the feature set actually used here though, since
neither paper's reproducible regressors (price, NIV) come from the
generation-mix source.

## What's reproducible and what isn't

Of the two papers' explanatory variables:

| Variable | This reproduction | Status |
|---|---|---|
| System/Imbalance Price (AR lags) | `price_lag1`, `price_lag2` (or `order=2` in `MarkovAutoregression`) | Direct match |
| NIV (lag 2) | `niv_lag2` | Direct match |
| De-rated Margin (DRM, lag 2) | Not reproduced | `BMRSClient.get_forecast("FOU2T14D")` raises `LiveOnlyEndpointError` by design; no historical margin data recoverable |
| Wind/Solar/Demand forecast error (lag 2, each) | Not reproduced | Requires day-ahead forecasts vs actuals; BMRS v2's forecast datasets are live-only, gb-bm-data has actuals only |
| NONBM (non-BM STOR volumes) | Not reproduced | No endpoint for this in gb-bm-data |
| Inter Delta (interconnector flow change) | Not reproduced | No interconnector-flow endpoint in gb-bm-data |
| LOLP (Ganesh & Bunn's sparse binary dummy) | Not reproduced | No LOLP endpoint in BMRS v2, same gap documented across this data layer's other reproductions |

**Two of 7-8 regressors survive.** This is reported upfront because it
materially limits how far either reproduction can be expected to match its
paper, well beyond the LOLP/de-rated-margin gap already documented for
Lucas et al. (2020).

## Part 1: Bunn et al. (2021), regime-switching regression

`src/03_bunn_regime_switching.py`. A 2-regime Markov-switching
autoregression (`statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression`,
order=2, time-varying transition probabilities driven by `niv_lag2`,
switching AR coefficients, switching exog effect, switching variance),
against a linear AR(2)+NIV OLS benchmark, matching the paper's own two-model
comparison.

**A numerical stability fix was required and is disclosed here in full.**
The raw price series has extreme spikes (max 1528.72 GBP/MWh, 29 points
above 500) that pushed the Hamilton filter's likelihood evaluation into
numerical underflow, collapsing the MLE fit to a degenerate solution
(all-zero coefficients, NaN log-likelihood) regardless of which switching
options were enabled, confirmed across several isolated diagnostic runs.
Winsorizing price at the [0.1, 99.9] percentiles (computed on the train
split only) before fitting the switching model fixed this. **RMSE is still
computed against the true, unclipped price** in both the in-sample and
out-of-sample figures below; the clipping only affects what the optimizer
sees during estimation. This is a real, deliberate deviation from the paper
(which does not report clipping), made purely for MLE convergence with this
library.

Evaluation scope, stated plainly: this reproduces the paper's in-sample RMSE
and one out-of-sample backtest RMSE (chronological 80/20 split, consistent
with the rest of this data layer's reproductions). The paper also reports a
train-2017-19/forecast-2016 backtest and a Nov-Dec 2016 rolling backtest;
neither is reproduced here.

**Results:**

| | Paper: regime-switching | Paper: linear | This repro: regime-switching | This repro: linear |
|---|---|---|---|---|
| In-sample RMSE | 19.29 | 18.98 | 27.43 | 25.78 |
| Out-of-sample RMSE | 14.7 | 15.4 | 16.47 | 15.37 |

**The regime-switching model underperforms its own linear benchmark in this
reproduction, on both in-sample and out-of-sample RMSE, the opposite of the
paper's finding** (where regime-switching modestly but consistently beats
linear, especially out-of-sample). This is reported as found. The fitted
regimes themselves are informative about why: regime 0 has sigma2 = 0.33
(near-deterministic) and regime 1 has sigma2 = 929 (highly volatile), with
both regimes' AR(1) coefficient close to 1.0 (near-unit-root) and nearly
identical constants (15.80 vs 15.79). The model has essentially found a
volatility-clustering split (quiet periods vs turbulent periods) rather than
two genuinely different price-generating regimes with different dynamics,
which is a real, weaker finding than the paper's own regime structure, most
plausibly explained by the missing 5 of 7 explanatory variables (de-rated
margin, forecast errors, NONBM, inter delta) documented above: without
fundamentals-driven regressors to distinguish regimes structurally, the
model falls back to distinguishing them by variance alone.

Full parameter estimates, standard errors, and both models' RMSEs:
`results/bunn_2021_comparison.json` (generated by the training run) and
`bunn_train_log.txt` (full statsmodels fit summary).

## Part 2: Ganesh & Bunn (2024), FCNN point and density forecasting

`src/04_ganesh_bunn_fcnn.py`. Two neural networks on the same reduced
feature set (price lags, NIV lag2, calendar):

- **Point forecast**: a fully-connected network with two hidden layers
  (128, 8 units), the paper's reported best point-forecast architecture
  (recalled from the source PDF at moderate confidence, not re-verified a
  second time).
- **Density forecast**: the same architecture trained with pinball loss for
  quantiles 0.01, 0.05, 0.50, 0.95, 0.99, evaluated by empirical interval
  coverage rather than by matching any specific paper pinball-loss number
  (not confidently recalled from the source PDF).

**Point forecast results:**

| Metric | Paper FCNN | This reproduction |
|---|---|---|
| RMSE | 14.5 | 15.12 |
| MAE | 11.4 | 10.16 |
| R2 | 0.66 | 0.48 |

RMSE lands close to the paper's (15.12 vs 14.5). MAE is actually *better*
than the paper's here (10.16 vs 11.4), while R2 is well below it (0.48 vs
0.66), the same pattern of "close typical error, weaker explained variance"
seen in [lucas-2020-reproduction](../lucas-2020-reproduction)'s paper-parity
run. Consistent explanation: a model missing the fundamentals regressors
that explain the *large, structural* price movements (de-rated margin,
forecast errors) can still track typical-sized moves reasonably well
(similar MAE) while doing a worse job explaining the full variance of the
series (lower R2), because the big moves it cannot see coming are exactly
what drives R2 down disproportionately.

Within this reproduction, the FCNN (RMSE 15.12) beats this reproduction's
own regime-switching model (out-of-sample RMSE 16.47), consistent with the
paper's own narrative that neural nets add value over the statistical
benchmark, even though neither individual number matches the paper's
reported figures for either model.

**Density forecast results** (no paper pinball-loss numbers reproduced,
these are the reproduction's own predictions evaluated on their own terms):

| Quantile interval | Target coverage | Empirical coverage |
|---|---|---|
| [5th, 95th] | 90% | 88.1% |
| [1st, 99th] | 98% | 98.2% |

Reasonably well calibrated at both tails, close to nominal coverage in both
cases, though this reproduction cannot say whether that calibration quality
matches the paper's own, since no comparable paper figure was confidently
extracted.

Full results: `results/ganesh_bunn_2024_comparison.json`.

## Reproducing this

```bash
pip install -e ../gb-bm-data
pip install -r requirements.txt
python src/01_fetch_data.py                # ~35-50 min, largest fetch in this data layer's portfolio
python src/02_build_features.py
python src/03_bunn_regime_switching.py      # several minutes, Markov-switching MLE
python src/04_ganesh_bunn_fcnn.py
```

## Author

Karan Homayounfar ([KNHNF](https://github.com/KNHNF)). Part of a small set
of independent GB balancing-market forecasting-paper reproductions built on
[gb-bm-data](../gb-bm-data), alongside a separate MSc dissertation on GB
balancing market cost forecasting.

## Licence

MIT, see [LICENSE](LICENSE).
