# Deng et al. (2023) reproduction

An independent reproduction attempt of the headline architecture in:

> Deng, Z., et al. (2023) 'Machine Learning and Deep Learning Forecasts of
> Electricity Imbalance Prices', *SSRN* (dissertation lit-review entry, GB
> imbalance price forecasting with a seasonal-attention BiLSTM, reported
> 25-37% improvement on extreme price events over a baseline).

**Confidence caveat, stated upfront:** the exact architecture, feature list,
and date range in Deng et al. (2023) could not be confirmed with high
confidence from the source PDF extraction used for this project. Where the
paper's own detail was uncertain, this reproduction targets the paper's
*headline methodological claim* (a seasonal-attention BiLSTM beats a plain
BiLSTM baseline, especially on extreme price events) on a defensible GB
balancing-market feature set, rather than claiming a feature-for-feature or
hyperparameter-for-hyperparameter match this project cannot verify. This is
a weaker reproduction than [lucas-2020-reproduction](../lucas-2020-reproduction)
for that reason, and is presented as such.

## Data

No fresh fetch. Reuses the raw CSVs already pulled by
`../lucas-2020-reproduction/src/01b_fetch_data_paper_window.py` (BMRS v2 +
Carbon Intensity API, 2018-01-01 to 2019-07-31, via the same `gb-bm-data`
client). Re-fetching an overlapping window here would be the same
rate-limited ~20-40 min pull again for no new data. This window's length was
chosen to be comparable to Lucas et al. (2020)'s ~19 months, not to match
Deng et al. (2023)'s own sample, whose exact dates are one of the details
not confirmed from the source PDF.

Feature table: 19,853 rows x 28 columns, built by `src/01_build_features.py`,
reusing the target and feature definitions from `lucas-2020-reproduction`
(target renamed `imbalance_price` = systemBuyPrice, NIV, demand, wind/solar/
gas/nuclear mix, calendar, short lags).

## Model

`src/02_bilstm_model.py`, PyTorch. Two models, same lookback window (48
settlement periods, 24 hours), same chronological 80/20 split, same feature
set:

- **BiLSTM baseline**: bidirectional LSTM, hidden size 64, linear head on
  the final timestep's output.
- **Seasonal-Attention BiLSTM**: same BiLSTM backbone, plus an additive
  (Bahdanau-style) attention layer over every timestep's output before the
  linear head. "Seasonal" here means the calendar features (hour,
  day_of_week, month, season) are part of the input vector at every
  timestep, not a separate seasonal sub-model, since the paper's own
  seasonal-conditioning mechanism was not confirmed from the extracted text.

Both trained 25 epochs, Adam, MSE loss, identical to each other except for
the attention layer, so any difference in results is attributable to
attention, not to incidental differences in training setup.

## Results

| Model | MAE | MSE | R2 |
|---|---|---|---|
| BiLSTM baseline | 11.67 | 297.08 | 37.5% |
| Seasonal-Attention BiLSTM | 11.89 | 311.81 | 34.4% |

**Extreme-event comparison** (bottom/top 5% of test-period true prices,
n=403 of 3,971 test points, matching the paper's own "extreme events"
framing):

| Model | MAE on extreme events | MAE on normal events |
|---|---|---|
| BiLSTM baseline | 25.44 | 10.12 |
| Seasonal-Attention BiLSTM | 26.36 | 10.26 |

Attention-over-baseline improvement on extreme events: **-3.6%**. The paper
reports a 25-37% improvement. This reproduction's attention model does not
beat its own baseline, on either the overall metrics or the extreme-event
subset specifically, and no tuning was applied to try to reach the paper's
reported range once this result came back. This is reported as run.

## Why the result diverges

Three real, honest candidates, none mutually exclusive:

1. **Uncertain architecture match.** Since the paper's exact seasonal-
   conditioning mechanism, attention formulation, and hyperparameters were
   not confirmed from the source PDF, the attention layer implemented here
   may simply not be the same mechanism the paper describes. A generic
   additive attention layer over a short 48-step window may add capacity
   without adding useful signal on this particular feature set.
2. **Feature set uncertainty.** As in the LOLP/de-rated-margin gaps
   documented for [lucas-2020-reproduction](../lucas-2020-reproduction),
   `gb-bm-data` cannot supply the same fundamentals-heavy feature set Bunn
   et al. (2021) and (by inheritance) likely Deng et al. (2023) use. An
   attention mechanism over a thinner, less fundamentals-driven feature set
   has less to selectively attend to.
3. **Small, short-window test set.** 403 extreme-event points in the test
   set is a small sample; a few percentage points of MAE difference either
   way is within noise at this scale, not necessarily a stable finding.

## What this is and isn't

This reproduces the paper's headline comparison structure (attention BiLSTM
vs plain BiLSTM, evaluated with special attention to extreme price events)
on real GB balancing-market data. It is not a replication: architecture
details, feature list, and date range are all approximations where the
source PDF's extracted detail was not confident enough to build from
directly, and the result (attention underperforming its own baseline) is
the opposite direction from the paper's claim. That divergence, and the
reasons for the lower confidence in this reproduction relative to the
others in this data layer's portfolio, are reported rather than hidden.

## Reproducing this

Requires `lucas-2020-reproduction`'s paper-window data to already exist
(`../lucas-2020-reproduction/data/raw_paper_window/`).

```bash
pip install -e ../gb-bm-data
pip install -r requirements.txt
python src/01_build_features.py
python src/02_bilstm_model.py
```

## Author

Karan Homayounfar ([KNHNF](https://github.com/KNHNF)). Part of a small set
of independent GB balancing-market forecasting-paper reproductions built on
[gb-bm-data](../gb-bm-data), alongside a separate MSc dissertation on GB
balancing market cost forecasting.

## Licence

MIT, see [LICENSE](LICENSE).
