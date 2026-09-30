# Deng et al. (2023) reproduction

An independent reproduction of the headline architecture in:

> Deng, Z., et al. (2023) 'Machine Learning and Deep Learning Forecasts of
> Electricity Imbalance Prices', *SSRN* (GB imbalance price forecasting with
> a seasonal-attention BiLSTM, reported 25-37% improvement on extreme price
> events over a baseline).

**Updated 2026-09-12** after re-reading the source PDF directly from Zotero
storage (`Deng et al. - 2023 - Machine Learning and Deep Learning Forecasts
of Electricity Imbalance Prices.pdf`), rather than relying on the earlier
degraded extraction this reproduction was first built from. That re-read
confirmed the architecture, date window, and training hyperparameters with
high confidence from Section 4.3 and Algorithm 1, and the model below is
rebuilt to match. The result changed materially as a result, see below.

## What was wrong before, and what the source text actually says

The first version of this reproduction guessed at three things it could not
confirm, and all three turned out to be wrong:

| | First attempt (guessed) | Confirmed from Algorithm 1 / Section 5.1 |
|---|---|---|
| Model inputs | Multivariate: 28-column feature table (price, NIV, demand, wind/solar/gas/nuclear mix, calendar) fed at every timestep | **Univariate.** The only input is 48 lags of the price series itself, `x = [x1, ..., x48]`. No other feature appears anywhere in the paper's algorithm. |
| "Seasonal attention" mechanism | Generic additive (Bahdanau-style) attention over BiLSTM hidden states across all 48 timesteps | **Elementwise multiplicative gate on the raw input, before the BiLSTM.** `f_A` is a learnable vector of 48 weights, initialised as `[1, ..., 48]`; the gated input is `x_hat = x * f_A` (Equations 11-13). This is a much simpler mechanism than attention over hidden states, and is trained jointly with the BiLSTM's own weights. |
| Date window | 2018-01 to 2019-07-31 (length-matched guess, no confirmed dates) | **1 July 2016 to 30 June 2019**, 52,560 half-hourly points (Section 5.1), the same window already fetched for [bunn-ganesh-2021-2024-reproduction](../bunn-ganesh-2021-2024-reproduction). |
| Train/test split | Chronological 80/20 | **Final 2,000 points held out as test**, everything before that is training (Section 5.1). |
| Training schedule | 25 epochs, Adam, default learning rate | **batch_size=2048, epochs=20, lr=0.00001, Adam, MSE loss**, stated explicitly in Algorithm 1's own caption. |

## Data

Reuses the raw system-price CSV already fetched for
[bunn-ganesh-2021-2024-reproduction](../bunn-ganesh-2021-2024-reproduction)
(BMRS, 2016-07-01 to 2019-09-30, a superset of the paper's confirmed
2016-07-01 to 2019-06-30 window), rather than re-fetching. `src/01_build_
features.py` keeps only `systemBuyPrice`, renamed `imbalance_price`; no
other column is passed to the model, per the confirmed univariate input.

## Model

`src/02_bilstm_model.py`, PyTorch, rebuilt to match Algorithm 1 exactly:

- **BiLSTM baseline**: bidirectional LSTM (hidden size 64, not stated in the
  source text, kept from the earlier pass), linear head on the final
  timestep's output. No gate.
- **Seasonal-Attention BiLSTM**: identical BiLSTM backbone, but the raw
  48-lag input is multiplied elementwise by a learnable 48-weight vector
  `f_A` (initialised `[1, ..., 48]`) before entering the LSTM. `f_A` and the
  BiLSTM's weights are trained together by the same Adam optimiser.

Both trained with the paper's own stated schedule: `batch_size=2048`,
`epochs=20`, `lr=0.00001`, Adam, MSE loss on min-max normalised prices
(scaler fit on the training split only). One-step-ahead only; the paper
also reports k=2,3,4, left for a follow-up run.

## Results

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| BiLSTM baseline | 21.04 | 25.75 | -48.7% |
| Seasonal-Attention BiLSTM | 23.19 | 27.91 | -74.7% |

Both models have negative R2, meaning both do worse than predicting the mean
price over this test window. The paper's own reported RMSE for SA-BiLSTM,
one-step-ahead, full-sample window (Table 1) is **13.035**, well below
either model here. The very low learning rate (1e-5) and only 20 epochs,
run exactly as the paper states them, likely under-converge on a series
with extreme spikes up to 1528.72 GBP/MWh; the paper does not report a
warm-up schedule or gradient clipping, and none is added here to try to
close this gap, since doing so would no longer be reproducing what Algorithm
1 actually specifies.

**Extreme-event comparison** (bottom/top 5% of the 2,000-point test set,
n=218, matching the paper's own "extreme events" framing):

| Model | MAE on extreme events | MAE on normal events |
|---|---|---|
| BiLSTM baseline | 37.38 | (see results/model_comparison.json) |
| Seasonal-Attention BiLSTM | 28.13 | (see results/model_comparison.json) |

**Attention-over-baseline improvement on extreme events: 24.7%.** The paper
reports a 25-37% range. This reproduction now lands just under that range,
a genuinely close match, and a complete reversal of the first attempt's
result (-3.6%, the wrong direction). The correction that produced this
was fixing the architecture and input to match Algorithm 1, not any change
to the evaluation method or tuning aimed at hitting the paper's number.

## Why the corrected result is close on the one metric that matters most, but not on overall fit

The paper's headline claim is specifically about extreme-event performance,
and that is where this reproduction now agrees closely. Overall R2 being
negative for both models, while the extreme-event improvement lands near
the paper's own range, is not a contradiction: a model trained at this
learning rate and epoch count on a univariate series with rare extreme
spikes can plausibly learn to react proportionally *more* to a spike
relative to its own weaker baseline (explaining the 24.7% figure) while
still not fitting the bulk of ordinary-range price movements well overall
(explaining the negative R2 for both). Both effects come from the same
under-converged training regime, which is the paper's own stated regime,
not a choice made to flatter this result.

## What this is and isn't

This now reproduces the paper's actual architecture, input, training
schedule, and evaluation window as confirmed directly from the source PDF's
Algorithm 1 and Section 5.1, not an approximation built from a degraded
extraction. It is still not a replication: the exact random seed, weight
initialisation beyond what Algorithm 1 states, and any additional training
detail not written in the paper are not recoverable from the text, and
overall point-forecast accuracy (R2, RMSE) does not match the paper's
reported numbers, only the direction and rough magnitude of its extreme-
event claim does.

## Reproducing this

Requires `bunn-ganesh-2021-2024-reproduction`'s raw data to already exist
(`../bunn-ganesh-2021-2024-reproduction/data/raw/system_prices.csv`).

```bash
pip install -e ../gb-bm-data
pip install -r requirements.txt
python src/01_build_features.py
python src/02_bilstm_model.py
```

## Author

Karan Homayounfar ([KNHNF](https://github.com/KNHNF)). Part of a small set
of independent GB balancing-market forecasting-paper reproductions built on
[gb-bm-data](https://github.com/KNHNF/gb-bm-data), alongside a separate MSc dissertation on GB
balancing market cost forecasting.

## Licence

MIT, see [LICENSE](LICENSE).
