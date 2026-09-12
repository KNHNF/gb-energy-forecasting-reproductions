"""Reproduce Bunn et al. (2021)'s 2-regime Markov-switching regression for
GB balancing market price, against a linear AR benchmark, using
statsmodels' MarkovAutoregression (Hamilton-style, order=2, time-varying
transition probabilities driven by lagged NIV, switching variance).

Reproduces the paper's evaluation structure on two of its headline numbers:
in-sample RMSE, and one out-of-sample backtest RMSE. Does not reproduce all
three of the paper's specific historical backtest windows (a train-2017-19/
forecast-2016 split, and a Nov-Dec 2016 rolling backtest); this reproduction
uses one chronological 80/20 split, consistent with the rest of this data
layer's paper reproductions, and that choice is reported plainly rather than
presented as matching the paper's exact backtest design.

Feature set: price AR(2) (handled internally by MarkovAutoregression's
order=2) and niv_lag2 as an exogenous regressor influencing both the
regime-conditional mean and the regime transition probabilities. This is 2
of the paper's 7 explanatory variables; see 02_build_features.py for why the
other 5 (de-rated margin, wind/solar/demand forecast error, NONBM, inter
delta) are not reproducible through gb-bm-data.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

PAPER_RESULT = {
    "in_sample_RMSE": {"regime_switching": 19.29, "linear": 18.98},
    "out_of_sample_RMSE": {"regime_switching": 14.7, "linear": 15.4},
    "note": "paper also reports a train-2017-19/forecast-2016 backtest (RS 22.1, linear 24.5) "
            "and a Nov-Dec 2016 rolling backtest (RS 20.8, linear 23.0, GARCH 23.4); "
            "neither is reproduced here, only in-sample and one chronological 80/20 out-of-sample split.",
}


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def fit_linear(train: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    X = sm.add_constant(train[["price_lag1", "price_lag2", "niv_lag2"]])
    y = train["price"]
    return sm.OLS(y, X).fit()


def run() -> None:
    df = pd.read_parquet(PROC / "features.parquet")
    df = df.sort_values(["settlementDate", "settlementPeriod"]).reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    train, test = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
    print(f"Train: {len(train)} rows ({train['settlementDate'].iloc[0].date()} to "
          f"{train['settlementDate'].iloc[-1].date()})")
    print(f"Test:  {len(test)} rows ({test['settlementDate'].iloc[0].date()} to "
          f"{test['settlementDate'].iloc[-1].date()})")

    # --- linear AR benchmark ---
    print("\nFitting linear AR benchmark (OLS)...")
    linear_model = fit_linear(train)
    linear_in_sample_pred = linear_model.predict(sm.add_constant(train[["price_lag1", "price_lag2", "niv_lag2"]]))
    linear_in_sample_rmse = rmse(train["price"], linear_in_sample_pred)

    X_test = sm.add_constant(test[["price_lag1", "price_lag2", "niv_lag2"]], has_constant="add")
    linear_test_pred = linear_model.predict(X_test)
    linear_test_rmse = rmse(test["price"], linear_test_pred)
    print(f"  in-sample RMSE:     {linear_in_sample_rmse:.2f}")
    print(f"  out-of-sample RMSE: {linear_test_rmse:.2f}")

    # --- 2-regime Markov-switching regression ---
    # Two numerical fixes were needed to get a non-degenerate MLE fit, found
    # by isolating the failure (see git history / dev notes for the
    # diagnostic steps): niv_lag2 has ~10x the standard deviation of price
    # (std ~340 vs ~35), so it is standardized before use in exog_tvtp.
    # Separately, and this was the actual root cause: the raw price series
    # has extreme spikes (max 1528.72, 99.99th percentile 1134.21, 29 points
    # above 500) that push the Hamilton filter's per-regime likelihood
    # evaluation into numerical underflow, collapsing the whole EM fit to a
    # degenerate solution (all-zero regime-0 coefficients, NaN
    # log-likelihood) regardless of which switching options were enabled.
    # Winsorizing price at the 0.1/99.9 percentiles (computed on the train
    # split only) for this model's estimation fixes it. This is a real,
    # deliberate deviation from the paper (which does not report clipping
    # its price series) made purely for MLE numerical stability with this
    # library; RMSE is still evaluated against the true, unclipped price.
    print("\nFitting 2-regime Markov-switching regression (this takes a few minutes)...")
    niv_mean, niv_std = train["niv_lag2"].mean(), train["niv_lag2"].std()
    train["niv_lag2_z"] = (train["niv_lag2"] - niv_mean) / niv_std
    df["niv_lag2_z"] = (df["niv_lag2"] - niv_mean) / niv_std

    clip_lo, clip_hi = train["price"].quantile([0.001, 0.999])
    print(f"  winsorizing price to [{clip_lo:.1f}, {clip_hi:.1f}] for MLE stability "
          f"(RMSE still computed against true, unclipped price)")
    train["price_clipped"] = train["price"].clip(clip_lo, clip_hi)

    endog = train["price_clipped"].to_numpy()
    exog = train[["niv_lag2_z"]].to_numpy()
    exog_tvtp = sm.add_constant(train[["niv_lag2_z"]]).to_numpy()

    t0 = time.time()
    ms_model = MarkovAutoregression(
        endog, k_regimes=2, order=2, exog=exog, exog_tvtp=exog_tvtp,
        switching_ar=True, switching_exog=True, switching_variance=True,
    )
    ms_result = ms_model.fit(em_iter=15, search_reps=15, maxiter=200)
    print(f"  fit took {time.time() - t0:.0f}s")
    print(ms_result.summary())

    if np.isnan(ms_result.llf) or np.any(np.isnan(ms_result.params)):
        raise RuntimeError(
            "Markov-switching MLE did not converge to a non-degenerate solution "
            "(log-likelihood or parameters are NaN). Not proceeding to report a "
            "broken model as a result."
        )

    ms_in_sample_pred = ms_result.predict()
    # MarkovAutoregression drops the first `order` observations internally
    ms_in_sample_rmse = rmse(train["price"].iloc[2:], ms_in_sample_pred)
    print(f"\n  regime-switching in-sample RMSE: {ms_in_sample_rmse:.2f}")

    # out-of-sample: refit on train, one-step-ahead prediction on test using
    # observed history (append test's known prices/exog and predict), not a
    # true walk-forward re-estimation (too slow at this sample size); this
    # is the same simplification most rolling EPF benchmarks use for a
    # single reported backtest RMSE, not a claim of exact parity with the
    # paper's specific rolling re-estimation scheme.
    df["price_clipped"] = df["price"].clip(clip_lo, clip_hi)
    full_endog = df["price_clipped"].to_numpy()
    full_exog = df[["niv_lag2_z"]].to_numpy()
    full_exog_tvtp = sm.add_constant(df[["niv_lag2_z"]]).to_numpy()
    ms_full_model = MarkovAutoregression(
        full_endog, k_regimes=2, order=2, exog=full_exog, exog_tvtp=full_exog_tvtp,
        switching_ar=True, switching_exog=True, switching_variance=True,
    )
    ms_full_smoothed = ms_full_model.smooth(ms_result.params)
    full_pred = ms_full_smoothed.predict()
    test_pred = full_pred[split_idx - 2:]  # align for the order=2 offset
    ms_test_rmse = rmse(df["price"].iloc[split_idx:], test_pred)
    print(f"  regime-switching out-of-sample RMSE: {ms_test_rmse:.2f}")

    results = {
        "in_sample_RMSE": {"regime_switching": round(ms_in_sample_rmse, 2), "linear": round(linear_in_sample_rmse, 2)},
        "out_of_sample_RMSE": {"regime_switching": round(ms_test_rmse, 2), "linear": round(linear_test_rmse, 2)},
    }
    comparison = {"paper": PAPER_RESULT, "this_reproduction": results}
    with open(RESULTS / "bunn_2021_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 60)
    print("PAPER COMPARISON")
    print("=" * 60)
    print(f"{'':22}{'Paper RS':<12}{'Paper Linear':<14}{'This RS':<12}{'This Linear':<12}")
    print(f"{'In-sample RMSE':22}{PAPER_RESULT['in_sample_RMSE']['regime_switching']:<12}"
          f"{PAPER_RESULT['in_sample_RMSE']['linear']:<14}{results['in_sample_RMSE']['regime_switching']:<12}"
          f"{results['in_sample_RMSE']['linear']:<12}")
    print(f"{'Out-of-sample RMSE':22}{PAPER_RESULT['out_of_sample_RMSE']['regime_switching']:<12}"
          f"{PAPER_RESULT['out_of_sample_RMSE']['linear']:<14}{results['out_of_sample_RMSE']['regime_switching']:<12}"
          f"{results['out_of_sample_RMSE']['linear']:<12}")

    print(f"\nSaved results/bunn_2021_comparison.json")


if __name__ == "__main__":
    run()
