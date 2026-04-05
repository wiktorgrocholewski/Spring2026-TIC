================================================================================
  Corn-Soybean Spread Trading Strategy
  Trading and Investment Club — Spring 2026
================================================================================

OVERVIEW
--------
This project implements a machine learning pipeline for trading the corn-soybean
spread, following the methodology of Marcos López de Prado's "Advances in
Financial Machine Learning" (2018), hereafter AFML. The strategy uses weather
data and Kalman filter estimates to predict whether a spread trade entered on a
given day will be profitable, using triple barrier labeling with frozen hedge
ratios.

Data: Teucrium CORN and SOYB ETFs (2011–2026), plus daily weather data from
6 locations (3 US Corn Belt, 3 Brazilian soy regions).
Models: XGBoost (primary), Random Forest with sequential bootstrap (secondary).


PROJECT STRUCTURE
-----------------
Spring2026-TIC/
├── data/
│   ├── corn.csv                   Raw CORN ETF prices from Yahoo Finance
│   ├── soybean.csv                Raw SOYB ETF prices from Yahoo Finance
│   ├── weather_raw.csv            Raw daily weather from Open-Meteo
│   ├── weather_rolled.csv         30-day rolling weather aggregates
│   └── full_dataset.csv           Merged price + weather (input to notebooks)
│
├── notebooks/
│   ├── 1_gettingdata.ipynb        Data fetching and preprocessing
│   └── 2_modelling.ipynb          Full ML pipeline: features → labels → CV → models
│
├── src/
│   ├── data.py                    Data fetching and merging functions
│   ├── features.py                Feature engineering (hedge ratios, Kalman, weather)
│   ├── labels.py                  Triple barrier labeling + sample weights
│   ├── cv.py                      Cross-validation, scoring, feature importance
│   └── backtest.py                Backtesting (to be implemented)
│
└── README.txt                     This file


NOTEBOOK WORKFLOW
-----------------
1_gettingdata.ipynb:
    Fetches Yahoo price data and Open-Meteo weather data, computes rolling
    weather aggregates, merges everything into full_dataset.csv.

2_modelling.ipynb (run cells in order):
    Cell  1: Imports
    Cell  2: Compute features (OLS hedge ratio, Kalman filter, calendar, vol)
    Cell  4: Triple barrier labeling + sample uniqueness weights
    Cell  6: Assemble full 97-feature matrix (exclude raw prices, intermediates)
    Cell  8: Dev/holdout split (375 trading days ≈ 1.5 years held out)
    Cell 10: Baseline XGBoost (6 features)
    Cell 12: Full feature set XGBoost (97 features)
    Cell 14: MDA feature importance (permutation, out-of-sample)
    Cell 16: MDI feature importance (impurity, in-sample)
    Cell 18: Feature selection (union of top 20s, MDA-negative veto, force Kalman)
    Cell 20: Retrain XGBoost with 24 selected features
    Cell 22: Random Forest with sequential bootstrap
    Cell 24: XGBoost hyperparameter tuning (81 grid search combinations)
    Cell 26: Pipeline hyperparameter tuning (27 combinations: delta, num_days, pt_sl)


MODULE REFERENCE
================


data.py — Data Fetching and Preprocessing
------------------------------------------
No direct AFML reference. ETF data used instead of futures to avoid roll gap
artifacts (see AFML Section 2.4, "Dealing with Multi-Product Series").

  fetch_yahoo_data(ticker, name, save_path, date_start=None, date_end=None)
      Download OHLCV data from Yahoo Finance via yfinance. Cleans column names
      (lowercase, appends asset name), saves to save_path/data/{name}.csv.

  merge_yahoo_datasets(names, merged_name, save_path)
      Inner-join multiple Yahoo datasets on date index. Keeps only dates
      present in all datasets.

  fetch_weather_data(locations, start_date, end_date, variables, save_path)
      Fetch daily weather from Open-Meteo Archive API for multiple locations.
      Default: 15 agronomically relevant variables (temperature, precipitation,
      soil moisture, evapotranspiration, humidity, VPD, wind, snowfall).
      Handles timezone normalization so all locations align on the same dates.
      Locations: central Iowa, central Illinois, central Indiana (US Corn Belt),
      Mato Grosso, Paraná, Rio Grande do Sul (Brazil soy regions).

  compute_rolling_weather(weather_path, window=30, save_path=None)
      30-day backward-looking rolling aggregates with variable-appropriate
      aggregation: means for temperature/humidity, sums for precipitation/ET,
      minima for soil moisture (worst-case stress), maxima for wind speed.

  merge_price_weather(price_path, weather_path, save_path=None)
      Merge price data (trading days) with weather data (calendar days).
      Forward-fills weather to trading days (weekends/holidays get Friday's
      weather, which is what you'd know in real time).


features.py — Feature Engineering
-----------------------------------

  compute_hedge_ratio(df, corn_col, soy_col, window=504)
      Rolling OLS: soy = alpha + gamma * corn + epsilon.
      504-day window ≈ 2 trading years. Adds columns: hedge_ratio, intercept,
      spread (= OLS residual).
      Based on: Chan (2013), Ch. 2–3 (cointegration and mean reversion).
      Validity of OLS despite autocorrelated residuals justified by
      Engle-Granger (1987) superconsistency (OLS converges at rate 1/T
      for cointegrated series).

  compute_kalman_hedge(df, corn_col, soy_col, delta=1e-5, R=1e-6,
                       burn=200, max_iter=10, tol=1e-5)
      3-state Kalman filter with mean-reverting spread dynamics.
      State = [alpha, gamma, spread]. Alpha and gamma are random walks;
      spread is AR(1) with persistence phi. phi and sigma2_s estimated
      iteratively (typically converges in 8 iterations to phi ≈ 0.996).
      delta is the KEY hyperparameter (tune via CV).
      Adds columns: kf_hedge_ratio, kf_intercept, kf_spread, kf_innovation,
      kf_z_score (innovation z), kf_level_z (spread z).
      Based on: 2-state Kalman filter from Palomar (2012) and Chan (2013,
      Ch. 2), extended to 3 states to prevent gamma distortion. State-space
      formulation follows Harvey (1989) and Durbin & Koopman (2012).
      The iterative phi/sigma2_s estimation avoids look-ahead bias that
      would arise from full-sample OLS estimation of half-life.

      Internal helper:
        _run_kalman_3state(corn, soy, delta, sigma2_s, R, phi)
            Single pass of the Kalman filter with fixed parameters.

  compute_calendar_features(df)
      Adds month (1–12) and day_of_week (0–4). Note: feature importance
      analysis showed month is detrimental and should be excluded.
      The seasonal information is already encoded in weather feature levels
      (temperature, soil moisture, etc.).

  compute_spread_vol(df, corn_col, soy_col, hedge_ratio_col, span=100)
      EWMA volatility of actual P&L: delta_soy - gamma_{t-1} * delta_corn.
      Same calculation as get_daily_vol in labels.py but stored as a feature.
      Based on: AFML Snippet 3.1 (daily volatility), adapted for a
      cointegration spread where percentage returns are undefined.
      Note: feature importance showed this is detrimental and should be excluded.


labels.py — Triple Barrier Labeling and Sample Weights
-------------------------------------------------------

  get_daily_vol(corn, soy, hedge_ratio, span=100)
      EWMA std of actual tradeable P&L (not spread.diff()). Uses hedge_ratio
      shifted by 1 day to avoid look-ahead. Returns pd.Series of daily vol.
      Based on: AFML Snippet 3.1 ("Computing Dynamic Thresholds").
      Adapted for spread: uses first differences instead of percentage returns
      because the spread crosses zero.
      The shift(1) on hedge_ratio avoids the contamination from AFML
      Section 2.4.1 (ETF trick), where rolling hedge ratio drift creates
      artificial P&L.

  get_vertical_barrier(trading_days, num_days=150, t_events=None)
      Vertical barrier at num_days TRADING days (not calendar days). Uses
      positional indexing. t_events defaults to every trading day; pass a
      subset to restrict entries.
      Based on: AFML Snippet 3.4. Modified to count trading days via
      positional indexing instead of pd.Timedelta(days=...), which counts
      calendar days and gives only ~105 trading days for a nominal 150.

  apply_pt_sl_on_t1(corn, soy, hedge_ratio, t1, trgt, pt_sl=[1, 1])
      Core triple barrier function. For each event, FREEZES gamma at entry
      and computes P&L path as:
          pnl[t] = (soy[t] - soy_entry) - gamma_entry * (corn[t] - corn_entry)
      Records first time P&L crosses upper (pt) or lower (sl) barrier.
      Returns DataFrame with columns [pt, sl, t1] of barrier touch timestamps.
      Based on: AFML Snippet 3.2 ("The Triple-Barrier Method"), Section 3.4.
      The frozen gamma is our adaptation for cointegration spreads—AFML's
      original assumes a single-asset return, not a spread with a drifting
      hedge ratio.

  get_bins(touches, corn, soy, hedge_ratio)
      Finds earliest barrier touch, computes frozen-gamma P&L at that time,
      labels by sign. Returns DataFrame with t1 (first touch time, needed for
      Ch.4 weights), ret (P&L), bin ({-1, 1}).
      Based on: AFML Snippets 3.5 / 3.7 ("Learning Side and Size").
      Separated from apply_pt_sl_on_t1 because (a) Chapter 4 needs the touch
      timestamps, and (b) meta-labeling would reinterpret the same touches
      differently (Section 3.6).

  get_avg_uniqueness(labels, trading_days)
      Chapter 4 sample weights. For each event, averages 1/concurrency over
      its lifespan. Pass to XGBoost as sample_weight.
      Based on: AFML Snippets 4.1–4.2 ("Average Uniqueness of a Label").
      Concurrency c_t = number of events whose [entry, t1] spans day t.
      Uniqueness u_{t,i} = 1/c_t. Average uniqueness is mean u over the
      event's lifespan.

  get_ind_matrix(trading_days, t1)
      Build binary indicator matrix. Entry (t, i) = 1 if bar t falls within
      event i's lifespan. Helper for sequential bootstrap; not called
      directly in the pipeline.
      Based on: AFML Snippet 4.3 ("Build an Indicator Matrix").

  seq_bootstrap(t1, trading_days, s_length=None, random_state=None)
      Sequential bootstrap. Draws with replacement where probability is
      proportional to average uniqueness if that observation were added to
      the current bag. Uses numba JIT if available (~0.1s per bag), otherwise
      falls back to pure numpy (~2s per bag).
      Based on: AFML Snippet 4.5 ("Return Sample from Sequential Bootstrap"),
      Section 4.5.1. The algorithm maintains a running concurrency array
      and recomputes uniqueness probabilities after each draw, producing
      bags closer to IID than standard bootstrap (verified by Monte Carlo
      in AFML Section 4.5.4).


cv.py — Cross-Validation, Scoring, and Feature Importance
-----------------------------------------------------------

  temporal_split(X, y, t1, sample_weight=None, n_holdout=375)
      Split into dev set and final holdout. Holdout is last n_holdout
      observations. Returns dict with X_dev, y_dev, t1_dev, w_dev,
      X_holdout, y_holdout, t1_holdout, w_holdout, cutoff_date.
      Motivation: AFML Section 11.4 ("Backtesting Is Not a Research Tool")
      and the multiple comparisons problem. Holdout is touched once after
      all development decisions are finalized.

  class WalkForwardPurgedCV(n_periods=3, t1=None)   [PRIMARY]
      Expanding-window walk-forward CV. For test period j, trains on periods
      1..j-1 only. Purges training observations whose labels leak into test.
      No embargo needed (never trains on future data). k=3 → 2 test folds.
      Based on: AFML Ch. 7 (purged CV concept), but uses walk-forward instead
      of symmetric k-fold. This is our modification to prevent feature leakage
      from sequentially estimated features (rolling OLS, Kalman filter), which
      AFML's symmetric purged k-fold does not address. See also Dixon et al.
      (2020), Section 4.2 on time series cross-validation.
      Methods:
        .get_n_splits()     Returns n_periods - 1
        .split(X)           Yields (train_idx, test_idx) arrays

  class PurgedKFold(n_splits=5, t1=None, pct_embargo=0.01)   [SECONDARY]
      Symmetric purged k-fold. WARNING: trains on future data.
      Not recommended for pipelines with sequentially estimated features.
      Kept for reference/comparison only.
      Based on: AFML Snippet 7.3 ("The Purged K-Fold Class"), Section 7.4.
      Purging: AFML Section 7.4.1. Embargo: AFML Section 7.4.2.
      Methods:
        .get_n_splits()     Returns n_splits
        .split(X)           Yields (train_idx, test_idx) arrays

  cv_score(model, X, y, t1, sample_weight=None, cv=None, scoring='accuracy')
      Run CV and return per-fold train/test scores with diagnostics.
      Prints a formatted table of fold sizes, dates, purged counts, and scores.
      scoring: 'accuracy' or 'neg_log_loss'.
      Returns dict with test_scores, train_scores, fold_sizes.
      Based on: AFML Ch. 7 cvScore function concept. Uses neg_log_loss as
      recommended in AFML Section 9.4 ("Scoring and Hyper-parameter Tuning")
      because log-loss evaluates predicted probability quality, which matters
      for bet sizing.

  feat_importance_mda(model, X, y, t1, sample_weight=None, cv=None,
                      scoring='neg_log_loss', n_repeats=1)
      MDA feature importance. Permutation-based, OOS.
      For each CV fold, shuffles each feature and measures performance drop.
      Positive = useful, negative = detrimental. Uses walk-forward CV by
      default. Returns DataFrame with 'mean' and 'std' columns, sorted
      by importance descending.
      Based on: AFML Snippet 8.3 ("MDA Feature Importance"), Section 8.3.2.
      Key properties from AFML: (1) can be applied to any classifier;
      (2) can conclude all features are unimportant; (3) susceptible to
      substitution effects with correlated features; (4) can identify
      features that are actively detrimental (negative importance).

  feat_importance_mdi(X, y, sample_weight=None, n_estimators=500,
                      random_state=42)
      MDI feature importance. In-sample, tree-based.
      Uses RF with max_features=1 (avoid masking), replaces 0 with NaN.
      Normalizes so importances sum to 1. Returns DataFrame with 'mean'
      and 'std' columns, sorted descending.
      Based on: AFML Snippet 8.2 ("MDI Feature Importance"), Section 8.3.1.
      max_features=1 follows AFML's recommendation to avoid masking effects
      (point 1 in Section 8.3.1). Replacing 0 with NaN follows point 1(b).
      WARNING (AFML point 2): MDI is in-sample—every feature will have some
      importance even if it has no predictive power. Always cross-check
      with MDA.


KEY DESIGN DECISIONS (with references)
---------------------------------------

1. Walk-forward CV over symmetric purged k-fold
   Problem: sequential feature estimates (rolling OLS window, Kalman filter)
   computed on future data leak into training features.
   AFML Ch. 7 addresses label leakage via purging, but not feature leakage
   from sequential estimation. Walk-forward eliminates this by only training
   on past data.
   See also: Dixon et al. (2020), Fig. 6.4.

2. Frozen gamma in labeling
   Problem: AFML Section 2.4.1 (ETF trick) shows that rolling hedge ratio
   drift creates ~6% artificial P&L. Our decomposition of spread.diff()
   into actual P&L + rebalancing artifact motivates freezing gamma at
   entry for both volatility estimation and barrier checking.

3. 3-state Kalman over 2-state
   Problem: 2-state Kalman (Palomar 2012, Chan 2013) treats the spread as
   iid noise. Cointegration implies AR(1) spread, so temporary deviations
   cause gamma distortion. Adding spread as a 3rd hidden state with AR(1)
   dynamics resolves this.
   delta=1e-4 causes model collapse to 2-state (phi → 1, half-life > 4000
   days), confirmed by worst CV performance across all 27 pipeline configs.
   State-space references: Harvey (1989), Durbin & Koopman (2012).

4. Feature selection via MDI∪MDA with MDA-negative veto + force-include
   AFML Ch. 8 recommends MDI and MDA as complementary methods. We take the
   union of both top 20s, veto features with negative MDA (actively harmful),
   and force-include kf_spread and kf_innovation for economic coherence
   (a spread strategy needs spread features). This removes 73 of 97 features.
   The veto catches kf_level_z (MDI rank 1–2, MDA negative): the model's
   favorite in-sample splitting variable that doesn't generalize.

5. Sequential bootstrap for Random Forest (AFML Section 4.5)
   Mean uniqueness ≈ 0.088 → standard bootstrap produces near-identical
   trees (AFML Section 6.3.3, "Observation Redundancy"). Sequential
   bootstrap (Snippet 4.5) draws proportional to uniqueness, producing
   bags closer to IID. Bag size set to avg_uniqueness × n_events, following
   AFML's max_samples recommendation in Section 4.5.

6. 1.5-year holdout
   AFML Section 11.3 ("Even If Your Backtest Is Flawless, It Is Probably
   Wrong") and Section 11.4 warn about selection bias from repeated testing.
   Holdout is touched once after all development decisions are finalized.

7. Neg log loss as scoring function for hyperparameter tuning
   AFML Section 9.4: "That is the right ML performance metric for
   hyper-parameter tuning of financial applications, not accuracy."
   Log-loss evaluates predicted probability quality, which matters when
   probabilities are used for position sizing (AFML Ch. 10).

8. No CUSUM filter
   AFML Section 2.5.2 proposes CUSUM for event-based sampling. With only
   ~3000 valid observations and 150-day barriers, we cannot afford to
   discard events. Overlap is handled by sample weights (Ch. 4) instead.

9. Trading days instead of calendar days for vertical barrier
   AFML Snippet 3.4 uses pd.Timedelta(days=...) which counts calendar days.
   For daily bars, this gives ~105 trading days for a nominal 150-day window.
   We use positional indexing to count actual trading days.


DEPENDENCIES
------------
Python 3.11+
pandas, numpy, scikit-learn, xgboost, yfinance
openmeteo-requests, requests-cache, retry-requests  (for weather data only)
numba (optional, speeds up sequential bootstrap ~20x)


REFERENCES
----------
López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
    - Ch. 2: Financial data structures, ETF trick (Section 2.4.1)
    - Ch. 3: Triple barrier labeling (Snippets 3.1–3.7)
    - Ch. 4: Sample weights, uniqueness, sequential bootstrap (Snippets 4.1–4.5)
    - Ch. 6: Ensemble methods, RF overfitting under redundancy (Section 6.3.3)
    - Ch. 7: Purged k-fold CV (Snippets 7.1–7.3)
    - Ch. 8: Feature importance—MDI (Snippet 8.2), MDA (Snippet 8.3)
    - Ch. 9: Hyperparameter tuning, grid search CV (Snippets 9.1, 9.3)
    - Ch. 11: Dangers of backtesting, selection bias

Chan, E. P. (2013). Algorithmic Trading. Wiley.
    - Ch. 2–3: Cointegration, mean reversion, half-life estimation
    - 2-state Kalman filter for hedge ratio estimation

Palomar, D. P. (2012). Pairs Trading lecture slides, HKUST.
    - 2-state Kalman filter formulation for pairs trading

Dixon, M. F., Halperin, I., & Bilokon, P. (2020). Machine Learning in
    Finance: From Theory to Practice. Springer.
    - Section 4.2: Time series cross-validation / walk-forward optimization

Harvey, A. C. (1989). Forecasting, Structural Time Series Models and the
    Kalman Filter. Cambridge University Press.
    - General state-space model formulation

Durbin, J. & Koopman, S. J. (2012). Time Series Analysis by State Space
    Methods (2nd ed.). Oxford University Press.
    - Kalman filter implementation details

Engle, R. F. & Granger, C. W. J. (1987). "Co-integration and Error
    Correction: Representation, Estimation, and Testing." Econometrica,
    55(2), 251–276.
    - Superconsistency of OLS for cointegrated series
