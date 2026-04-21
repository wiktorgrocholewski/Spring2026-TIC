# This file contains functions for creating features for the trading strategy.



#####################
# 1. This function calculates the rolling hedge ratio
#####################

def compute_hedge_ratio(df, corn_col, soy_col, window=504):
    """
    Compute rolling OLS hedge ratio with intercept:
    soy = alpha + gamma * corn + epsilon
    spread = soy - alpha - gamma * corn (= epsilon, the cointegrating residual)
    """
    import numpy as np

    corn = df[corn_col].values
    soy = df[soy_col].values
    n = len(df)

    hedge_ratio = np.full(n, np.nan)
    intercept = np.full(n, np.nan)

    for t in range(window, n):
        x = corn[t - window:t]
        y = soy[t - window:t]
        X = np.column_stack([np.ones(window), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        intercept[t] = beta[0]
        hedge_ratio[t] = beta[1]

    df["hedge_ratio"] = hedge_ratio
    df["intercept"] = intercept
    df["spread"] = df[soy_col] - df["intercept"] - df["hedge_ratio"] * df[corn_col]

    valid = np.isfinite(hedge_ratio).sum()
    print(f"Window:          {window} trading days")
    print(f"Valid rows:      {valid} / {n}")
    print(f"Hedge ratio:     {np.nanmin(hedge_ratio):.4f} to {np.nanmax(hedge_ratio):.4f}")
    print(f"Intercept:       {np.nanmin(intercept):.4f} to {np.nanmax(intercept):.4f}")
    print(f"Spread mean:     {np.nanmean(df['spread'].values):.4f}")

    return df


#####################
# 2. Kalman filter for hedge ratio, spread, and z-score
#####################

def _run_kalman_3state(corn, soy, delta, sigma2_s, R, phi):
    """
    Run the 3-state Kalman filter once with fixed parameters.
    
    Internal helper — called by compute_kalman_hedge during
    iterative estimation. See compute_kalman_hedge for the full
    state-space model documentation.
    
    Returns arrays: alphas, gammas, spread_levels, innovations, inn_variances
    """
    import numpy as np

    n = len(corn)
    F = np.diag([1.0, 1.0, phi])
    Q = np.diag([delta, delta, sigma2_s])
    x = np.array([0.0, 0.5, 0.0])
    P = np.diag([10.0, 10.0, 1.0])

    alphas = np.full(n, np.nan)
    gammas = np.full(n, np.nan)
    spread_levels = np.full(n, np.nan)
    innovations = np.full(n, np.nan)
    inn_variances = np.full(n, np.nan)

    for t in range(n):
        # 1. Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # 2. Innovation
        H_t = np.array([1.0, corn[t], 1.0])
        v_t = soy[t] - H_t @ x_pred
        S_t = H_t @ P_pred @ H_t + R

        # 3. Update
        K_t = P_pred @ H_t / S_t
        x = x_pred + K_t * v_t
        P = P_pred - np.outer(K_t, K_t) * S_t

        # 4. Store
        alphas[t] = x[0]
        gammas[t] = x[1]
        spread_levels[t] = x[2]
        innovations[t] = v_t
        inn_variances[t] = S_t

    return alphas, gammas, spread_levels, innovations, inn_variances


def compute_kalman_hedge(df, corn_col, soy_col, delta=1e-5, R=1e-6,
                         burn=200, max_iter=10, tol=1e-5):
    """
    Estimate hedge ratio, intercept, spread level, and z-scores using a
    3-state Kalman filter with mean-reverting spread dynamics.
    
    phi (AR(1) coefficient) and sigma2_s (spread innovation variance) are
    estimated iteratively: run the filter, estimate phi and sigma2_s from
    the output spread, re-run with updated parameters, repeat until
    convergence. This avoids circular dependence on external half-life
    estimates.
    
    State-space model:
        State:        xi_t = [alpha_t, gamma_t, s_t]         (3 x 1)
        Transition:   xi_t = F @ xi_{t-1} + v_t,             v_t ~ N(0, Q)
                      F = diag(1, 1, phi)
                      Q = diag(delta, delta, sigma2_s)
                      alpha, gamma: random walk
                      s_t: AR(1) with phi < 1 (mean-reverting)
        Measurement:  soy_t = [1, corn_t, 1] @ xi_t + w_t,  w_t ~ N(0, R)
    
    Parameters
    ----------
    df : pd.DataFrame   — must contain corn_col and soy_col columns
    corn_col : str       — column name for corn prices
    soy_col : str        — column name for soybean prices
    delta : float        — process noise for alpha and gamma. This is a
                           HYPERPARAMETER encoding how fast the hedge ratio
                           can drift. Tune via cross-validation. (default 1e-5)
    R : float            — measurement noise. Numerical floor; s_t handles
                           spread dynamics, R only captures iid noise. (default 1e-6)
    burn : int           — burn-in period to skip for parameter estimation
                           (filter needs time to stabilize). (default 200)
    max_iter : int       — max iterations for phi/sigma2_s convergence. (default 10)
    tol : float          — convergence tolerance for phi and sigma2_s. (default 1e-5)
    
    Returns
    -------
    pd.DataFrame — input df with added columns:
        'kf_hedge_ratio'   — Kalman-filtered gamma_t
        'kf_intercept'     — Kalman-filtered alpha_t
        'kf_spread'        — Kalman-filtered spread level s_t
        'kf_innovation'    — prediction error v_t (new information each day)
        'kf_z_score'       — innovation / sqrt(innovation variance)
        'kf_level_z'       — s_t / expanding std of s_t (spread distance from 0)
    """
    import numpy as np
    import pandas as pd

    corn = df[corn_col].values
    soy = df[soy_col].values

    # --- Initial guesses for iteratively estimated parameters ---
    phi = 0.992
    sigma2_s = 0.04

    # --- Iterative estimation of phi and sigma2_s ---
    for iteration in range(max_iter):
        alphas, gammas, spread_levels, innovations, inn_variances = \
            _run_kalman_3state(corn, soy, delta, sigma2_s, R, phi)

        # Estimate phi: AR(1) coefficient of the Kalman spread
        s = spread_levels[burn:]
        phi_new = np.corrcoef(s[1:], s[:-1])[0, 1]

        # Estimate sigma2_s: variance of spread innovations
        spread_inno = s[1:] - phi_new * s[:-1]
        sigma2_s_new = np.var(spread_inno)

        # Check convergence
        if abs(phi_new - phi) < tol and abs(sigma2_s_new - sigma2_s) < tol:
            phi = phi_new
            sigma2_s = sigma2_s_new
            break

        phi = phi_new
        sigma2_s = sigma2_s_new

    # --- Final run with converged parameters ---
    alphas, gammas, spread_levels, innovations, inn_variances = \
        _run_kalman_3state(corn, soy, delta, sigma2_s, R, phi)

    # --- Write to dataframe ---
    df['kf_hedge_ratio'] = gammas
    df['kf_intercept'] = alphas
    df['kf_spread'] = spread_levels
    df['kf_innovation'] = innovations
    df['kf_z_score'] = innovations / np.sqrt(inn_variances)

    # Level z-score: how far is the spread from zero
    # Uses expanding std (backward-looking, no future leakage)
    spread_series = pd.Series(spread_levels, index=df.index)
    rolling_std = spread_series.expanding(min_periods=50).std()
    df['kf_level_z'] = spread_levels / rolling_std.values

    # --- Print summary ---
    half_life = -1 / np.log(phi) if phi < 1 else float('inf')
    print(f"Converged in {iteration + 1} iteration(s)")
    print(f"  phi      = {phi:.6f}  (half-life = {half_life:.0f} trading days)")
    print(f"  sigma2_s = {sigma2_s:.6f}")
    print(f"  delta    = {delta:.1e}  (hyperparameter — tune via CV)")
    print(f"  R        = {R:.1e}  (numerical floor)")
    print(f"Hedge ratio:      {np.nanmin(gammas[burn:]):.4f} to {np.nanmax(gammas[burn:]):.4f}")
    print(f"Spread level std: {np.nanstd(spread_levels[burn:]):.4f}")
    print(f"Innovation z std: {np.nanstd(df['kf_z_score'].values[burn:]):.4f} (ideal ~ 1.0)")
    print(f"Level z std:      {np.nanstd(df['kf_level_z'].values[burn:]):.4f}")

    return df


#####################
# 3. Calendar features
#####################

def compute_calendar_features(df):
    """
    Add month and day-of-week as features.
    
    Relevant for agricultural commodities where planting (April-May)
    and harvest (September-October) drive structural shifts in the
    corn-soy spread.
    
    Parameters
    ----------
    df : pd.DataFrame — must have a DatetimeIndex
    
    Returns
    -------
    pd.DataFrame — input df with added columns:
        'month'       — integer 1-12
        'day_of_week' — integer 0 (Monday) to 4 (Friday)
    """
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek

    return df


#####################
# 4. Spread volatility as a feature
#####################

def compute_spread_vol(df, corn_col, soy_col, hedge_ratio_col, span=100):
    """
    EWMA volatility of the spread's actual P&L, added as a feature column.
    
    Same calculation as get_daily_vol in labels.py (used for barrier width),
    but stored in the dataframe so XGBoost can use it as a predictor.
    
    Uses actual tradeable P&L: delta_soy - gamma_{t-1} * delta_corn
    to avoid hedge ratio drift contamination.
    
    Parameters
    ----------
    df : pd.DataFrame      — must contain price and hedge ratio columns
    corn_col : str          — column name for corn prices
    soy_col : str           — column name for soybean prices
    hedge_ratio_col : str   — column name for hedge ratio (OLS or Kalman)
    span : int              — EWMA span in trading days (default 100)
    
    Returns
    -------
    pd.DataFrame — input df with added column:
        'spread_vol' — daily EWMA volatility of actual P&L
    """
    pnl = df[soy_col].diff() - df[hedge_ratio_col].shift(1) * df[corn_col].diff()
    df['spread_vol'] = pnl.ewm(span=span).std()

    return df