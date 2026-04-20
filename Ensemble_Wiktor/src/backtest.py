# Backtest for the corn-soybean strategy.
#
# Single-position, rebalance-and-replace framework:
#
#   At each rebalance date, close the previous spread trade entirely
#   and open a new one with the current target size and the hedge
#   ratio observed at that moment. Between rebalances, hold the
#   position and hedge ratio flat.
#
# No overlapping bets. Gross exposure <= 1 by construction. Look-ahead-
# free by virtue of trading on yesterday's close with yesterday's signal
# (position.shift(1), gamma.shift(1) in the P&L computation).


import numpy as np
import pandas as pd

from src.betsize import bet_size_from_prob, select_entry_dates


# ===================================================================
# Single-position backtest
# ===================================================================

def run_backtest(prob_pos, hedge_ratio, corn, soy,
                 all_dates=None, trade_freq='daily', method='prob',
                 notional=None):
    """
    Run a single-position backtest on a spread strategy.

    At each rebalance date, compute a target position from the model's
    probability prediction, freeze the hedge ratio, and hold until the
    next rebalance. Daily P&L uses the position and gamma from the
    previous close (look-ahead-free).

    Parameters
    ----------
    prob_pos : pd.Series
        P(y = +1) from the model, indexed by candidate rebalance dates.
    hedge_ratio : pd.Series
        Hedge ratio (gamma) indexed by all trading days. Frozen at each
        rebalance date and held until the next one.
    corn, soy : pd.Series
        Price series indexed by all trading days.
    all_dates : pd.DatetimeIndex, optional
        Output dates for the P&L series. Defaults to prob_pos.index
        intersected with the price indices.
    trade_freq : {'daily', 'monthly'}
        'daily'   : rebalance every day prob_pos is available.
        'monthly' : rebalance on the first trading day of each month.
    method : {'prob', 'sign'}
        'prob' : size from §10.3 probability formula in [-1, +1].
        'sign' : size is +/-1 based on sign of (p - 0.5). The single-
                 position analogue of §10.2 budgeting — uses only the
                 side of the model's prediction, not its magnitude.
    notional : float, optional
        Reference capital for converting P&L to returns. If None,
        defaults to the gross spread notional on the first output date:
            notional = soy[0] + gamma[0] * corn[0]
        Pass a float to override (e.g., soy[0] alone if you prefer the
        single-leg convention).

    Returns
    -------
    dict with:
        'position'        : pd.Series  - target position (step function,
                                          held flat between rebalances)
        'gamma'           : pd.Series  - hedge ratio held (step function)
        'daily_pnl'       : pd.Series  - daily P&L in spread dollar units
        'cum_pnl'         : pd.Series  - cumulative P&L
        'daily_ret'       : pd.Series  - daily P&L / notional
        'cum_ret'         : pd.Series  - cumulative P&L / notional
        'notional'        : float      - reference capital used
        'rebalance_dates' : pd.DatetimeIndex
    """
    # 0. Resolve output dates
    if all_dates is None:
        all_dates = prob_pos.index.intersection(corn.index).intersection(soy.index)
    all_dates = pd.DatetimeIndex(sorted(all_dates))

    # 1. Rebalance dates: candidates are prob_pos dates, filtered to all_dates
    rebalance = select_entry_dates(prob_pos.index, trade_freq)
    rebalance = rebalance[rebalance.isin(all_dates)]

    # 2. Target size at each rebalance date
    probs_at_reb = prob_pos.loc[rebalance].values
    if method == 'prob':
        sizes = bet_size_from_prob(probs_at_reb)
    elif method == 'sign':
        sizes = np.where(probs_at_reb >= 0.5, 1.0, -1.0)
    else:
        raise ValueError(f"method must be 'prob' or 'sign', got {method!r}")

    # 3. Step-function position and gamma over all_dates
    position_at_reb = pd.Series(sizes, index=rebalance)
    gamma_at_reb    = hedge_ratio.loc[rebalance]

    position_series = position_at_reb.reindex(all_dates, method='ffill').fillna(0.0)
    gamma_series    = gamma_at_reb.reindex(all_dates, method='ffill').fillna(0.0)

    # 4. Daily P&L using previous close's position and gamma (no look-ahead)
    #    PnL_t = m_{t-1} * [(soy_t - soy_{t-1}) - gamma_{t-1} * (corn_t - corn_{t-1})]
    position_lag = position_series.shift(1).fillna(0.0)
    gamma_lag    = gamma_series.shift(1).fillna(0.0)

    dsoy  = soy.loc[all_dates].diff()
    dcorn = corn.loc[all_dates].diff()

    daily_pnl = (position_lag * (dsoy - gamma_lag * dcorn)).fillna(0.0)
    cum_pnl   = daily_pnl.cumsum()

    # 5. Convert P&L to returns via a fixed reference notional
    if notional is None:
        # Gross spread notional on the first output date
        soy0   = soy.loc[all_dates].iloc[0]
        corn0  = corn.loc[all_dates].iloc[0]
        gamma0 = hedge_ratio.loc[all_dates].iloc[0]
        notional = float(soy0 + gamma0 * corn0)

    daily_ret = daily_pnl / notional
    cum_ret   = cum_pnl   / notional

    return {
        'position':        position_series,
        'gamma':           gamma_series,
        'daily_pnl':       daily_pnl,
        'cum_pnl':         cum_pnl,
        'daily_ret':       daily_ret,
        'cum_ret':         cum_ret,
        'notional':        notional,
        'rebalance_dates': rebalance,
    }


# ===================================================================
# Market-neutrality diagnostic: regress strategy returns on leg returns
# ===================================================================

def market_neutrality(daily_pnl, corn, soy, label=''):
    """
    Regress daily strategy P&L on corn and soybean returns.

    A market-neutral spread strategy should have near-zero loadings on
    both legs — the hedge ratio is supposed to neutralize directional
    exposure, leaving only the residual spread move as source of P&L.
    Significant loadings (especially on a single leg) indicate the
    hedge is either wrong or inconsistently applied.

    IMPORTANT: if this function is being used on an in-sample window
    (e.g., to look at strategy behavior during 2008 or COVID while
    the model was trained on those periods), the resulting market-
    neutrality metric is still valid — the regression only looks at
    the CORRELATION between strategy P&L and leg returns, not at
    whether the P&L itself was realistically attainable.

    Parameters
    ----------
    daily_pnl : pd.Series — strategy P&L in dollar units
    corn, soy : pd.Series — price series
    label : str — optional label printed in the summary

    Returns
    -------
    dict with:
        'beta_corn', 'beta_soy' : OLS slope coefficients
        't_corn', 't_soy'       : t-statistics
        'r2'                    : regression R^2
        'n'                     : number of observations
        'ann_ret', 'ann_vol'    : annualized return and vol of daily_pnl
    """
    idx   = daily_pnl.index.intersection(corn.index).intersection(soy.index)
    y     = daily_pnl.loc[idx].values
    dcorn = corn.loc[idx].diff().values
    dsoy  = soy.loc[idx].diff().values

    mask = np.isfinite(y) & np.isfinite(dcorn) & np.isfinite(dsoy)
    y, dcorn, dsoy = y[mask], dcorn[mask], dsoy[mask]

    X = np.column_stack([np.ones(len(y)), dcorn, dsoy])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    resid = y - X @ beta
    n, k  = X.shape
    sigma2 = (resid @ resid) / (n - k)
    cov    = sigma2 * np.linalg.inv(X.T @ X)
    se     = np.sqrt(np.diag(cov))
    tvals  = beta / se

    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = (resid ** 2).sum()
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    ann_ret = y.sum() * (252.0 / n)
    ann_vol = y.std(ddof=1) * np.sqrt(252.0)

    if label:
        print(f"=== {label} ===")
    print(f"  n = {n}  |  P&L ann_ret = {ann_ret:+.3f}  ann_vol = {ann_vol:.3f}")
    print(f"  beta_corn = {beta[1]:+.5f}  (t = {tvals[1]:+.2f})")
    print(f"  beta_soy  = {beta[2]:+.5f}  (t = {tvals[2]:+.2f})")
    print(f"  R^2       = {r2:.4f}")

    return {
        'beta_corn': float(beta[1]),
        'beta_soy':  float(beta[2]),
        't_corn':    float(tvals[1]),
        't_soy':     float(tvals[2]),
        'r2':        float(r2),
        'n':         int(n),
        'ann_ret':   float(ann_ret),
        'ann_vol':   float(ann_vol),
    }