import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt


# =============================================================================
# 1. SPREAD COMPUTATION
# =============================================================================

def compute_rolling_spread(prices, s1, s2, window=252):
    """
    Rolling OLS hedge ratio with intercept:
        s1 = alpha + beta * s2 + epsilon
        spread = s1 - alpha - beta * s2

    Parameters
    ----------
    prices : pd.DataFrame  — columns are ticker names
    s1, s2 : str           — ticker names
    window : int           — lookback window in trading days

    Returns
    -------
    spread : pd.Series     — spread time series (NaN for first `window` rows)
    beta   : pd.Series     — rolling hedge ratio
    alpha  : pd.Series     — rolling intercept
    """
    n = len(prices)
    beta_arr  = np.full(n, np.nan)
    alpha_arr = np.full(n, np.nan)

    y_vals = prices[s1].values
    x_vals = prices[s2].values

    for t in range(window, n):
        x = x_vals[t - window:t]
        y = y_vals[t - window:t]
        X = np.column_stack([np.ones(window), x])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        alpha_arr[t] = coef[0]
        beta_arr[t]  = coef[1]

    spread = y_vals - alpha_arr - beta_arr * x_vals

    return (
        pd.Series(spread,    index=prices.index, name=f'{s1}-{s2}_spread'),
        pd.Series(beta_arr,  index=prices.index, name=f'{s1}-{s2}_beta'),
        pd.Series(alpha_arr, index=prices.index, name=f'{s1}-{s2}_alpha'),
    )


def compute_all_spreads(prices, pairs, window=252):
    """
    Wrapper: compute rolling spreads for all pairs.

    Returns
    -------
    spreads : dict  {(s1, s2): pd.Series}
    betas   : dict  {(s1, s2): pd.Series}
    alphas  : dict  {(s1, s2): pd.Series}
    """
    spreads, betas, alphas = {}, {}, {}
    for s1, s2 in pairs:
        spread, beta, alpha = compute_rolling_spread(prices, s1, s2, window)
        spreads[(s1, s2)] = spread
        betas[(s1, s2)]   = beta
        alphas[(s1, s2)]  = alpha
        print(f'Spread computed: {s1}-{s2}')
    return spreads, betas, alphas


# =============================================================================
# 2. OU PARAMETER ESTIMATION
# =============================================================================

def estimate_ou_params(spread):
    """
    MLE estimation of OU parameters and log-likelihood score for a spread
    window. Implements equations (2)-(4) from Lee, Leung & Ning (2023).

    Parameters
    ----------
    spread : array-like  — spread values, no NaNs, length >= 2

    Returns
    -------
    mu    : float  — speed of mean reversion (higher = faster)
    theta : float  — long-term mean
    sigma : float  — volatility
    ll    : float  — average log-likelihood score (used for MRB/MRR weights)
    """
    x  = np.asarray(spread, dtype=float)
    n  = len(x) - 1
    dt = 1  # daily

    Xx  = np.sum(x[:-1])
    Xy  = np.sum(x[1:])
    Xxx = np.sum(x[:-1] ** 2)
    Xxy = np.sum(x[:-1] * x[1:])
    Xyy = np.sum(x[1:] ** 2)

    # Optimal theta
    denom = n * (Xxx - Xxy) - (Xx ** 2 - Xx * Xy)
    if abs(denom) < 1e-12:
        return np.nan, np.nan, np.nan, np.nan

    theta = (Xy * Xxx - Xx * Xxy) / denom

    # Optimal mu
    num_mu = Xxy - theta * Xx - theta * Xy + n * theta ** 2
    den_mu = Xxx - 2 * theta * Xx + n * theta ** 2
    if den_mu <= 0 or num_mu / den_mu <= 0:
        return np.nan, np.nan, np.nan, np.nan

    mu = -np.log(num_mu / den_mu) / dt

    if mu <= 0:
        return np.nan, np.nan, np.nan, np.nan

    # Optimal sigma
    e1 = np.exp(-mu * dt)
    e2 = np.exp(-2 * mu * dt)
    sigma2 = (2 * mu / (n * (1 - e2))) * (
        Xyy
        - 2 * e1 * Xxy
        + e2 * Xxx
        - 2 * theta * (1 - e1) * (Xy - e1 * Xx)
        + n * theta ** 2 * (1 - e1) ** 2
    )
    sigma = np.sqrt(max(sigma2, 1e-10))

    # Average log-likelihood
    sigma_tilde = np.sqrt(sigma ** 2 * (1 - e2) / (2 * mu))
    ll = (
        -0.5 * np.log(2 * np.pi)
        - np.log(sigma_tilde)
        - (1 / (2 * n * sigma_tilde ** 2))
        * np.sum(
            (x[1:] - x[:-1] * e1 - theta * (1 - e1)) ** 2
        )
    )

    return mu, theta, sigma, ll


# =============================================================================
# 3. PORTFOLIO WEIGHT COMPUTATION
# =============================================================================

def compute_mrb_weights(ou_params):
    """
    Mean Reversion Budgeting (MRB) weights — equation (4) of the paper.
    Allocates more capital to pairs that are more OU-like and revert faster.

    Parameters
    ----------
    ou_params : dict  {pair: (mu, theta, sigma, ll)}
                pairs with NaN params are assigned zero weight

    Returns
    -------
    weights : dict  {pair: float}  — sum to 1
    """
    pairs  = list(ou_params.keys())
    mus    = np.array([ou_params[p][0] for p in pairs], dtype=float)
    sigmas = np.array([ou_params[p][2] for p in pairs], dtype=float)
    lls    = np.array([ou_params[p][3] for p in pairs], dtype=float)

    # Relative speed of mean reversion: mu / sigma
    mu_r = np.where(sigmas > 0, mus / sigmas, np.nan)

    # Min-max normalisation (set NaNs to 0 after normalisation)
    def minmax(arr):
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0 or valid.max() == valid.min():
            return np.zeros_like(arr)
        norm = (arr - valid.min()) / (valid.max() - valid.min())
        norm = np.where(np.isfinite(norm), norm, 0.0)
        return norm

    mu_r_norm = minmax(mu_r)
    ll_norm   = minmax(lls)

    raw = mu_r_norm * ll_norm
    total = raw.sum()

    if total <= 0:
        # Fallback to equal weights
        w = np.ones(len(pairs)) / len(pairs)
    else:
        w = raw / total

    return {p: float(w[i]) for i, p in enumerate(pairs)}


def compute_mrr_weights(ou_params):
    """
    Mean Reversion Ranking (MRR) weights — equation (5) of the paper.
    Ranks pairs by mu/sigma * ll score and assigns linearly spaced weights.

    Parameters
    ----------
    ou_params : dict  {pair: (mu, theta, sigma, ll)}

    Returns
    -------
    weights : dict  {pair: float}  — sum to 1
    """
    pairs  = list(ou_params.keys())
    n      = len(pairs)
    mus    = np.array([ou_params[p][0] for p in pairs], dtype=float)
    sigmas = np.array([ou_params[p][2] for p in pairs], dtype=float)
    lls    = np.array([ou_params[p][3] for p in pairs], dtype=float)

    mu_r = np.where(sigmas > 0, mus / sigmas, 0.0)

    def minmax(arr):
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0 or valid.max() == valid.min():
            return np.zeros_like(arr)
        norm = (arr - valid.min()) / (valid.max() - valid.min())
        return np.where(np.isfinite(norm), norm, 0.0)

    score = minmax(mu_r) * minmax(lls)

    # Rank ascending (lowest score gets lowest weight)
    rank_order = np.argsort(np.argsort(score))  # 0 = worst, n-1 = best

    # Linearly spaced weights per equation (5)
    # w_k = (n - 1 + 2k) / (2n(n-1))  for k = 0..n-1 (0-indexed rank)
    if n == 1:
        w = np.array([1.0])
    else:
        w = np.array([(n - 1 + 2 * k) / (2 * n * (n - 1))
                      for k in rank_order], dtype=float)
        w = np.where(w < 0, 0.0, w)
        w /= w.sum()

    return {p: float(w[i]) for i, p in enumerate(pairs)}


# =============================================================================
# 4. TRADING SIGNAL
# =============================================================================

def compute_trading_signal(spread_window, K=1.0):
    """
    Entry/exit signal based on rolling mean and SD of spread.
    Uses the LAST value of the window as the current observation.

    Parameters
    ----------
    spread_window : array-like  — recent M days of spread values
    K             : float       — entry threshold (multiples of SD)

    Returns
    -------
    signal : int  — +1 (long spread), -1 (short spread), 0 (no trade)
    """
    x   = np.asarray(spread_window, dtype=float)
    mu  = np.mean(x[:-1])   # mean of past M-1 days
    sd  = np.std(x[:-1])    # SD of past M-1 days
    cur = x[-1]              # current spread value

    if sd == 0:
        return 0

    if cur < mu - K * sd:
        return 1    # spread too low → long
    elif cur > mu + K * sd:
        return -1   # spread too high → short
    else:
        return 0


# =============================================================================
# 5. BACKTEST ENGINE
# =============================================================================

def run_backtest(prices, spreads, pairs,
                 K=1.0, M=63, ols_window=252,
                 rebal_freq=63, ou_window=252,
                 method='MRR', initial_capital=1.0):
    """
    Full backtest of the multi-pair diversification framework.

    Parameters
    ----------
    prices          : pd.DataFrame  — ETF close prices
    spreads         : dict          — {(s1,s2): pd.Series} from compute_all_spreads
    pairs           : list          — list of (s1, s2) tuples
    K               : float         — entry threshold
    M               : int           — rolling window for signal (days)
    ols_window      : int           — OLS lookback used when computing spreads
    rebal_freq      : int           — rebalancing frequency in trading days (~63=quarterly)
    ou_window       : int           — lookback for OU estimation at each rebal date
    method          : str           — 'MRB', 'MRR', or 'EW' (equal weight)
    initial_capital : float         — starting capital per pair (normalised)

    Returns
    -------
    portfolio_returns : pd.Series   — daily portfolio returns
    weights_history  : pd.DataFrame — weights at each rebalancing date
    """
    # Align all spreads to a common index, drop NaN rows
    spread_df = pd.DataFrame({p: spreads[p] for p in pairs})
    valid_start = spread_df.dropna().index[0]
    spread_df = spread_df.loc[valid_start:]
    dates = spread_df.index
    T = len(dates)

    # Identify rebalancing points (after enough history for OU estimation)
    rebal_dates = []
    t = ou_window  # first rebal needs ou_window of spread history
    while t < T:
        rebal_dates.append(t)
        t += rebal_freq

    # Storage
    portfolio_returns = pd.Series(0.0, index=dates)
    weights_history   = []
    current_weights   = {p: 1.0 / len(pairs) for p in pairs}  # equal weight initially
    positions         = {p: 0 for p in pairs}                  # current position per pair

    for t in range(M, T):
        date = dates[t]

        # --- Rebalancing ---
        if t in rebal_dates:
            ou_params = {}
            for p in pairs:
                window_spread = spread_df[p].iloc[t - ou_window:t].dropna()
                if len(window_spread) < 30:
                    ou_params[p] = (np.nan, np.nan, np.nan, np.nan)
                else:
                    ou_params[p] = estimate_ou_params(window_spread)

            if method == 'MRB':
                current_weights = compute_mrb_weights(ou_params)
            elif method == 'MRR':
                current_weights = compute_mrr_weights(ou_params)
            else:  # Equal weight
                current_weights = {p: 1.0 / len(pairs) for p in pairs}

            weights_history.append({'date': date, **{str(p): current_weights[p] for p in pairs}})

            # Liquidate all positions on rebalancing
            positions = {p: 0 for p in pairs}

        # --- Trading signals & P&L ---
        daily_pnl = 0.0

        for p in pairs:
            spread_window = spread_df[p].iloc[t - M:t + 1].values
            if np.any(np.isnan(spread_window)):
                continue

            signal       = compute_trading_signal(spread_window, K)
            prev_pos     = positions[p]
            w            = current_weights[p]
            mean_spread  = np.mean(spread_window[:-1])
            spread_std   = np.std(spread_window[:-1])
            s_prev       = spread_window[-2]
            s_curr       = spread_window[-1]

            # Normalise daily spread change by rolling std → unit-free return
            if spread_std < 1e-10:
                continue
            spread_ret = (s_curr - s_prev) / spread_std
            spread_ret = np.clip(spread_ret, -3, 3)  # cap at ±3 std moves

            # P&L: position * normalised spread return * capital weight
            daily_pnl += prev_pos * spread_ret * w

            # Position update
            if prev_pos == 0:
                positions[p] = signal
            elif prev_pos == 1 and s_curr > mean_spread:
                positions[p] = 0
            elif prev_pos == -1 and s_curr < mean_spread:
                positions[p] = 0

        portfolio_returns[date] = daily_pnl

    weights_df = pd.DataFrame(weights_history).set_index('date') if weights_history else pd.DataFrame()

    return portfolio_returns, weights_df


# =============================================================================
# 6. PERFORMANCE METRICS
# =============================================================================

def compute_performance(returns, freq=252):
    """
    Compute key performance statistics from a daily return series.
 
    Parameters
    ----------
    returns : pd.Series  — daily portfolio returns
    freq    : int        — trading days per year
 
    Returns
    -------
    stats : dict
    """
    returns = returns.dropna()  # keep zeros, only drop NaN
 
    # Aggregate to monthly returns (sum within each month)
    monthly = returns.resample('ME').sum()
 
    ann_return  = monthly.mean() * 12        # 12 months per year
    ann_vol     = monthly.std() * np.sqrt(12)
    sharpe      = ann_return / ann_vol if ann_vol > 0 else np.nan
    cum         = returns.cumsum()           # daily cumsum for drawdown
    rolling_max = cum.cummax()
    drawdown    = cum - rolling_max
    max_dd      = drawdown.min()
 
    return {
        'Ann. Return (normalized)': round(ann_return, 4),
        'Ann. Vol (normalized)':    round(ann_vol, 4),
        'Sharpe':                   round(sharpe, 4),
        'Max Drawdown (normalized)':round(max_dd, 4),
        'Final Cumulative P&L':     round(cum.iloc[-1], 4),
    }

# =============================================================================
# 7. SENSITIVITY ANALYSIS
# =============================================================================
 
def sensitivity_K(prices, spreads, pairs,
                  K_values=None, M=63, ols_window=252,
                  rebal_freq=63, ou_window=252):
    """
    Sensitivity analysis over entry threshold K.
    Tests each K value for all three methods (EW, MRB, MRR).
 
    Returns
    -------
    results : dict  {K: {method: stats}}
    """
    if K_values is None:
        K_values = [0.5, 0.75, 1.0, 1.25, 1.5]
 
    results = {}
    for K in K_values:
        results[K] = {}
        for method in ['EW', 'MRB', 'MRR']:
            returns, _ = run_backtest(
                prices=prices, spreads=spreads, pairs=pairs,
                K=K, M=M, ols_window=ols_window,
                rebal_freq=rebal_freq, ou_window=ou_window,
                method=method
            )
            results[K][method] = compute_performance(returns)
        print(f'K={K} done')
    return results
 
 
def sensitivity_window(prices, pairs,
                       windows=None, K=1.0, M=63,
                       rebal_freq=63, ou_window=252):
    """
    Sensitivity analysis over OLS hedge ratio window.
    Recomputes spreads for each window value.
 
    Returns
    -------
    results : dict  {window: {method: stats}}
    """
    if windows is None:
        windows = [126, 252, 504]
 
    results = {}
    for w in windows:
        print(f'Computing spreads for window={w}...')
        spreads_w, _, _ = compute_all_spreads(prices, pairs, window=w)
        results[w] = {}
        for method in ['EW', 'MRB', 'MRR']:
            returns, _ = run_backtest(
                prices=prices, spreads=spreads_w, pairs=pairs,
                K=K, M=M, ols_window=w,
                rebal_freq=rebal_freq, ou_window=ou_window,
                method=method
            )
            results[w][method] = compute_performance(returns)
        print(f'Window={w} done')
    return results
 
 
def sensitivity_rebal(prices, spreads, pairs,
                      rebal_values=None, K=1.0, M=63,
                      ols_window=252, ou_window=252):
    """
    Sensitivity analysis over rebalancing frequency.
    Tests monthly (21), quarterly (63), semi-annual (126) rebalancing.
 
    Returns
    -------
    results : dict  {rebal_freq: {method: stats}}
    """
    if rebal_values is None:
        rebal_values = [21, 63, 126]
 
    labels = {21: 'Monthly', 63: 'Quarterly', 126: 'Semi-annual'}
    results = {}
    for rf in rebal_values:
        results[rf] = {}
        for method in ['EW', 'MRB', 'MRR']:
            returns, _ = run_backtest(
                prices=prices, spreads=spreads, pairs=pairs,
                K=K, M=M, ols_window=ols_window,
                rebal_freq=rf, ou_window=ou_window,
                method=method
            )
            results[rf][method] = compute_performance(returns)
        print(f'Rebal={labels.get(rf, rf)} done')
    return results
 
 
def sensitivity_ou_window(prices, spreads, pairs,
                          ou_windows=None, K=1.0, M=63,
                          ols_window=252, rebal_freq=63):
    """
    Sensitivity analysis over OU estimation window.
    Only meaningful for MRB and MRR (EW ignores OU params).
 
    Returns
    -------
    results : dict  {ou_window: {method: stats}}
    """
    if ou_windows is None:
        ou_windows = [63, 126, 252]
 
    results = {}
    for ow in ou_windows:
        results[ow] = {}
        for method in ['EW', 'MRB', 'MRR']:
            returns, _ = run_backtest(
                prices=prices, spreads=spreads, pairs=pairs,
                K=K, M=M, ols_window=ols_window,
                rebal_freq=rebal_freq, ou_window=ow,
                method=method
            )
            results[ow][method] = compute_performance(returns)
        print(f'OU window={ow} done')
    return results
 
 
def plot_sensitivity(results, param_name, title=None):
    """
    Bar chart of Sharpe ratios across parameter values and methods.
 
    Parameters
    ----------
    results    : dict  {param_value: {method: stats}}
    param_name : str   — x-axis label
    title      : str   — plot title
    """
    methods    = ['EW', 'MRB', 'MRR']
    param_vals = list(results.keys())
    x          = np.arange(len(param_vals))
    width      = 0.25
    colors     = ['steelblue', 'darkorange', 'seagreen']
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    # --- Sharpe ratio ---
    ax = axes[0]
    for i, method in enumerate(methods):
        sharpes = [results[v][method]['Sharpe'] for v in param_vals]
        ax.bar(x + i * width, sharpes, width, label=method, color=colors[i])
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(v) for v in param_vals])
    ax.set_xlabel(param_name)
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio')
    ax.legend()
    ax.grid(axis='y', alpha=0.4)
    ax.axhline(0, color='black', linewidth=0.8)
 
    # --- Final cumulative P&L ---
    ax = axes[1]
    for i, method in enumerate(methods):
        pnls = [results[v][method]['Final Cumulative P&L'] for v in param_vals]
        ax.bar(x + i * width, pnls, width, label=method, color=colors[i])
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(v) for v in param_vals])
    ax.set_xlabel(param_name)
    ax.set_ylabel('Cumulative P&L (normalized)')
    ax.set_title('Final Cumulative P&L')
    ax.legend()
    ax.grid(axis='y', alpha=0.4)
    ax.axhline(0, color='black', linewidth=0.8)
 
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
 
 
def run_sensitivity_all(prices, spreads, pairs,
                        K=1.0, M=63, ols_window=252,
                        rebal_freq=63, ou_window=252):
    """
    Run all four sensitivity tests and plot results.
 
    Returns
    -------
    all_results : dict  {test_name: results_dict}
    """
    print('=== Sensitivity: K ===')
    res_K = sensitivity_K(
        prices, spreads, pairs,
        M=M, ols_window=ols_window,
        rebal_freq=rebal_freq, ou_window=ou_window
    )
    plot_sensitivity(res_K, param_name='K (entry threshold)',
                     title='Sensitivity to Entry Threshold K')
 
    print('\n=== Sensitivity: OLS Window ===')
    res_win = sensitivity_window(
        prices, pairs,
        K=K, M=M, rebal_freq=rebal_freq, ou_window=ou_window
    )
    plot_sensitivity(res_win, param_name='OLS Window (days)',
                     title='Sensitivity to OLS Hedge Ratio Window')
 
    print('\n=== Sensitivity: Rebalancing Frequency ===')
    res_rebal = sensitivity_rebal(
        prices, spreads, pairs,
        K=K, M=M, ols_window=ols_window, ou_window=ou_window
    )
    plot_sensitivity(res_rebal, param_name='Rebal Frequency (days)',
                     title='Sensitivity to Rebalancing Frequency')
 
    print('\n=== Sensitivity: OU Window ===')
    res_ou = sensitivity_ou_window(
        prices, spreads, pairs,
        K=K, M=M, ols_window=ols_window, rebal_freq=rebal_freq
    )
    plot_sensitivity(res_ou, param_name='OU Window (days)',
                     title='Sensitivity to OU Estimation Window')
 
    return {
        'K':          res_K,
        'ols_window': res_win,
        'rebal':      res_rebal,
        'ou_window':  res_ou,
    }