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