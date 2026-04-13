# src/regime.py
#
# Bayesian regime-switching model with weather-driven regime probabilities.
#
# Model:
#   spread_t = (1-s_t)*c0 + s_t*c1 + phi*spread_{t-1} + eps_t
#   eps_t ~ N(0, sigma2)
#
#   z_t = w_t' alpha + delta * s_{t-1} + eta_t,  eta_t ~ N(0,1)   [Markov]
#   z_t = w_t' alpha + eta_t,                     eta_t ~ N(0,1)   [Independent]
#   s_t = I[z_t > 0]
#
# Identification: c0 < c1  (regime 0 = low spread, regime 1 = high spread)
#
# Building blocks from Paap (2025) Lecture 5:
#   - Example I  (Albert-Chib probit)   for the regime equation
#   - Example III (Switching regression) for the observation equation
#
# Two prior configurations for theta = (c0, c1, phi):
#   diffuse:      p(theta, sigma2) propto sigma^{-2}
#   informative:  theta | sigma2 ~ N(b0, sigma2 * B0),  p(sigma2) propto sigma^{-2}
#
# Prior for alpha (probit):
#   flat:          p(alpha) propto 1
#   informative:   alpha ~ N(a0, A0)

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_regime_data(df, spread_col, weather_cols, standardize_weather=True):
    """
    Align spread and weather arrays for the Gibbs sampler.

    Parameters
    ----------
    df : pd.DataFrame with datetime index
    spread_col : str — column name for the pre-computed spread
    weather_cols : list of str — column names for weather regressors
    standardize_weather : bool — if True, z-score weather variables

    Returns
    -------
    y : (T,) array — spread at time t
    y_lag : (T,) array — spread at time t-1
    W : (T, 1+k_w) array — intercept + weather regressors (aligned with y)
    col_names : list — ['intercept'] + weather column names
    """
    # drop rows with any NaN in relevant columns
    cols = [spread_col] + list(weather_cols)
    sub = df[cols].dropna()

    spread = sub[spread_col].values
    weather = sub[weather_cols].values

    if standardize_weather:
        w_mean = weather.mean(axis=0)
        w_std = weather.std(axis=0)
        w_std[w_std == 0] = 1.0
        weather = (weather - w_mean) / w_std

    y = spread[1:]           # spread_t
    y_lag = spread[:-1]      # spread_{t-1}
    W = np.column_stack((np.ones(len(y)), weather[1:]))   # intercept + weather

    col_names = ['intercept'] + list(weather_cols)
    return y, y_lag, W, col_names


# ---------------------------------------------------------------------------
# Gibbs sampler
# ---------------------------------------------------------------------------

def gibbs_regime_switching(y, y_lag, W,
                           nos=10000, nob=2000, nod=1,
                           markov=True,
                           prior_alpha_mean=None, prior_alpha_var=None,
                           prior_theta_mean=None, prior_theta_var=None,
                           seed=0):
    """
    Gibbs sampler for the regime-switching spread model.

    Parameters
    ----------
    y : (T,) array — spread at time t
    y_lag : (T,) array — spread at time t-1
    W : (T, k) array — probit regressors (intercept + weather)
    nos : int — number of valid posterior draws to keep
    nob : int — burn-in draws to discard
    nod : int — thinning (keep every nod-th draw)
    markov : bool — if True, add s_{t-1} as extra probit regressor
        (regime persistence). If False, independent switching.
    prior_alpha_mean : (k_alpha,) array or None
        Prior mean for probit coefficients.  k_alpha = k+1 if markov, k if not.
        None = flat prior.
    prior_alpha_var : (k_alpha, k_alpha) array or None
        Prior covariance for alpha.  None = flat prior.
    prior_theta_mean : (3,) array or None
        Prior mean for theta = (c0, c1, phi).  None = diffuse prior.
    prior_theta_var : (3,3) array or None
        Prior variance *scale* B0 for theta (so that theta|sigma2 ~ N(b0, sigma2*B0)).
        None = diffuse prior p(theta, sigma2) propto sigma^{-2}.
    seed : int

    Returns
    -------
    draws : dict
        'theta'       : (nos, 3) — draws of (c0, c1, phi)
        'sigma2'      : (nos,)   — draws of sigma^2
        'alpha'       : (nos, k_alpha) — draws of probit coefficients
                        last column is delta (persistence) if markov=True
        'regime_prob' : (T,)     — posterior mean of P(s_t = 1 | y)
    """
    np.random.seed(seed)

    T = len(y)
    k_w = W.shape[1]                       # weather regressors (incl. intercept)
    k_alpha = k_w + 1 if markov else k_w   # +1 for delta (s_{t-1} coefficient)

    # --- identify prior type ---
    flat_alpha = (prior_alpha_mean is None)
    diffuse_theta = (prior_theta_mean is None)

    if not flat_alpha:
        a0 = prior_alpha_mean
        A0_inv = np.linalg.inv(prior_alpha_var)

    if not diffuse_theta:
        b0 = prior_theta_mean
        B0_inv = np.linalg.inv(prior_theta_var)

    # --- storage ---
    total = nos * nod + nob
    draw_theta  = np.zeros((total, 3))
    draw_sigma2 = np.zeros(total)
    draw_alpha  = np.zeros((total, k_alpha))
    regime_sum  = np.zeros(T)        # running sum for posterior regime prob
    regime_cnt  = 0                  # count of valid draws contributing

    # --- starting values ---
    # simple OLS on spread_t = c + phi*spread_{t-1}
    X_init = np.column_stack((np.ones(T), y_lag))
    b_init = np.linalg.lstsq(X_init, y, rcond=None)[0]

    c0 = b_init[0] - 0.5
    c1 = b_init[0] + 0.5
    phi = np.clip(b_init[1], -0.99, 0.999)
    theta = np.array([c0, c1, phi])
    sigma2 = np.var(y - X_init @ b_init)

    alpha = np.zeros(k_alpha)
    st = np.zeros(T)

    # --- Gibbs sampler ---
    for i in range(total):

        if i % 5000 == 0:
            print(f'  iteration {i} / {total}')

        # ------------------------------------------------------------------
        # Step 1:  draw s_t from Bernoulli  (slide 28 adapted)
        # ------------------------------------------------------------------
        mu0 = theta[0] + theta[2] * y_lag       # regime-0 conditional mean
        mu1 = theta[1] + theta[2] * y_lag       # regime-1 conditional mean
        sd = np.sqrt(sigma2)

        # pre-compute observation log-likelihoods
        log_lik_1 = norm.logpdf(y, mu1, sd)
        log_lik_0 = norm.logpdf(y, mu0, sd)

        if markov:
            # Sequential draw: s_t depends on s_{t-1}
            # alpha layout: [alpha_weather (k_w), delta (1)]
            alpha_w = alpha[:k_w]
            delta = alpha[-1]
            W_alpha = W @ alpha_w       # (T,) — weather part of probit index

            # s_{t-1} is binary, so precompute both cases vectorised
            # case 0: s_{t-1}=0, probit index = W_alpha
            # case 1: s_{t-1}=1, probit index = W_alpha + delta
            logcdf_0 = norm.logcdf(W_alpha)              # log Phi(.) if s_{t-1}=0
            logsf_0  = norm.logsf(W_alpha)               # log(1-Phi(.))
            logcdf_1 = norm.logcdf(W_alpha + delta)      # log Phi(.) if s_{t-1}=1
            logsf_1  = norm.logsf(W_alpha + delta)       # log(1-Phi(.))

            # combine with observation likelihoods
            # lp1[case, t] = log P(s_t=1 | s_{t-1}=case) + log p(y_t | s_t=1)
            lp1_given0 = logcdf_0 + log_lik_1     # (T,)
            lp0_given0 = logsf_0  + log_lik_0
            lp1_given1 = logcdf_1 + log_lik_1
            lp0_given1 = logsf_1  + log_lik_0

            uniforms = np.random.uniform(size=T)

            for t in range(T):
                if t == 0 or st[t - 1] == 0.0:
                    log_diff = lp0_given0[t] - lp1_given0[t]
                else:
                    log_diff = lp0_given1[t] - lp1_given1[t]

                if log_diff > 500:
                    p1 = 0.0
                elif log_diff < -500:
                    p1 = 1.0
                else:
                    p1 = 1.0 / (1.0 + np.exp(log_diff))

                st[t] = 1.0 if uniforms[t] < p1 else 0.0

        else:
            # Independent: vectorised draw (original code)
            log_prior_1 = norm.logcdf(W @ alpha[:k_w])
            log_prior_0 = norm.logsf(W @ alpha[:k_w])

            log_omega1 = log_prior_1 + log_lik_1
            log_omega0 = log_prior_0 + log_lik_0

            prob = 1.0 / (1.0 + np.exp(log_omega0 - log_omega1))
            prob = np.clip(prob, 1e-12, 1 - 1e-12)

            st = (prob > np.random.uniform(size=T)).astype(float)

            # ensure both regimes have observations
            for _ in range(50):
                if np.sum(st) >= 2 and np.sum(1 - st) >= 2:
                    break
                st = (prob > np.random.uniform(size=T)).astype(float)

        # ensure both regimes have observations (both versions)
        if np.sum(st) < 2 or np.sum(1 - st) < 2:
            # skip this iteration, keep previous draws
            draw_theta[i]  = draw_theta[max(i-1, 0)]
            draw_sigma2[i] = draw_sigma2[max(i-1, 0)]
            draw_alpha[i]  = draw_alpha[max(i-1, 0)]
            continue

        # ------------------------------------------------------------------
        # Step 2:  draw z_t from truncated normal  (Albert-Chib, slide 14)
        # ------------------------------------------------------------------
        # Build the full probit design matrix W_full (T x k_alpha)
        if markov:
            st_lag = np.zeros(T)
            st_lag[1:] = st[:-1]       # s_{t-1}, with s_0 = 0
            W_full = np.column_stack((W, st_lag))
        else:
            W_full = W

        mean_z = W_full @ alpha
        ub = (st == 0) * norm.cdf(-mean_z) + (st == 1)
        lb = (st == 1) * norm.cdf(-mean_z)
        lb = np.clip(lb, 1e-12, 1 - 1e-12)
        ub = np.clip(ub, 1e-12, 1 - 1e-12)
        z = mean_z + norm.ppf(lb + (ub - lb) * np.random.uniform(size=T))

        # ------------------------------------------------------------------
        # Step 3:  draw alpha  (probit regression on z, slide 15)
        # ------------------------------------------------------------------
        WfWf = W_full.T @ W_full
        if flat_alpha:
            A_post = np.linalg.inv(WfWf)
            a_post = A_post @ (W_full.T @ z)
        else:
            A_post = np.linalg.inv(WfWf + A0_inv)
            a_post = A_post @ (W_full.T @ z + A0_inv @ a0)

        alpha = np.linalg.cholesky(A_post) @ np.random.normal(size=k_alpha) + a_post

        # ------------------------------------------------------------------
        # Step 4:  draw theta = (c0, c1, phi) with c0 < c1  (slide 30 adapted)
        # ------------------------------------------------------------------
        X_tilde = np.column_stack(((1 - st), st, y_lag))
        XtX = X_tilde.T @ X_tilde

        if diffuse_theta:
            B_post = np.linalg.inv(XtX)
            b_post = B_post @ (X_tilde.T @ y)
            dof_sigma = T
        else:
            B_post = np.linalg.inv(XtX + B0_inv)
            b_post = B_post @ (X_tilde.T @ y + B0_inv @ b0)
            dof_sigma = T + 3    # 3 = dim(theta)

        chol = np.linalg.cholesky(sigma2 * B_post)
        for _ in range(200):
            theta_cand = chol @ np.random.normal(size=3) + b_post
            if theta_cand[0] < theta_cand[1]:          # identification c0 < c1
                theta = theta_cand
                break

        # ------------------------------------------------------------------
        # Step 5:  draw sigma2 from IG-2  (slide 29 adapted)
        # ------------------------------------------------------------------
        res = y - X_tilde @ theta
        rss = res.T @ res

        if not diffuse_theta:
            rss = rss + (theta - b0).T @ B0_inv @ (theta - b0)

        u = np.random.normal(size=dof_sigma)
        sigma2 = rss / (u.T @ u)

        # ------------------------------------------------------------------
        # store draws
        # ------------------------------------------------------------------
        draw_theta[i]  = theta
        draw_sigma2[i] = sigma2
        draw_alpha[i]  = alpha

        if i >= nob:
            regime_sum += st
            regime_cnt += 1

    # --- select valid draws (remove burn-in, apply thinning) ---
    idx = range(nob, total, nod)

    draws = {
        'theta':       draw_theta[idx],
        'sigma2':      draw_sigma2[idx],
        'alpha':       draw_alpha[idx],
        'regime_prob': regime_sum / regime_cnt,
    }
    return draws