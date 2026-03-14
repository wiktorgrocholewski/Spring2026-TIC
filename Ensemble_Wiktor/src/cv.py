# This file contains functions for Cross Validation of the trading strategy.
#
# Two validators:
#
#   1. WalkForwardPurgedCV (PRIMARY) — expanding-window walk-forward.
#      For test fold j, trains ONLY on periods 1..j-1. Eliminates the
#      feature leakage problem where sequential estimates (rolling OLS,
#      Kalman filter) computed on future data bleed into training features.
#      Purging at the train/test boundary handles label overlap.
#      No embargo needed — we never train on data after test.
#
#   2. PurgedKFold (SECONDARY) — symmetric purged k-fold from AFML Ch.7.
#      Kept for reference / comparison, but NOT recommended as the primary
#      validator for this pipeline because training on future data leaks
#      through sequentially estimated features.


import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score


# ===================================================================
# 1. Walk-Forward Expanding-Window CV with Purging
# ===================================================================

class WalkForwardPurgedCV:
    """
    Walk-forward cross-validator with expanding training window and purging.

    Splits observations into n_periods contiguous time-ordered blocks.
    For test period j (j >= 2), trains on all periods 1..j-1, then
    purges training observations whose labels (t1) leak into the test
    period.

    Period 1 is always train-only (no test), so n_periods = 3 gives
    2 test folds with expanding training sets.

    Why walk-forward instead of symmetric k-fold:
        Features like rolling OLS hedge ratio and Kalman filter estimates
        are computed sequentially from past prices. If we trained on
        period 3 to predict period 1 (as symmetric k-fold does), the
        model would see hedge ratios that were estimated using period-1
        prices — a form of feature information leakage that purging
        cannot fix.

    Parameters
    ----------
    n_periods : int
        Number of time blocks (default 3). Yields n_periods - 1 test folds.
    t1 : pd.Series
        First barrier touch timestamp for each observation.
        Index = entry date, value = exit date (from get_bins output).
    """

    def __init__(self, n_periods=3, t1=None):
        if t1 is None:
            raise ValueError("t1 (first barrier touch Series) is required")
        if n_periods < 2:
            raise ValueError("Need at least 2 periods (1 train + 1 test)")
        self.n_periods = n_periods
        self.t1 = t1

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_periods - 1

    def split(self, X, y=None, groups=None):
        """
        Yield (train_indices, test_indices) for each fold.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix. Must have a DatetimeIndex matching t1's index.

        Yields
        ------
        train_idx : np.ndarray of int — positional indices into X
        test_idx  : np.ndarray of int — positional indices into X
        """
        t1 = self.t1.loc[X.index]
        n = len(X)
        indices = np.arange(n)

        # Split into n_periods contiguous blocks
        bounds = [(i * n) // self.n_periods for i in range(self.n_periods + 1)]

        for j in range(1, self.n_periods):
            # Test: period j
            test_start = bounds[j]
            test_end = bounds[j + 1]  # exclusive
            test_idx = indices[test_start:test_end]
            test_start_date = X.index[test_start]

            # Train: all periods before j (expanding window)
            train_idx = indices[:test_start].copy()

            # --- Purge: remove training obs whose t1 leaks into test ---
            if len(train_idx) > 0:
                t1_train = t1.iloc[train_idx]
                leaked = t1_train >= test_start_date
                purge_idx = train_idx[leaked.values]
                train_idx = np.setdiff1d(train_idx, purge_idx)

            yield train_idx, test_idx


# ===================================================================
# 2. Symmetric Purged K-Fold (AFML Ch.7 — kept for reference)
# ===================================================================

class PurgedKFold:
    """
    Purged k-fold cross-validator for overlapping labels.

    WARNING: trains on future data. Use WalkForwardPurgedCV instead for
    pipelines with sequentially estimated features (rolling OLS, Kalman).

    Splits observations into k contiguous time-ordered folds. For each
    test fold, removes from training:
      (a) observations whose label spans into the test period (purge)
      (b) observations in a buffer after the test period (embargo)

    Parameters
    ----------
    n_splits : int
        Number of folds (default 5).
    t1 : pd.Series
        First barrier touch timestamp for each observation.
        Index = entry date, value = exit date (from get_bins output).
    pct_embargo : float
        Fraction of total observations to embargo after each test fold
        (default 0.01).
    """

    def __init__(self, n_splits=5, t1=None, pct_embargo=0.01):
        if t1 is None:
            raise ValueError("t1 (first barrier touch Series) is required")
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        t1 = self.t1.loc[X.index]
        n = len(X)
        embargo = int(n * self.pct_embargo)
        indices = np.arange(n)
        fold_bounds = [(i * n) // self.n_splits for i in range(self.n_splits + 1)]

        for j in range(self.n_splits):
            test_start = fold_bounds[j]
            test_end = fold_bounds[j + 1]

            test_idx = indices[test_start:test_end]
            test_times = X.index[test_start:test_end]

            train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

            # Purge
            if test_start > 0:
                train_before = indices[:test_start]
                t1_before = t1.iloc[train_before]
                leaked = t1_before >= test_times[0]
                purge_idx = train_before[leaked.values]
                train_idx = np.setdiff1d(train_idx, purge_idx)

            # Embargo
            if embargo > 0 and test_end < n:
                embargo_end = min(test_end + embargo, n)
                embargo_idx = indices[test_end:embargo_end]
                train_idx = np.setdiff1d(train_idx, embargo_idx)

            yield train_idx, test_idx


# ===================================================================
# 3. Scoring function (works with either CV class)
# ===================================================================

def cv_score(model, X, y, t1, sample_weight=None, cv=None,
             scoring='accuracy'):
    """
    Run cross-validation and return per-fold train & test scores.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Must have .fit(X, y, sample_weight=...) and .predict(X).
        For scoring='neg_log_loss', must also have .predict_proba(X).
    X : pd.DataFrame
        Feature matrix with DatetimeIndex.
    y : pd.Series
        Labels, aligned to X.
    t1 : pd.Series
        First barrier touch timestamps (from get_bins).
    sample_weight : pd.Series, optional
        Uniqueness weights (from get_avg_uniqueness). Applied during
        training only — test scoring is unweighted so it reflects
        true out-of-sample performance.
    cv : cross-validator instance, optional
        A WalkForwardPurgedCV or PurgedKFold instance. If None, defaults
        to WalkForwardPurgedCV(n_periods=3, t1=t1).
    scoring : str
        'accuracy' or 'neg_log_loss' (default 'accuracy').

    Returns
    -------
    dict with keys:
        'test_scores'  : list of float — per-fold test scores
        'train_scores' : list of float — per-fold train scores
        'fold_sizes'   : list of dict  — per-fold size diagnostics
    """
    from sklearn.base import clone

    if cv is None:
        cv = WalkForwardPurgedCV(n_periods=3, t1=t1)

    test_scores = []
    train_scores = []
    fold_sizes = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X)):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # --- Fold diagnostics ---
        n_test = len(test_idx)
        n_train = len(train_idx)

        # "purged" = observations between train and test that were removed
        test_start_pos = test_idx[0]
        n_before_test = test_start_pos  # everything before test
        n_purged = n_before_test - n_train

        fold_sizes.append({
            'fold': fold_i + 1,
            'train': n_train,
            'test': n_test,
            'purged': n_purged,
            'train_start': str(X_train.index[0].date()),
            'train_end': str(X_train.index[-1].date()),
            'test_start': str(X_test.index[0].date()),
            'test_end': str(X_test.index[-1].date()),
        })

        # --- Fit with sample weights on training set only ---
        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight.iloc[train_idx].values

        m = clone(model)
        m.fit(X_train.values, y_train.values, **fit_params)

        # --- Score ---
        if scoring == 'neg_log_loss':
            prob_test = m.predict_proba(X_test.values)
            test_scores.append(-log_loss(y_test.values, prob_test))
            prob_train = m.predict_proba(X_train.values)
            train_scores.append(-log_loss(y_train.values, prob_train))
        else:
            pred_test = m.predict(X_test.values)
            test_scores.append(accuracy_score(y_test.values, pred_test))
            pred_train = m.predict(X_train.values)
            train_scores.append(accuracy_score(y_train.values, pred_train))

    # --- Print summary ---
    print(f"{'Fold':<6} {'Train':>6} {'Test':>6} {'Purged':>7} "
          f"{'Train Dates':>25} {'Test Dates':>25} "
          f"{'Train Score':>12} {'Test Score':>11}")
    print("-" * 105)
    for fs, tr_s, te_s in zip(fold_sizes, train_scores, test_scores):
        train_dates = f"{fs['train_start']} → {fs['train_end']}"
        test_dates = f"{fs['test_start']} → {fs['test_end']}"
        print(f"{fs['fold']:<6} {fs['train']:>6} {fs['test']:>6} {fs['purged']:>7} "
              f"{train_dates:>25} {test_dates:>25} "
              f"{tr_s:>12.4f} {te_s:>11.4f}")
    print("-" * 105)
    print(f"{'Mean':<6} {'':>6} {'':>6} {'':>7} "
          f"{'':>25} {'':>25} "
          f"{np.mean(train_scores):>12.4f} {np.mean(test_scores):>11.4f}")
    print(f"{'Std':<6} {'':>6} {'':>6} {'':>7} "
          f"{'':>25} {'':>25} "
          f"{np.std(train_scores):>12.4f} {np.std(test_scores):>11.4f}")

    return {
        'test_scores': test_scores,
        'train_scores': train_scores,
        'fold_sizes': fold_sizes,
    }