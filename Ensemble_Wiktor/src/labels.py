# This file contains functions for assigning labels


def get_daily_vol(corn, soy, hedge_ratio, span=100):
    '''
    Daily volatility of the spread's actual P&L, using EWMA std.
    
    Adapted from AFML Snippet 3.1 for a cointegration spread where
    percentage returns are undefined (spread crosses zero).
    
    Uses actual tradeable P&L:  delta_soy - gamma_{t-1} * delta_corn
    instead of spread.diff(), which is contaminated by rolling-window
    hedge ratio drift (see AFML Section 2.4.1, ETF Trick).
    
    All inputs are backward-looking: no look-ahead bias.
    
    Parameters
    ----------
    corn : pd.Series     — corn price series (DatetimeIndex)
    soy : pd.Series      — soybean price series (DatetimeIndex)
    hedge_ratio : pd.Series — rolling hedge ratio (DatetimeIndex)
    span : int            — EWMA span in trading days (default 100)
        
    Returns
    -------
    pd.Series — daily volatility estimate, aligned to input index
    '''
    # Actual P&L using YESTERDAY's gamma (no look-ahead)
    pnl = soy.diff() - hedge_ratio.shift(1) * corn.diff()
    
    # EWMA standard deviation (backward-looking by construction)
    vol = pnl.ewm(span=span).std()
    
    return vol


def get_vertical_barrier(trading_days, num_days=150, t_events=None):
    '''
    Set vertical barrier as num_days TRADING days after each event.
    
    Adapted from AFML Snippet 3.4. Uses positional indexing instead of
    pd.Timedelta to count trading days, not calendar days. This matters
    for daily bars where weekends and holidays would otherwise shrink the
    effective holding period.
    
    Parameters
    ----------
    trading_days : pd.DatetimeIndex — index of all trading days in the dataset
    num_days : int                  — vertical barrier in trading days (default 150)
    t_events : pd.DatetimeIndex     — event timestamps (default None = every trading day).
                                      Use this to restrict entries to specific dates,
                                      e.g. days the association actually trades.
        
    Returns
    -------
    pd.Series — vertical barrier timestamps, indexed by event date.
                Events too close to the end of the series are dropped.
    '''
    import pandas as pd

    if t_events is None:
        # Default: every trading day is an event
        t1 = pd.Series(
            trading_days[num_days:],
            index=trading_days[:-num_days]
        )
    else:
        # Sparse events: find each event's position in the daily index,
        # then step forward num_days positions
        event_positions = trading_days.searchsorted(t_events)
        barrier_positions = event_positions + num_days
        valid_mask = barrier_positions < len(trading_days)
        
        t1 = pd.Series(
            trading_days[barrier_positions[valid_mask]],
            index=t_events[valid_mask]
        )
    
    return t1


def apply_pt_sl_on_t1(corn, soy, hedge_ratio, t1, trgt, pt_sl=[1, 1]):
    '''
    Find the first barrier touch for each event.
    
    Adapted from AFML Snippet 3.2 for a cointegration spread.
    Uses frozen gamma at entry to compute the actual tradeable P&L path,
    avoiding hedge ratio drift contamination.
    
    For each event (entry date), we:
      1. Freeze gamma at the entry date.
      2. Compute the P&L path to the vertical barrier:
         pnl[t] = (soy[t] - soy[entry]) - gamma_entry * (corn[t] - corn[entry])
      3. Record the first time P&L crosses the upper (profit-take) or
         lower (stop-loss) barrier, if ever.
    
    Parameters
    ----------
    corn : pd.Series        — corn price series (DatetimeIndex)
    soy : pd.Series         — soybean price series (DatetimeIndex)  
    hedge_ratio : pd.Series — rolling hedge ratio (DatetimeIndex)
    t1 : pd.Series          — vertical barrier timestamps (index = entry dates,
                               from get_vertical_barrier)
    trgt : pd.Series        — volatility target for barrier width (from get_daily_vol)
    pt_sl : list of 2 floats — [profit_take_multiplier, stop_loss_multiplier].
                               0 = disable that barrier. (default [1, 1])
        
    Returns
    -------
    pd.DataFrame — columns ['pt', 'sl', 't1'] with the timestamp of each
                   barrier touch (NaT if not touched). Index = entry dates.
    '''
    import pandas as pd
    import numpy as np

    events = t1.index
    trgt = trgt.reindex(events)
    
    # Barrier widths (NaN if disabled)
    if pt_sl[0] > 0:
        upper = pt_sl[0] * trgt
    else:
        upper = pd.Series(np.nan, index=events)
    
    if pt_sl[1] > 0:
        lower = -pt_sl[1] * trgt
    else:
        lower = pd.Series(np.nan, index=events)
    
    # Output: timestamp of each barrier touch
    result = pd.DataFrame(index=events, columns=['pt', 'sl', 't1'], dtype='datetime64[ns]')
    result['t1'] = t1
    
    for entry in events:
        barrier = t1.loc[entry]
        gamma_entry = hedge_ratio.loc[entry]
        corn_entry = corn.loc[entry]
        soy_entry = soy.loc[entry]
        
        # P&L path with frozen gamma (day 0 = 0 by construction)
        path_corn = corn.loc[entry:barrier]
        path_soy = soy.loc[entry:barrier]
        pnl_path = (path_soy - soy_entry) - gamma_entry * (path_corn - corn_entry)
        
        # First day P&L >= upper barrier
        if pt_sl[0] > 0:
            pt_touches = pnl_path[pnl_path >= upper.loc[entry]]
            if len(pt_touches) > 0:
                result.loc[entry, 'pt'] = pt_touches.index[0]
        
        # First day P&L <= lower barrier
        if pt_sl[1] > 0:
            sl_touches = pnl_path[pnl_path <= lower.loc[entry]]
            if len(sl_touches) > 0:
                result.loc[entry, 'sl'] = sl_touches.index[0]
    
    return result


def get_bins(touches, corn, soy, hedge_ratio):
    '''
    Assign labels based on which barrier was touched first.
    
    Adapted from AFML Snippet 3.5 / 3.7 for a cointegration spread.
    Computes the frozen-gamma P&L at the time of first barrier touch,
    then labels by sign of that P&L.
    
    Parameters
    ----------
    touches : pd.DataFrame  — output of apply_pt_sl_on_t1, with columns
                               ['pt', 'sl', 't1'] of barrier touch timestamps
    corn : pd.Series        — corn price series
    soy : pd.Series         — soybean price series
    hedge_ratio : pd.Series — rolling hedge ratio
    
    Returns
    -------
    pd.DataFrame — columns:
        't1'   : timestamp of first barrier touch (needed for Ch.4 sample weights)
        'ret'  : P&L at first touch (frozen gamma)
        'bin'  : label in {-1, 1}. Sign of ret.
    '''
    import numpy as np
    import pandas as pd

    # Find the earliest barrier touch across pt, sl, t1
    first_touch = touches[['pt', 'sl', 't1']].min(axis=1)
    
    out = pd.DataFrame(index=touches.index)
    out['t1'] = first_touch
    
    # Compute frozen-gamma P&L at the first touch time
    for entry in out.index:
        touch_time = out.loc[entry, 't1']
        gamma_entry = hedge_ratio.loc[entry]
        
        pnl = (soy.loc[touch_time] - soy.loc[entry]) \
            - gamma_entry * (corn.loc[touch_time] - corn.loc[entry])
        
        out.loc[entry, 'ret'] = pnl
    
    out['ret'] = out['ret'].astype(float)
    out['bin'] = np.sign(out['ret']).astype(int)
    
    return out


def get_avg_uniqueness(labels, trading_days):
    '''
    Compute average uniqueness (sample weight) for each event.
    
    Adapted from AFML Snippets 4.1–4.2. Measures how much each event's 
    outcome overlaps with other concurrent events. Events that share more
    of their lifespan with other events get lower uniqueness (downweighted).
    
    Parameters
    ----------
    labels : pd.DataFrame     — output of get_bins, must have column 't1'
                                 (first barrier touch). Index = entry dates.
    trading_days : pd.DatetimeIndex — all trading days in the dataset
    
    Returns
    -------
    pd.Series — average uniqueness per event (between 0 and 1),
                indexed by entry date. Pass to XGBoost as sample_weight.
    '''
    import pandas as pd

    t0 = labels.index
    t1 = labels['t1']
    
    # Step 1: Count concurrent events on each trading day
    # For each day t, c_t = number of events where entry <= t <= first_touch
    concurrency = pd.Series(0.0, index=trading_days)
    for entry, exit_date in t1.items(): # .items() gives you both index(entry) and value(exit_date)
        concurrency.loc[entry:exit_date] += 1
    
    # Step 2: Average uniqueness per event
    # For event i, average 1/c_t over all days t in [entry_i, exit_i]
    avg_u = pd.Series(index=t0, dtype=float)
    for entry in t0:
        exit_date = t1.loc[entry]
        event_concurrency = concurrency.loc[entry:exit_date]
        avg_u.loc[entry] = (1.0 / event_concurrency).mean()
    
    return avg_u