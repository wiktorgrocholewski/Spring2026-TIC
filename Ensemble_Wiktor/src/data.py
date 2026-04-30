# This file contains functions to fetch and preprocess data for the ensemble model.





#####################
# 1. This function fetches historical stock data from Yahoo Finance using the yfinance library. 
# It saves the cleaned data, ie with correctly defined columns, in the csv file with the specified name in the folder /data.
#####################

def fetch_yahoo_data(ticker, name, save_path, date_start=None, date_end=None):
    import yfinance as yf
    import pandas as pd

    # Fetch data from Yahoo Finance with max or specified date range
    if date_start and date_end:
        data = yf.download(ticker, start=date_start, end=date_end, timeout=10)
    else:
        data = yf.download(ticker, period='max', timeout=10)

    # Clean data to have nice columns

    # Flatten MultiIndex columns (e.g. ('Close', 'AAPL') -> 'close')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Clean column names: lowercase, append asset name
    data.columns = [f"{col.lower()}_{name}" for col in data.columns]

    # Clean up the index
    data.index.name = 'date'

    data.to_csv(f'{save_path}/data/{name}.csv')

    return data


#####################
# 2. This function merges the financial data from Yahoo based on date only for dates available in all datasets.
# It saves the merged data in the csv file with the specified name in the folder /data.
#####################

def merge_yahoo_datasets(names, merged_name, save_path):
    import pandas as pd

    # Load all datasets
    dataframes = []
    for name in names:
        df = pd.read_csv(f'{save_path}/data/{name}.csv', parse_dates=['date'], index_col='date')
        dataframes.append(df)

    # Merge on date using inner join to keep only common dates
    merged_data = dataframes[0]
    for df in dataframes[1:]:
        merged_data = merged_data.join(df, how='inner')

    # Save merged dataset
    merged_data.to_csv(f'{save_path}/data/{merged_name}.csv')

    return merged_data



#####################
# 3. This function fetches the weather data for a few relevant loactions
#####################

def fetch_weather_data(
    locations,
    start_date="2005-01-01",
    end_date=None,
    variables=None,
    save_path=None,
):
    """
    Fetch daily weather data from Open-Meteo Archive API for multiple locations.

    Parameters
    ----------
    locations : dict
        Mapping of location name to (latitude, longitude) tuple.
        Example: {"iowa_corn_belt": (42.0, -93.5), "mato_grosso": (-12.6, -55.5)}
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. If None, uses today's date.
    variables : list of str, optional
        Daily weather variables to request. If None, uses a curated set
        relevant for crop stress modeling.
    save_path : str, optional
        If provided, saves the combined DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame
        Combined weather data for all locations with columns prefixed by location name.
    """
    import openmeteo_requests
    import pandas as pd
    import requests_cache
    from retry_requests import retry
    from datetime import date

    if end_date is None:
        end_date = date.today().isoformat()

    # --- Default variables: agronomically relevant for crop stress ---
    if variables is None:
        variables = [
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "soil_moisture_0_to_7cm_mean",
            "soil_moisture_7_to_28cm_mean",
            "soil_moisture_28_to_100cm_mean",
            "soil_moisture_0_to_100cm_mean",
            "soil_temperature_0_to_7cm_mean",
            "et0_fao_evapotranspiration_sum",
            "relative_humidity_2m_mean",
            "vapour_pressure_deficit_max",
            "wet_bulb_temperature_2m_mean",
            "snowfall_water_equivalent_sum",
            "wind_speed_10m_max",
        ]

    # --- API client setup ---
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"

    location_names = list(locations.keys())
    latitudes = [coord[0] for coord in locations.values()]
    longitudes = [coord[1] for coord in locations.values()]

    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "start_date": start_date,
        "end_date": end_date,
        "daily": variables,
        "timezone": "auto",
    }

    responses = openmeteo.weather_api(url, params=params)

    # --- Parse each location's response ---
    location_dfs = []

    for i, response in enumerate(responses):
        name = location_names[i]
        print(
            f"Processing '{name}': {response.Latitude():.2f}°N, "
            f"{response.Longitude():.2f}°W"
        )

        daily = response.Daily()

        # Build date index — normalize() strips time so all locations align
        dates = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        ).normalize().tz_localize(None)

        # Extract all variables by looping — not by hardcoded index
        data = {"date": dates}
        for j, var in enumerate(variables):
            data[f"{var}_{name}"] = daily.Variables(j).ValuesAsNumpy()

        location_dfs.append(pd.DataFrame(data).set_index("date"))

    # --- Combine all locations on the date index ---
    combined = pd.concat(location_dfs, axis=1)

    if save_path:
        combined.to_csv(save_path)
        print(f"\nSaved to: {save_path}")

    return combined



#####################
# 3. This function transforms the weather data into rolling aggregates that are more relevant for crop stress modeling,
#  such as 30-day rolling means for temperature and 30-day rolling sums for precipitation. 
# It saves the transformed data in the csv file with the specified name in the folder /data.
#####################

def compute_rolling_weather(weather_path, window=30, save_path=None):    
    """
    Compute backward-looking rolling aggregates of weather features.

    Uses agronomically appropriate aggregation for each variable type:
    - temperature, humidity, VPD, soil temp, wet bulb → rolling MEAN
    - precipitation, evapotranspiration, snowfall     → rolling SUM
    - soil moisture                                   → rolling MIN (worst-case stress)
    - wind speed max                                  → rolling MAX (storm damage)

    Parameters
    ----------
    weather_path : str
        Path to the raw daily weather CSV (as saved by fetch_weather_data).
    window : int
        Number of calendar days for the rolling window.
    save_path : str, optional
        If provided, saves the result to this CSV path.

    Returns
    -------
    pd.DataFrame
        Rolling aggregates with same index as input. First (window-1) rows will be NaN.
    """
    import pandas as pd

    # Allow passing a filepath directly
    weather_df = pd.read_csv(weather_path, index_col="date", parse_dates=True)

    # --- Define aggregation rules by variable keyword ---
    sum_keywords = ["precipitation_sum", "evapotranspiration_sum", "snowfall"]
    min_keywords = ["soil_moisture"]
    max_keywords = ["wind_speed_10m_max"]
    # Everything else gets mean (temperature, humidity, VPD, wet bulb, soil temp, etc.)

    def get_agg_method(col_name):
        if any(kw in col_name for kw in sum_keywords):
            return "sum"
        if any(kw in col_name for kw in min_keywords):
            return "min"
        if any(kw in col_name for kw in max_keywords):
            return "max"
        return "mean"

    # --- Compute rolling aggregates ---
    result = pd.DataFrame(index=weather_df.index)

    for col in weather_df.columns:
        agg = get_agg_method(col)
        rolled = weather_df[col].rolling(window=window, min_periods=window)

        if agg == "sum":
            result[f"{col}_roll{window}d_sum"] = rolled.sum()
        elif agg == "min":
            result[f"{col}_roll{window}d_min"] = rolled.min()
        elif agg == "max":
            result[f"{col}_roll{window}d_max"] = rolled.max()
        else:
            result[f"{col}_roll{window}d_mean"] = rolled.mean()

    result = result.iloc[window - 1:]  # Drop initial rows with insufficient data for rolling

    print(f"Window:          {window} days")
    print(f"Input columns:   {len(weather_df.columns)}")
    print(f"Output columns:  {len(result.columns)}")
    print(f"Date range:      {result.index.min()} to {result.index.max()}")
    print(f"Valid rows:      {result.dropna().shape[0]} / {len(result)}")

    if save_path:
        result.to_csv(save_path)
        print(f"Saved to:        {save_path}")

    return result



#####################
# 4. This function merges the weather and price data
#####################


def merge_price_weather(price_path, weather_path, save_path=None):
    """
    Merge price/spread data with rolling weather features on date index.

    Parameters
    ----------
    price_path : str
        Path to price/spread CSV (trading days only), e.g. merged_corn_soybean.csv
    weather_path : str
        Path to rolling weather CSV (calendar days), e.g. weather_rolled.csv
    save_path : str, optional
        If provided, saves the merged DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame
        Merged dataframe on trading days with all price and weather columns.
    """
    import pandas as pd

    prices = pd.read_csv(price_path, index_col="date", parse_dates=True)
    weather = pd.read_csv(weather_path, index_col="date", parse_dates=True)

    # Align weather to trading days, forward-fill weekends/holidays
    weather_aligned = weather.reindex(prices.index, method="ffill")

    merged = pd.concat([prices, weather_aligned], axis=1)

    print(f"Price columns:   {len(prices.columns)}")
    print(f"Weather columns: {len(weather.columns)}")
    print(f"Merged shape:    {merged.shape}")
    print(f"Date range:      {merged.index.min()} to {merged.index.max()}")
    print(f"Missing values:  {merged.isna().sum().sum()}")

    if save_path:
        merged.to_csv(save_path)
        print(f"Saved to:        {save_path}")

    return merged

def merge_weather_datasets(weather_dfs, merged_name, save_path=None):
    """
    Merge multiple weather DataFrames on date index.

    Each input DataFrame must either:
    - already have a DatetimeIndex (any name), OR
    - have a 'date' column that can be parsed to datetimes.

    Inputs are normalized to a DatetimeIndex named 'date' before concatenation,
    so the result is guaranteed to have a single 'date' index (no duplicate
    'date' columns leaking through). Concatenation uses an outer join on the
    date index to preserve all dates across inputs.

    Parameters
    ----------
    weather_dfs : list of pd.DataFrame
        Weather DataFrames to merge.
    merged_name : str
        Label used in printed diagnostics.
    save_path : str, optional
        If provided, saves the merged DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame
        Merged weather DataFrame indexed by date.
    """
    import pandas as pd

    def _normalize(df, i):
        # Already datetime-indexed: just make sure the index is named 'date'
        if isinstance(df.index, pd.DatetimeIndex):
            out = df.copy()
            out.index.name = "date"
            return out

        # Otherwise we need a 'date' column to promote to the index
        if "date" not in df.columns:
            raise ValueError(
                f"Input DataFrame at position {i} has neither a DatetimeIndex "
                f"nor a 'date' column. Columns found: {list(df.columns)[:5]}..."
            )

        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])
        out = out.set_index("date")

        # If a leftover unnamed index column got read from CSV (e.g. 'Unnamed: 0'),
        # drop it — it's just a stale RangeIndex.
        for junk in ("Unnamed: 0", "index"):
            if junk in out.columns:
                out = out.drop(columns=junk)

        return out

    normalized = [_normalize(df, i) for i, df in enumerate(weather_dfs)]

    # Outer join on the date index (concat default) so mismatched date ranges
    # surface as NaN rather than silently dropping rows.
    merged = pd.concat(normalized, axis=1)

    # Detect duplicate column names across locations (would indicate a bug upstream)
    dup_cols = merged.columns[merged.columns.duplicated()].tolist()
    if dup_cols:
        print(f"  WARNING: {len(dup_cols)} duplicate column names after merge: "
              f"{dup_cols[:3]}...")

    print(f"Merged '{merged_name}':")
    print(f"  Input datasets: {len(weather_dfs)}")
    print(f"  Output shape:   {merged.shape}")
    print(f"  Date range:     {merged.index.min()} to {merged.index.max()}")
    print(f"  Missing values: {merged.isna().sum().sum()}")

    if save_path:
        merged.to_csv(save_path)
        print(f"  Saved to:       {save_path}")

    return merged