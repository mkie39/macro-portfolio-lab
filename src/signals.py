import pandas as pd
import numpy as np


def calc_ts_momentum(
    prices: pd.DataFrame, lookback: int = 252, skip: int = 21
) -> pd.DataFrame:
    """
    Calculates Time Series Momentum, dropping the most recent month.
    """
    # 1. Lag prices by 'skip' days to avoid the 1-month mean reversion trap
    lagged_prices = prices.shift(skip)

    # 2. Calculate return from the start of the lookback up to the lag point
    momentum_scores = (lagged_prices / prices.shift(lookback)) - 1

    # 3. Generate Signal: 1 (Long) if positive, 0 (Cash) if negative
    signals = np.where(momentum_scores > 0, 1, 0)

    # Return as a formatted DataFrame
    return pd.DataFrame(signals, index=prices.index, columns=prices.columns)


def calc_rolling_volatility(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Calculates annualized rolling volatility using daily returns.
    """
    # Calculate daily percentage returns
    daily_returns = prices.pct_change()

    # Calculate rolling standard deviation and annualize it (assuming 252 trading days)
    rolling_vol = daily_returns.rolling(window=window).std() * np.sqrt(252)

    return rolling_vol


def calc_inverse_vol_weights(
    rolling_vol: pd.DataFrame, signals: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculates inverse volatility weights, applied only to active signals.
    """
    # Invert the volatility (1 / vol)
    inv_vol = 1.0 / rolling_vol

    # Only apply weights where the time-series momentum signal is active (e.g., 1)
    active_inv_vol = inv_vol * signals

    # Normalize weights so they sum to 1.0 (100% exposure) across the row for that day
    weights = active_inv_vol.div(active_inv_vol.sum(axis=1), axis=0)

    # Fill NaNs with 0 (for days with no active signals or missing data)
    return weights.fillna(0)
