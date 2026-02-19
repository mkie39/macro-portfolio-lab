import pandas as pd
import numpy as np
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")  # Suppresses convergence warnings from the solver


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


def calc_ewma_volatility(prices: pd.DataFrame, span: int = 60) -> pd.DataFrame:
    """
    Calculates annualized exponentially weighted moving average (EWMA) volatility.
    Reacts much faster to recent market shocks than simple rolling volatility.
    """
    daily_returns = prices.pct_change()

    # Calculate EWMA standard deviation and annualize (adjust=False matches standard financial modeling)
    ewma_vol = daily_returns.ewm(span=span, adjust=False).std() * np.sqrt(252)

    return ewma_vol


def calc_garch_volatility(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Calculates annualized rolling GARCH(1,1) volatility.
    """
    # GARCH models perform better when returns are scaled by 100
    returns = prices.pct_change().dropna() * 100
    garch_vol = pd.DataFrame(index=returns.index, columns=returns.columns)

    for col in returns.columns:
        asset_ret = returns[col]
        vol_series = pd.Series(index=asset_ret.index, dtype=float)

        # Rolling window estimation (computational bottleneck)
        for i in range(window, len(asset_ret)):
            train_data = asset_ret.iloc[i - window : i]
            # p=1 (lag variance), q=1 (lag squared return)
            am = arch_model(train_data, vol="Garch", p=1, q=1, rescale=False)
            res = am.fit(disp="off")

            # Forecast exactly 1 step ahead (next day's variance)
            forecast = res.forecast(horizon=1, reindex=False)
            vol_series.iloc[i] = np.sqrt(forecast.variance.iloc[-1, 0])

        garch_vol[col] = vol_series

    # Unscale the returns (divide by 100) and annualize
    garch_vol = (garch_vol / 100.0) * np.sqrt(252)

    # Reindex back to the original prices dataframe to handle the dropped NaN rows
    return garch_vol.reindex(prices.index).ffill()


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


def calc_ls_ts_momentum(
    prices: pd.DataFrame, lookback: int = 252, skip: int = 21
) -> pd.DataFrame:
    """
    Calculates Long/Short Time Series Momentum.
    """
    lagged_prices = prices.shift(skip)
    momentum_scores = (lagged_prices / prices.shift(lookback)) - 1

    # 1 (Long) if positive, -1 (Short) if negative
    signals = np.where(momentum_scores > 0, 1, -1)

    return pd.DataFrame(signals, index=prices.index, columns=prices.columns)


def calc_ls_inv_vol_weights(
    rolling_vol: pd.DataFrame, signals: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculates Long/Short inverse volatility weights targeting 100% Gross Exposure.
    """
    inv_vol = 1.0 / rolling_vol

    # Raw weights with direction (+ or -)
    raw_weights = inv_vol * signals

    # Normalize by the sum of absolute weights so total gross exposure is exactly 1.0 (100%)
    gross_exposure = raw_weights.abs().sum(axis=1)
    weights = raw_weights.div(gross_exposure, axis=0)

    return weights.fillna(0)


def calc_dual_momentum_signals(
    prices: pd.DataFrame, lookback: int = 252, skip: int = 21, top_n: int = 2
) -> pd.DataFrame:
    """
    Calculates Dual Momentum signals: Must be in the Top N AND have positive absolute momentum.
    """
    # 1. Calculate the 12-1 month momentum score
    lagged_prices = prices.shift(skip)
    momentum_scores = (lagged_prices / prices.shift(lookback)) - 1

    # 2. Cross-Sectional Gate: Rank assets daily (1 is highest momentum)
    # ascending=False means the highest return gets rank 1
    ranks = momentum_scores.rank(axis=1, ascending=False)

    # 3. Absolute Gate: Momentum must be > 0
    is_positive = momentum_scores > 0

    # 4. Combine: Must be in top_n AND positive
    # Returns boolean True/False, multiply by 1 to get 1s and 0s
    signals = ((ranks <= top_n) & is_positive) * 1

    return pd.DataFrame(signals, index=prices.index, columns=prices.columns)


def calc_smoothed_momentum_signals(
    prices: pd.DataFrame, ma_window: int = 200, top_n: int = 2
) -> pd.DataFrame:
    """
    Calculates Smoothed Dual Momentum using the distance from a Moving Average.
    """
    # 1. Calculate the long-term moving average
    sma = prices.rolling(window=ma_window).mean()

    # 2. Momentum Score: Percentage distance above/below the SMA
    momentum_scores = (prices / sma) - 1

    # 3. Cross-Sectional Gate: Rank assets by how far above their SMA they are
    ranks = momentum_scores.rank(axis=1, ascending=False)

    # 4. Absolute Gate: Price must be strictly above its SMA
    is_positive = momentum_scores > 0

    # 5. Combine: Top N assets that are also in a smoothed uptrend
    signals = ((ranks <= top_n) & is_positive) * 1

    return pd.DataFrame(signals, index=prices.index, columns=prices.columns).fillna(0)


def calc_residual_momentum_signals(
    prices: pd.DataFrame,
    benchmark_ticker: str = "SPY",
    lookback: int = 252,
    skip: int = 21,
    top_n: int = 2,
) -> pd.DataFrame:
    """
    Calculates Residual Momentum by stripping out the benchmark's Beta.
    """
    # 1. Daily log returns
    returns = np.log(prices / prices.shift(1))
    bench_returns = returns[benchmark_ticker]

    # 2. Rolling Beta calculation (Covariance / Variance)
    rolling_cov = returns.rolling(window=lookback).cov(bench_returns)
    rolling_var = bench_returns.rolling(window=lookback).var()
    beta = rolling_cov.div(rolling_var, axis=0)

    # 3. Calculate idiosyncratic residuals (Asset Return - (Beta * Benchmark Return))
    residuals = returns.sub(beta.multiply(bench_returns, axis=0), fill_value=0)

    # 4. Sum residuals over the lookback window, skipping the most recent month
    lagged_residuals = residuals.shift(skip)
    residual_mom_scores = lagged_residuals.rolling(window=lookback - skip).sum()

    # 5. Dual Gates: Rank the residuals, and ensure residual momentum is > 0
    ranks = residual_mom_scores.rank(axis=1, ascending=False)
    is_positive = residual_mom_scores > 0

    signals = ((ranks <= top_n) & is_positive) * 1

    return pd.DataFrame(signals, index=prices.index, columns=prices.columns).fillna(0)
