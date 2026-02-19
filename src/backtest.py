import pandas as pd
import numpy as np


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculates daily log returns."""
    return np.log(prices / prices.shift(1))


def run_backtest_ts_momentum(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    weights: pd.DataFrame,
    tc_bps: float = 10.0,
) -> pd.DataFrame:
    """
    Runs a backtest with end-of-month rebalancing and proportional transaction costs.
    """
    # 1. Calculate daily asset returns
    daily_returns = np.log(prices / prices.shift(1))

    # 2. Monthly Rebalancing: Isolate weights only on the last trading day of each month
    # Group by year and month, take the last row, then reindex to daily and forward-fill
    eom_weights = weights.groupby([weights.index.year, weights.index.month]).tail(1)
    daily_weights = eom_weights.reindex(prices.index).ffill()

    # 3. Shift weights by 1 period to prevent look-ahead bias
    execution_weights = daily_weights.shift(1)

    # 4. Calculate Turnover & Transaction Costs
    # diff() finds where weights changed. abs().sum() gets total portfolio % traded.
    weight_changes = execution_weights.diff().fillna(0)
    turnover = weight_changes.abs().sum(axis=1)

    # Convert basis points to a decimal (e.g., 10 bps = 0.0010)
    tc_decimal = tc_bps / 10000.0
    transaction_costs = turnover * tc_decimal

    # 5. Calculate Net Portfolio Returns
    gross_portfolio_returns = (daily_returns * execution_weights).sum(axis=1)
    net_portfolio_returns = gross_portfolio_returns - transaction_costs

    # 6. Calculate Cumulative Equity
    cumulative_returns = np.exp(net_portfolio_returns.cumsum())

    return pd.DataFrame(
        {
            "Gross_Daily_Return": gross_portfolio_returns,
            "Net_Daily_Return": net_portfolio_returns,
            "Turnover": turnover,
            "Transaction_Costs": transaction_costs,
            "Cumulative_Equity": cumulative_returns,
        }
    )
