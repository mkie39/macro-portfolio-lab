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
    rates_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Runs a backtest with end-of-month rebalancing and proportional transaction costs.

    Parameters
    ----------
    prices    : daily asset prices (normalised to USD per foreign unit for FX)
    signals   : +1 / 0 / -1 signal DataFrame (used for position direction reference)
    weights   : portfolio weights, same columns as prices
    tc_bps    : one-way transaction cost in basis points
    rates_df  : optional DataFrame of per-asset interest-rate differentials in % p.a.
                (same columns as weights, e.g. output of calculate_fx_carry_signal).
                When provided, daily interest income = weight * differential% / 100 / 360
                is added to the price return on every day.
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
    weight_changes = execution_weights.diff().fillna(0)
    turnover = weight_changes.abs().sum(axis=1)
    tc_decimal = tc_bps / 10000.0
    transaction_costs = turnover * tc_decimal

    # 5. Daily spot (price) return
    price_returns = (daily_returns * execution_weights).sum(axis=1)

    # 6. Daily interest income (carry accrual)
    # Each day accrues: weight_i * (foreign_rate_i - USD_rate_i) % / 100 / 360
    # Positive for long high-yielders, positive for short low-yielders.
    if rates_df is not None:
        aligned_rates = rates_df.reindex(prices.index).ffill()
        daily_carry = (execution_weights * aligned_rates / 100.0 / 360.0).sum(axis=1)
    else:
        daily_carry = pd.Series(0.0, index=prices.index)

    # 7. Gross = price + carry;  Net = gross - transaction costs
    gross_portfolio_returns = price_returns + daily_carry
    net_portfolio_returns = gross_portfolio_returns - transaction_costs

    # 8. Calculate Cumulative Equity
    cumulative_returns = np.exp(net_portfolio_returns.cumsum())

    return pd.DataFrame(
        {
            "Price_Return": price_returns,
            "Interest_Income": daily_carry,
            "Gross_Daily_Return": gross_portfolio_returns,
            "Net_Daily_Return": net_portfolio_returns,
            "Turnover": turnover,
            "Transaction_Costs": transaction_costs,
            "Cumulative_Equity": cumulative_returns,
        }
    )
