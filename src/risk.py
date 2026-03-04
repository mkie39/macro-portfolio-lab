import pandas as pd
import numpy as np


def calculate_vol_scaled_weights(
    prices_df: pd.DataFrame,
    signal_weights_df: pd.DataFrame,
    target_vol: float = 0.10,
    vol_window: int = 60,
) -> pd.DataFrame:
    """
    Converts directional carry weights into inverse-volatility-scaled, dollar-neutral weights,
    then applies portfolio-level volatility scaling to hit a target annualised volatility.

    Parameters
    ----------
    prices_df         : daily asset prices (normalised to USD per foreign unit for FX)
    signal_weights_df : equal-weighted carry weights (+1/n, 0, -1/n) from weight_by_signal.
                        Only the *sign* (direction) is used; magnitudes are replaced.
    target_vol        : desired annualised portfolio volatility (default 10%).
                        Pass None to skip portfolio-level scaling and return
                        the normalised inverse-vol weights directly.
    vol_window        : lookback window in trading days for rolling volatility (default 60).

    Returns
    -------
    DataFrame of float weights, same shape as signal_weights_df.
    Dollar-neutral by construction: sum(weights, axis=1) ≈ 0,
    with gross exposure scaled to deliver approximately target_vol.

    Design notes
    ------------
    Step 1 — Per-asset rolling vol
        Annualised 60-day rolling σ from log returns.

    Step 2 — Inverse-vol weights
        Active longs  : raw_weight =  1 / σ_i
        Active shorts : raw_weight = -1 / σ_i
        Inactive      : 0
        Low-vol assets therefore get *larger* absolute weights so every position
        contributes roughly the same amount of daily P&L volatility.

    Step 3 — Dollar-neutral normalisation
        Long  side: divide by Σ(1/σ_i) for all longs  → long  weights sum to +1.0
        Short side: divide by Σ(1/σ_i) for all shorts → short weights sum to -1.0
        Net exposure = 0 (dollar neutral) ✓

    Step 4 — Portfolio-level vol targeting  (optional, requires target_vol ≠ None)
        Compute the 60-day rolling vol of the inv-vol-weighted portfolio returns.
        Scale today's weights by  target_vol / lagged_portfolio_vol.
        The lag of 1 ensures there is no look-ahead bias.
        The scalar is clipped to [0.25, 4.0] to prevent excessive leverage or
        near-zero exposure in extreme vol regimes.
    """
    log_returns = np.log(prices_df / prices_df.shift(1))

    # ── Step 1: per-asset 60-day rolling annualised volatility ───────────────
    asset_vol = log_returns.rolling(vol_window).std() * np.sqrt(252)
    asset_vol = asset_vol.replace(0.0, np.nan)

    # ── Step 2: inverse-vol raw weights, split by direction ──────────────────
    direction = np.sign(
        signal_weights_df
    )  # extracts +1 / 0 / -1 from any float weights
    inv_vol = 1.0 / asset_vol

    long_inv = inv_vol.where(
        direction > 0, 0.0
    )  # positive at long  positions, 0 elsewhere
    short_inv = inv_vol.where(
        direction < 0, 0.0
    )  # positive at short positions, 0 elsewhere

    # ── Step 3: normalise to dollar-neutral (+1 long leg, -1 short leg) ──────
    long_sum = long_inv.sum(axis=1).replace(0.0, np.nan)
    short_sum = short_inv.sum(axis=1).replace(0.0, np.nan)

    # long_inv / long_sum  → positive weights summing to +1
    # short_inv / short_sum → positive values summing to +1, then negated → -1
    vol_weights = long_inv.div(long_sum, axis=0) - short_inv.div(short_sum, axis=0)

    # ── Step 4: portfolio-level vol scaling to target_vol ────────────────────
    if target_vol is not None:
        # Fictitious portfolio returns using current inv-vol weights (for vol estimation)
        port_returns = (log_returns * vol_weights).sum(axis=1)
        port_vol_60d = port_returns.rolling(vol_window).std() * np.sqrt(252)

        # Shift by 1: vol estimate uses data through yesterday → no look-ahead
        vol_scalar = (target_vol / port_vol_60d).shift(1).clip(0.25, 4.0)
        vol_weights = vol_weights.mul(vol_scalar, axis=0)

    return vol_weights.fillna(0.0)
