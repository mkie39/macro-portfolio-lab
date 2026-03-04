import pandas as pd
import numpy as np


def weight_by_signal(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts +1 / 0 / -1 signals into dollar-neutral portfolio weights.

    Each long position receives an equal share of +100% gross long exposure.
    Each short position receives an equal share of -100% gross short exposure.

    For a row with `L` longs and `S` shorts:
        long  weight = +1 / L   per long  position
        short weight = -1 / S   per short position
        zero  weight =  0       for flat  positions

    The portfolio is always dollar-neutral: sum(weights) = 0,
    total long weight = +1.0, total short weight = -1.0.

    Parameters
    ----------
    signals_df : DataFrame of +1 / 0 / -1 integer signals

    Returns
    -------
    DataFrame of float weights, same shape as signals_df.
    """
    weights = pd.DataFrame(0.0, index=signals_df.index, columns=signals_df.columns)

    n_long = (signals_df == 1).sum(axis=1).replace(0, np.nan)
    n_short = (signals_df == -1).sum(axis=1).replace(0, np.nan)

    # Broadcast: divide each +1 by the row's long count, each -1 by short count
    long_mask = signals_df == 1
    short_mask = signals_df == -1

    weights[long_mask] = long_mask[long_mask].div(n_long, axis=0)
    weights[short_mask] = short_mask[short_mask].mul(-1).div(n_short, axis=0)

    return weights.fillna(0.0)
