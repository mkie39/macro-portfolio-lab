import pandas as pd
import numpy as np
from pathlib import Path
import os
import yfinance as yf

# --- CONFIGURATION ---
# Define the root directory relative to this script
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"

print(f"Project Root: {ROOT_DIR}")
print(f"Data Directory: {DATA_DIR}")

# --- FUNCTIONS ---


def load_prices(filename: str) -> pd.DataFrame:
    """Loads price data from a CSV file in the data directory."""
    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {filename} in {DATA_DIR}")

    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df = df.sort_index()
    return df


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values and ensures data types are numeric."""
    # Force numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Forward Fill (limit 5 days)
    df_clean = df.ffill(limit=5)

    # Drop rows where everything is still missing (e.g. weekends/holidays)
    df_clean = df_clean.dropna(how="all")

    return df_clean


def download_data(tickers: list, start_date: str, end_date: str, filename: str):
    """Downloads historical data for a list of tickers and saves to CSV."""
    print(f"Downloading data for {tickers}...")

    # auto_adjust=True handles dividends/splits
    df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    # Flatten MultiIndex if necessary (yfinance update quirks)
    if isinstance(df.columns, pd.MultiIndex):
        # If 'Close' is in the top level, grab it
        if "Close" in df.columns.levels[0]:
            df = df["Close"]
        # Fallback: sometimes yf returns columns like (Ticker, 'Close')
        # We just want the Close prices.

    # Ensure data dir exists
    output_path = DATA_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path)
    print(f"Saved {len(df)} rows to {output_path}")
    return df


# --- EXECUTION ---

if __name__ == "__main__":
    # 1. Define your universe
    tickers = [
        "SPY",  # US Equities
        "EZU",  # Eurozone Equities
        "IEF",  # US Treasuries (7-10Y)
        "GLD",  # Gold
        "EURUSD=X",  # FX Rate
    ]

    # 2. Download Data
    download_data(
        tickers,
        start_date="2018-01-01",
        end_date="2025-12-31",
        filename="market_prices.csv",
    )

    # 3. Test Loader & Cleaner
    df = load_prices("market_prices.csv")
    df_clean = clean_prices(df)

    print("\n--- Data Head ---")
    print(df_clean.head())

    print("\n--- Data Info ---")
    print(df_clean.info())
