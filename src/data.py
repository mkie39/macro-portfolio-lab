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

    # Forward Fill (reasonable limit of 5 days)
    df_clean = df.ffill(limit=5)

    # Drop rows where everything is still missing (e.g. weekends/holidays)
    df_clean = df_clean.dropna()

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
        # Just want the Close prices.

    # Ensure data dir exists
    output_path = DATA_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path)
    print(f"Saved {len(df)} rows to {output_path}")
    return df


FX_TICKERS = [
    "AUDUSD=X",
    "NZDUSD=X",
    "GBPUSD=X",
    "EURUSD=X",
    "USDJPY=X",
    "USDCHF=X",
    "USDCAD=X",
    "USDMXN=X",
    "USDBRL=X",
]


def download_fx_data(start_date: str, end_date: str, filename: str) -> pd.DataFrame:
    """Downloads FX spot rates via yfinance and saves to CSV with cleaned column names."""
    print(f"Downloading FX spot rates...")
    df = yf.download(FX_TICKERS, start=start_date, end=end_date, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df = df["Close"]

    # Strip "=X" suffix from column names
    df.columns = [col.replace("=X", "") for col in df.columns]

    output_path = DATA_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"Saved {len(df)} rows to {output_path}")
    return df


def create_manual_rates(filename: str) -> pd.DataFrame:
    """Creates monthly central bank policy rates from 2018-01 to 2024-12 and saves to CSV."""
    # Each entry: (year, month, rate)
    rate_schedule = {
        "USD": [
            (2018, 1, 1.50),
            (2018, 6, 2.00),
            (2018, 12, 2.50),
            (2019, 8, 2.25),
            (2019, 12, 1.75),
            (2020, 3, 0.25),
            (2022, 3, 0.50),
            (2022, 6, 1.75),
            (2022, 9, 3.25),
            (2022, 12, 4.50),
            (2023, 2, 4.75),
            (2023, 5, 5.25),
            (2024, 12, 5.25),
        ],
        "AUD": [
            (2018, 1, 1.50),
            (2020, 3, 0.25),
            (2020, 9, 0.10),
            (2022, 5, 0.85),
            (2022, 9, 2.35),
            (2022, 12, 3.10),
            (2023, 6, 4.10),
            (2023, 11, 4.35),
            (2024, 12, 4.35),
        ],
        "NZD": [
            (2018, 1, 1.75),
            (2020, 3, 0.25),
            (2021, 10, 0.50),
            (2022, 2, 1.00),
            (2022, 7, 2.50),
            (2022, 11, 4.25),
            (2023, 5, 5.50),
            (2024, 12, 4.25),
        ],
        "GBP": [
            (2018, 1, 0.50),
            (2018, 8, 0.75),
            (2020, 3, 0.10),
            (2021, 12, 0.25),
            (2022, 2, 0.50),
            (2022, 6, 1.25),
            (2022, 11, 3.00),
            (2023, 2, 4.00),
            (2023, 8, 5.25),
            (2024, 12, 4.75),
        ],
        "EUR": [
            (2018, 1, -0.40),
            (2022, 7, 0.00),
            (2022, 9, 1.25),
            (2022, 12, 2.50),
            (2023, 3, 3.50),
            (2023, 9, 4.00),
            (2024, 6, 3.75),
            (2024, 12, 3.00),
        ],
        "JPY": [
            (2018, 1, -0.10),
            (2024, 3, 0.10),
            (2024, 12, 0.25),
        ],
        "CHF": [
            (2018, 1, -0.75),
            (2022, 6, -0.25),
            (2022, 9, 0.50),
            (2023, 3, 1.50),
            (2023, 6, 1.75),
            (2024, 3, 1.50),
            (2024, 6, 1.25),
            (2024, 12, 0.50),
        ],
        "CAD": [
            (2018, 1, 1.25),
            (2018, 7, 1.50),
            (2020, 3, 0.25),
            (2022, 3, 0.50),
            (2022, 6, 1.50),
            (2022, 9, 3.25),
            (2022, 12, 4.25),
            (2023, 7, 5.00),
            (2024, 12, 3.25),
        ],
        "MXN": [
            (2018, 1, 7.25),
            (2018, 9, 7.75),
            (2019, 9, 7.25),
            (2020, 6, 5.00),
            (2020, 12, 4.25),
            (2021, 12, 5.50),
            (2022, 6, 7.75),
            (2022, 12, 10.50),
            (2023, 3, 11.25),
            (2024, 12, 10.00),
        ],
        "BRL": [
            (2018, 1, 6.50),
            (2020, 8, 2.00),
            (2021, 3, 2.75),
            (2021, 9, 6.25),
            (2021, 12, 9.25),
            (2022, 3, 11.75),
            (2022, 8, 13.75),
            (2023, 6, 13.75),
            (2023, 12, 11.75),
            (2024, 9, 10.75),
            (2024, 12, 12.25),
        ],
    }

    date_index = pd.date_range("2018-01-01", "2024-12-31", freq="MS")
    df = pd.DataFrame(index=date_index)

    for currency, schedule in rate_schedule.items():
        series = pd.Series(
            {pd.Timestamp(y, m, 1): r for y, m, r in schedule}
        ).sort_index()
        df[currency] = series.reindex(date_index).ffill()

    df.index.name = "Date"
    output_path = DATA_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"Saved manual rates ({len(df)} rows) to {output_path}")
    return df


def align_fx_data(spot_df: pd.DataFrame, rates_df: pd.DataFrame) -> dict:
    """Forward-fills monthly rates to daily frequency and aligns both dataframes on date index."""
    # Reindex rates to the spot date range, forward-filling to daily frequency
    daily_index = spot_df.index
    rates_daily = rates_df.reindex(daily_index).ffill()

    # Align on common dates
    common_idx = spot_df.index.intersection(rates_daily.index)
    spot_aligned = spot_df.loc[common_idx]
    rates_aligned = rates_daily.loc[common_idx]

    return {"spot": spot_aligned, "rates": rates_aligned}


# --- EXECUTION ---

if __name__ == "__main__":
    # 1. Define your universe
    tickers = [
        # Broad Equities
        "SPY",
        "QQQ",
        "IWM",
        "EFA",
        "EEM",
        # US Sectors
        "XLK",
        "XLF",
        "XLV",
        "XLE",
        # Fixed Income & Hard Assets
        "TLT",
        "IEF",
        "LQD",
        "GLD",
        "GSG",
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

    # --- FX Carry Data ---
    START, END = "2018-01-01", "2024-12-31"

    spot_df = download_fx_data(START, END, filename="fx_spot.csv")
    rates_df = create_manual_rates(filename="fx_rates_manual.csv")
    aligned = align_fx_data(spot_df, rates_df)

    print("\n--- FX Spot Head ---")
    print(aligned["spot"].head())

    print("\n--- FX Rates (daily) Head ---")
    print(aligned["rates"].head())
