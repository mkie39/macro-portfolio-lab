# Macro Portfolio Lab

A Python-based research framework for constructing, backtesting, and analyzing multi-asset macro strategies. This project simulates an institutional-grade workflow, moving from raw data engineering to signal generation and portfolio optimization.

**Current Status:** Phase 1 (Data Infrastructure & Hygiene)

---

## 🚀 Key Features

### 1. Robust Data Pipeline (`src/data.py`)
- **Automated Ingestion:** Fetches adjusted OHLCV data for a multi-asset universe (Equities, Fixed Income, FX, Commodities) using `yfinance`.
- **Calendar Alignment:** Implements logic to handle **calendar fragmentation** (e.g., mismatches between NYSE and Eurex holidays).
- **Cleaner Logic:** Uses strictly defined forward-filling (`ffill` with 5-day limit) to preserve serial correlation without introducing look-ahead bias.

### 2. Research Notebooks (`notebooks/`)
- **01_data_quality.ipynb:** Visualizes "dirty" vs. "clean" data using heatmaps to identify missing data patterns (e.g., Thanksgiving vs. May Day). Includes rebased performance analysis.

---

## 📊 Asset Universe
The lab currently tracks a global macro basket to simulate real-world diversification and currency exposure:

| Ticker       | Asset Class           | Role in Portfolio                |
| :----------- | :-------------------- | :------------------------------- |
| **SPY**      | US Equities           | Global Growth / Risk-On          |
| **EZU**      | Eurozone Equities     | Home Market Bias (NL/EU context) |
| **IEF**      | US Treasuries (7-10Y) | Duration / Flight-to-Safety      |
| **GLD**      | Gold                  | Inflation / Real Rates Hedge     |
| **EURUSD=X** | FX Spot               | Currency Risk Management         |