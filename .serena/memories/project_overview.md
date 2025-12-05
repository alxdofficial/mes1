# MES1 Project Overview

## Purpose
This is a **financial analysis/EDA (Exploratory Data Analysis) project** focused on S&P 500 data analysis with various "qualifiers" that provide different analytical views of the market data.

## Structure

```
mes1/
├── eda/                    # Main analysis scripts
│   ├── sp500.py            # Main script - downloads S&P 500 data and plots with qualifiers
│   └── sp500feature.py     # Recovery analysis - drawdown vs days-to-recovery histogram
├── qualifiers/             # Pluggable analysis modules
│   ├── __init__.py         # Exports all qualifier classes
│   ├── moving_average.py   # MovingAverage - simple moving average overlay
│   ├── drawdown_days.py    # DrawdownDays - days since all-time high
│   ├── inflation_adjusted.py # InflationAdjusted - CPI-adjusted prices
│   ├── gold_adjusted.py    # GoldAdjusted - price in gold terms
│   └── adjusted_returns.py # AdjustedReturns - daily % returns comparison
├── outputs/                # Generated plots (PNG files)
│   ├── sp500_with_qualifiers.png
│   ├── sp500_1950_to_present.png
│   └── sp500_recovery_analysis.png
├── .env                    # Contains FRED_API_KEY for economic data
└── .venv/                  # Python virtual environment
```

## Key Components

### Qualifier Pattern
All qualifier classes follow a common interface:
- `__init__()` - Initialize with optional parameters
- `calculate(data)` - Process S&P 500 DataFrame, returns result
- `get_label()` - Return string label for plotting
- `plot(ax, dates, result)` - Optional custom plotting method
- `plot_type` attribute: 'overlay' (on main chart) or 'subplot' (separate panel)

### Dependencies
- `yfinance` - Download stock data (S&P 500 via ^GSPC ticker)
- `matplotlib` - Plotting
- `pandas` - Data manipulation
- `fredapi` - FRED economic data (requires API key for CPI data)

### Main Script Flow (eda/sp500.py)
1. Download S&P 500 data from yfinance for date range
2. Instantiate list of qualifiers
3. Calculate each qualifier's metrics
4. Create subplots: main price chart + subplot qualifiers
5. Overlay qualifiers go on main chart, subplot qualifiers get their own panels
6. Save to outputs/

## Qualifiers Summary

| Qualifier | Type | Description |
|-----------|------|-------------|
| MovingAverage | overlay | N-day simple moving average |
| DrawdownDays | subplot | Days since all-time high |
| InflationAdjusted | overlay | S&P 500 adjusted for CPI |
| GoldAdjusted | subplot | S&P 500 priced in gold |
| AdjustedReturns | subplot | Daily % returns: nominal, inflation-adj, gold-adj |
