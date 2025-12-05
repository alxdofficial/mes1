import os
import numpy as np
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv


class AdjustedReturns:
    """Calculate and compare daily percent returns: nominal, inflation-adjusted, and gold-adjusted."""

    def __init__(self, fred_api_key=None, gold_ticker='GC=F'):
        """
        Initialize AdjustedReturns qualifier.

        Args:
            fred_api_key: FRED API key (optional - loads from .env if not provided)
            gold_ticker: Ticker symbol for gold (default: GC=F for gold futures)
        """
        if fred_api_key is None:
            # Load from .env file
            load_dotenv()
            fred_api_key = os.getenv('FRED_API_KEY')

        if fred_api_key is None:
            raise ValueError(
                "FRED API key required. Either pass it directly or set FRED_API_KEY in .env file. "
                "Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        self.fred = Fred(api_key=fred_api_key)
        self.gold_ticker = gold_ticker
        self.plot_type = 'subplot'  # Plots in separate panel below

    def calculate(self, data):
        """
        Calculate daily percent returns for nominal, inflation-adjusted, and gold-adjusted prices.

        Args:
            data: DataFrame with 'Close' column and DatetimeIndex

        Returns:
            Tuple of (nominal_returns, inflation_adjusted_returns, gold_adjusted_returns)
        """
        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, tuple) or hasattr(data.columns, 'levels'):
            close_prices = data['Close'].squeeze()
        else:
            close_prices = data['Close']

        # Get CPI data from FRED
        print("  Fetching CPI data from FRED...")
        cpi_data = self.fred.get_series('CPIAUCSL')
        cpi_aligned = cpi_data.reindex(close_prices.index, method='ffill')
        latest_cpi = cpi_aligned.iloc[-1]
        inflation_adjusted_prices = close_prices * (latest_cpi / cpi_aligned)

        # Get gold price data
        print(f"  Fetching gold price data ({self.gold_ticker})...")
        start_date = close_prices.index[0]
        end_date = close_prices.index[-1]
        gold_data = yf.download(self.gold_ticker, start=start_date, end=end_date, progress=False)

        if isinstance(gold_data.columns, tuple) or hasattr(gold_data.columns, 'levels'):
            gold_prices = gold_data['Close'].squeeze()
        else:
            gold_prices = gold_data['Close']

        gold_aligned = gold_prices.reindex(close_prices.index, method='ffill')
        gold_adjusted_prices = close_prices / gold_aligned

        # Calculate daily percent returns
        nominal_returns = close_prices.pct_change() * 100
        inflation_adjusted_returns = inflation_adjusted_prices.pct_change() * 100
        gold_adjusted_returns = gold_adjusted_prices.pct_change() * 100

        return (nominal_returns, inflation_adjusted_returns, gold_adjusted_returns)

    def plot(self, ax, dates, result):
        """
        Custom plotting method with overlapped percent returns.

        Args:
            ax: Matplotlib axis to plot on
            dates: DatetimeIndex for x-axis
            result: Tuple of (nominal_returns, inflation_adjusted_returns, gold_adjusted_returns)
        """
        nominal_returns, inflation_adjusted_returns, gold_adjusted_returns = result

        # Plot all three return series
        ax.plot(dates, nominal_returns, linewidth=0.8, color='blue', alpha=0.7, label='Nominal Returns')
        ax.plot(dates, inflation_adjusted_returns, linewidth=0.8, color='green', alpha=0.7, label='Inflation-Adjusted Returns')
        ax.plot(dates, gold_adjusted_returns, linewidth=0.8, color='gold', alpha=0.7, label='Gold-Adjusted Returns')

        # Add zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

        ax.set_ylabel('Daily Return (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)

    def get_label(self):
        """Return label for plotting."""
        return 'Daily Percent Returns (Nominal vs Inflation-Adjusted vs Gold-Adjusted)'
