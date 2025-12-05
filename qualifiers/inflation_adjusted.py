import os
from fredapi import Fred
from dotenv import load_dotenv


class InflationAdjusted:
    """Calculate inflation-adjusted S&P 500 prices using CPI data."""

    def __init__(self, fred_api_key=None):
        """
        Initialize InflationAdjusted qualifier.

        Args:
            fred_api_key: FRED API key (optional - loads from .env if not provided)
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
        self.plot_type = 'overlay'  # Indicates this should overlay on main chart

    def calculate(self, data):
        """
        Calculate inflation-adjusted S&P 500 prices.

        Args:
            data: DataFrame with 'Close' column and DatetimeIndex

        Returns:
            Series with inflation-adjusted prices (in latest date's dollars)
        """
        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, tuple) or hasattr(data.columns, 'levels'):
            close_prices = data['Close'].squeeze()
        else:
            close_prices = data['Close']

        # Get CPI data from FRED
        print("  Fetching CPI data from FRED...")
        cpi_data = self.fred.get_series('CPIAUCSL')

        # Reindex CPI to match S&P 500 dates and forward-fill (CPI is monthly)
        cpi_aligned = cpi_data.reindex(close_prices.index, method='ffill')

        # Use latest CPI as reference (show in "today's dollars")
        latest_cpi = cpi_aligned.iloc[-1]

        # Calculate inflation-adjusted prices
        adjusted_prices = close_prices * (latest_cpi / cpi_aligned)

        return adjusted_prices

    def get_label(self):
        """Return label for plotting."""
        return 'Inflation-Adjusted S&P 500 (Real Dollars)'

    def get_color(self):
        """Return color for plotting."""
        return 'green'
