import yfinance as yf


class GoldAdjusted:
    """Calculate S&P 500 priced in gold (ounces)."""

    def __init__(self, gold_ticker='GC=F'):
        """
        Initialize GoldAdjusted qualifier.

        Args:
            gold_ticker: Ticker symbol for gold (default: GC=F for gold futures)
        """
        self.gold_ticker = gold_ticker
        self.plot_type = 'overlay'  # Indicates this should overlay on main chart

    def calculate(self, data):
        """
        Calculate S&P 500 priced in gold ounces.

        Args:
            data: DataFrame with 'Close' column and DatetimeIndex

        Returns:
            Series with S&P 500 prices divided by gold prices (in ounces)
        """
        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, tuple) or hasattr(data.columns, 'levels'):
            close_prices = data['Close'].squeeze()
        else:
            close_prices = data['Close']

        # Get gold price data
        print(f"  Fetching gold price data ({self.gold_ticker})...")
        start_date = close_prices.index[0]
        end_date = close_prices.index[-1]
        gold_data = yf.download(self.gold_ticker, start=start_date, end=end_date, progress=False)

        # Handle MultiIndex columns from yfinance
        if isinstance(gold_data.columns, tuple) or hasattr(gold_data.columns, 'levels'):
            gold_prices = gold_data['Close'].squeeze()
        else:
            gold_prices = gold_data['Close']

        # Align gold prices with S&P 500 dates (forward fill for missing dates)
        gold_aligned = gold_prices.reindex(close_prices.index, method='ffill')

        # Calculate S&P 500 in gold ounces
        sp500_in_gold = close_prices / gold_aligned

        return sp500_in_gold

    def get_label(self):
        """Return label for plotting."""
        return 'S&P 500 Priced in Gold (oz)'

    def get_color(self):
        """Return color for plotting."""
        return 'gold'
