class MovingAverage:
    """Calculate moving average of price data."""

    def __init__(self, window=100):
        """
        Initialize MovingAverage qualifier.

        Args:
            window: Number of periods for moving average (default: 10)
        """
        self.window = window

    def calculate(self, data):
        """
        Calculate moving average from price data.

        Args:
            data: DataFrame with 'Close' column

        Returns:
            Series with same length as input data (NaN for initial window-1 periods)
        """
        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, tuple) or hasattr(data.columns, 'levels'):
            close_prices = data['Close'].squeeze()
        else:
            close_prices = data['Close']

        return close_prices.rolling(window=self.window).mean()

    def get_label(self):
        """Return label for plotting."""
        return f'{self.window}-Day Moving Average'
