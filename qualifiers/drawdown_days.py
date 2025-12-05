import numpy as np
import matplotlib.pyplot as plt


class DrawdownDays:
    """Track days below previous high and visualize with color-coded drawdown severity."""

    def __init__(self):
        """Initialize DrawdownDays qualifier."""
        pass

    def calculate(self, data):
        """
        Calculate number of days below previous high for each day.

        Args:
            data: DataFrame with 'Close' column

        Returns:
            Tuple of (days_below_high, percent_drawdown) Series with same length as input
        """
        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, tuple) or hasattr(data.columns, 'levels'):
            close_prices = data['Close'].squeeze()
        else:
            close_prices = data['Close']

        # Convert to numpy for faster iteration
        close_values = close_prices.values

        # Calculate running maximum (previous high)
        running_max = np.maximum.accumulate(close_values)

        # Calculate percent drawdown from high
        percent_drawdown = ((close_values - running_max) / running_max) * 100

        # Calculate days below previous high
        days_below_high = np.zeros(len(close_values))
        days_counter = 0

        for i in range(len(close_values)):
            if i == 0:
                days_below_high[i] = 0
                continue

            if close_values[i] < running_max[i]:
                days_counter += 1
            else:
                days_counter = 0

            days_below_high[i] = days_counter

        return days_below_high, percent_drawdown

    def plot(self, ax, dates, result):
        """
        Custom plotting method with color coding based on drawdown severity.

        Args:
            ax: Matplotlib axis to plot on
            dates: DatetimeIndex for x-axis
            result: Tuple of (days_below_high, percent_drawdown)
        """
        days_below_high, percent_drawdown = result

        # Create color map based on drawdown percentage
        # Green (0%) -> Yellow (-10%) -> Orange (-20%) -> Red (-30%+)
        colors = []
        for pct in percent_drawdown:
            if pct >= -5:
                colors.append('green')
            elif pct >= -10:
                colors.append('yellowgreen')
            elif pct >= -20:
                colors.append('orange')
            elif pct >= -30:
                colors.append('darkorange')
            else:
                colors.append('red')

        # Plot with color segments
        for i in range(1, len(dates)):
            ax.plot(dates[i-1:i+1], days_below_high[i-1:i+1],
                   color=colors[i], linewidth=1, alpha=0.8)

        ax.set_ylabel('Days Below High', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add color legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='< 5% drawdown'),
            Patch(facecolor='yellowgreen', label='5-10% drawdown'),
            Patch(facecolor='orange', label='10-20% drawdown'),
            Patch(facecolor='darkorange', label='20-30% drawdown'),
            Patch(facecolor='red', label='> 30% drawdown')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

    def get_label(self):
        """Return label for plotting."""
        return 'Days Below Previous High (Color-coded by Drawdown %)'
