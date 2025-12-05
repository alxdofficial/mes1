import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.dates import MonthLocator, DateFormatter
from qualifiers import MovingAverage, DrawdownDays, InflationAdjusted, GoldAdjusted, AdjustedReturns

# ============================================================================
# PARAMETERS - Configure analysis here
# ============================================================================
START_DATE = '2000-01-01'
END_DATE = '2025-10-31'
PLOT_WIDTH = 40  # inches
PLOT_HEIGHT_PER_SUBPLOT = 8  # inches
# ============================================================================

# Load S&P 500 data
print(f"Downloading S&P 500 data from {START_DATE} to {END_DATE}...")
sp500 = yf.download('^GSPC', start=START_DATE, end=END_DATE)

print(f"\nData loaded successfully!")
print(f"Total trading days: {len(sp500)}")
print(f"Date range: {sp500.index[0]} to {sp500.index[-1]}")
print(f"\nFirst few rows:")
print(sp500.head())
print(f"\nLast few rows:")
print(sp500.tail())

# Instantiate qualifiers
qualifiers = [
    AdjustedReturns(),  # Plots daily % returns: nominal, inflation-adjusted, gold-adjusted
    # MovingAverage(window=100),
    DrawdownDays(),
]

# Calculate qualifier metrics
print("\nCalculating qualifiers...")
overlay_results = []
subplot_results = []

for qualifier in qualifiers:
    result = qualifier.calculate(sp500)
    plot_type = getattr(qualifier, 'plot_type', 'subplot')

    if plot_type == 'overlay':
        overlay_results.append((qualifier, qualifier.get_label(), result))
    else:
        subplot_results.append((qualifier, qualifier.get_label(), result))

    print(f"  - {qualifier.get_label()}")

# Create subplots: main price + subplot qualifiers only
num_plots = 1 + len(subplot_results)
fig, axes = plt.subplots(num_plots, 1, figsize=(PLOT_WIDTH, PLOT_HEIGHT_PER_SUBPLOT * num_plots), sharex=True)

# Handle single subplot case
if num_plots == 1:
    axes = [axes]

# Plot main S&P 500 price
axes[0].plot(sp500.index, sp500['Close'], linewidth=1, color='blue', alpha=0.8, label='S&P 500 (Nominal)')

# Plot overlay qualifiers on main chart
for qualifier, label, result in overlay_results:
    color = getattr(qualifier, 'get_color', lambda: 'red')()
    axes[0].plot(sp500.index, result, linewidth=1, color=color, alpha=0.7, label=label, linestyle='--')

axes[0].set_title(f'S&P 500 Index ({START_DATE} - {END_DATE})', fontsize=16, fontweight='bold')
axes[0].set_ylabel('Price', fontsize=12)
axes[0].grid(True, alpha=0.3, which='both')
axes[0].legend(loc='upper left', fontsize=10)

# Add monthly grid lines and year labels
from matplotlib.dates import YearLocator
axes[0].xaxis.set_minor_locator(MonthLocator(interval=1))
axes[0].xaxis.set_major_locator(YearLocator())
axes[0].xaxis.set_major_formatter(DateFormatter('%Y'))
axes[0].grid(True, which='minor', axis='x', color='lightgrey', alpha=0.5, linestyle='-', linewidth=0.5)
axes[0].tick_params(axis='x', colors='grey')

# Plot each subplot qualifier underneath
for idx, (qualifier, label, result) in enumerate(subplot_results, start=1):
    axes[idx].set_title(label, fontsize=14, fontweight='bold')

    # Check if qualifier has custom plot method
    if hasattr(qualifier, 'plot'):
        qualifier.plot(axes[idx], sp500.index, result)
    else:
        # Default plotting
        axes[idx].plot(sp500.index, result, linewidth=1, color='red', alpha=0.8)
        axes[idx].set_ylabel('Value', fontsize=12)
        axes[idx].grid(True, alpha=0.3, which='both')

    # Add monthly grid lines
    axes[idx].xaxis.set_minor_locator(MonthLocator(interval=1))
    axes[idx].xaxis.set_major_locator(YearLocator())
    axes[idx].xaxis.set_major_formatter(DateFormatter('%Y'))
    axes[idx].grid(True, which='minor', axis='x', color='lightgrey', alpha=0.5, linestyle='-', linewidth=0.5)
    axes[idx].tick_params(axis='x', colors='grey')

# Set x-label on bottom plot
axes[-1].set_xlabel('Date', fontsize=12)

# Configure x-axis labels
for ax in axes:
    ax.tick_params(axis='x', rotation=0, labelsize=8, colors='grey')

plt.tight_layout()

# Save plot to outputs folder
OUTPUT_PATH = Path('outputs/sp500_with_qualifiers.png')
OUTPUT_PATH.parent.mkdir(exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {OUTPUT_PATH}")
