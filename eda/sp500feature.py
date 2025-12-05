"""Non-temporal analysis of S&P 500 features and patterns."""

import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================================
# PARAMETERS - Configure analysis here
# ============================================================================
START_DATE = '2000-01-01'
END_DATE = '2025-10-31'
PLOT_WIDTH = 16  # inches
PLOT_HEIGHT = 10  # inches
BIN_WIDTH = 2.5  # Percent width of each bin
# ============================================================================

# Load S&P 500 data
print(f"Downloading S&P 500 data from {START_DATE} to {END_DATE}...")
sp500 = yf.download('^GSPC', start=START_DATE, end=END_DATE)

print(f"\nData loaded successfully!")
print(f"Total trading days: {len(sp500)}")
print(f"Date range: {sp500.index[0]} to {sp500.index[-1]}")

# Handle MultiIndex columns from yfinance
if isinstance(sp500.columns, tuple) or hasattr(sp500.columns, 'levels'):
    close_prices = sp500['Close'].squeeze()
else:
    close_prices = sp500['Close']

close_values = close_prices.values

# Calculate running maximum (previous high)
print("\nCalculating recovery times from drawdowns...")
running_max = np.maximum.accumulate(close_values)

# Calculate percent drawdown from high
percent_drawdown = ((close_values - running_max) / running_max) * 100

# Calculate days to recover to previous high for each day
days_to_recovery = np.full(len(close_values), np.nan)

for i in range(len(close_values)):
    if close_values[i] < running_max[i]:  # Currently in drawdown
        # Look forward to find when price reaches this high again
        target_high = running_max[i]
        for j in range(i + 1, len(close_values)):
            if close_values[j] >= target_high:
                days_to_recovery[i] = j - i
                break
        # If never recovered, leave as NaN

# Create bins for drawdown percentages
bin_edges = np.arange(0, -50 - BIN_WIDTH, -BIN_WIDTH)  # 0%, -2.5%, -5%, ..., -50%
bin_centers = bin_edges[:-1] - BIN_WIDTH / 2

# Bin the data
bin_indices = np.digitize(percent_drawdown, bin_edges[::-1]) - 1
bin_indices = len(bin_edges) - 2 - bin_indices  # Reverse to match our bin order

# Calculate average days to recovery for each bin
avg_days_by_bin = []
sample_counts = []

for bin_idx in range(len(bin_edges) - 1):
    mask = (bin_indices == bin_idx) & (~np.isnan(days_to_recovery))
    if np.sum(mask) > 0:
        avg_days = np.mean(days_to_recovery[mask])
        avg_days_by_bin.append(avg_days)
        sample_counts.append(np.sum(mask))
    else:
        avg_days_by_bin.append(0)
        sample_counts.append(0)

print(f"\nBinned {len([c for c in sample_counts if c > 0])} drawdown levels")
print(f"Total days in drawdown: {np.sum(~np.isnan(days_to_recovery))}")

# Create bar plot
fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))

# Filter out bins with no samples
valid_bins = [i for i, count in enumerate(sample_counts) if count > 0]
valid_centers = [bin_centers[i] for i in valid_bins]
valid_days = [avg_days_by_bin[i] for i in valid_bins]
valid_counts = [sample_counts[i] for i in valid_bins]

bars = ax.bar(valid_centers, valid_days, width=BIN_WIDTH * 0.8,
              color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)

# Add sample count labels on top of bars
for i, (center, days, count) in enumerate(zip(valid_centers, valid_days, valid_counts)):
    if count > 0:
        ax.text(center, days + max(valid_days) * 0.02, f'n={count}',
                ha='center', va='bottom', fontsize=8, color='darkgray')

ax.set_xlabel('Drawdown from Previous High (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Days to Recovery', fontsize=14, fontweight='bold')
ax.set_title(f'S&P 500: Recovery Time vs Drawdown Depth ({START_DATE} to {END_DATE})',
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linewidth=0.8)

# Set x-axis to show drawdown as negative percentages
ax.set_xlim(min(valid_centers) - BIN_WIDTH, max(valid_centers) + BIN_WIDTH)

plt.tight_layout()

# Save plot to outputs folder
OUTPUT_PATH = Path('outputs/sp500_recovery_analysis.png')
OUTPUT_PATH.parent.mkdir(exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {OUTPUT_PATH}")

# Print summary statistics
print("\n=== Summary Statistics ===")
print(f"Average recovery time overall: {np.nanmean(days_to_recovery):.1f} days")
print(f"Median recovery time overall: {np.nanmedian(days_to_recovery):.1f} days")
print(f"Max recovery time: {np.nanmax(days_to_recovery):.0f} days")
print("\nRecovery time by drawdown depth:")
for center, days, count in zip(valid_centers, valid_days, valid_counts):
    if count > 0:
        print(f"  {center:>6.1f}% drawdown: {days:>6.1f} days avg (n={count})")
