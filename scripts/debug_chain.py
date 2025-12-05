#!/usr/bin/env python3
"""Debug script to inspect options chain data at each pipeline stage."""

import argparse
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd

from es_options.data_sources import EODClient
from es_options.processing.chain_cleaner import ChainCleaner
from es_options.processing.smoothing import CallPriceSmoother


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug options chain loading")
    parser.add_argument("--symbol", type=str, default="ES")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--expiry", type=int, default=0, help="Expiry index to inspect (0=nearest)")
    args = parser.parse_args()

    trade_date = date.fromisoformat(args.date) if args.date else date.today()

    # Step 1: Fetch raw chain
    print(f"\n{'='*60}")
    print(f"STEP 1: Fetching raw chain for {args.symbol} on {trade_date}")
    print(f"{'='*60}")

    client = EODClient()
    raw_chain = client.get_option_chain(args.symbol, trade_date)

    if raw_chain.empty:
        print("ERROR: No data returned from API")
        print("Check your EOD_API_KEY in .env")
        return

    print(f"Raw chain shape: {raw_chain.shape}")
    print(f"Columns: {list(raw_chain.columns)}")
    print(f"\nExpiries ({raw_chain['expiry'].nunique()}):")
    for exp in sorted(raw_chain["expiry"].unique())[:10]:
        count = len(raw_chain[raw_chain["expiry"] == exp])
        print(f"  {exp.date() if hasattr(exp, 'date') else exp}: {count} options")

    print(f"\nStrike range: {raw_chain['strike'].min():.0f} - {raw_chain['strike'].max():.0f}")
    print(f"\nSample raw data:")
    print(raw_chain.head(10).to_string())

    # Step 2: Clean chain
    print(f"\n{'='*60}")
    print("STEP 2: Cleaning chain")
    print(f"{'='*60}")

    cleaner = ChainCleaner()
    clean_chain = cleaner.clean(raw_chain, pd.Timestamp(trade_date))
    calls = cleaner.filter_calls(clean_chain)

    print(f"After cleaning: {len(clean_chain)} options ({len(raw_chain) - len(clean_chain)} removed)")
    print(f"Calls only: {len(calls)}")

    if calls.empty:
        print("ERROR: No calls after cleaning")
        print("\nFiltering stats for raw chain:")
        print(f"  Volume > 0: {(raw_chain['volume'] > 0).sum()}")
        print(f"  OI > 0: {(raw_chain['open_interest'] > 0).sum()}")
        print(f"  Bid > 0: {(raw_chain['bid'] > 0).sum()}")
        return

    # Step 3: Inspect a single expiry
    print(f"\n{'='*60}")
    print(f"STEP 3: Inspecting expiry index {args.expiry}")
    print(f"{'='*60}")

    expiries = sorted(calls["expiry"].unique())
    if args.expiry >= len(expiries):
        print(f"Only {len(expiries)} expiries available")
        return

    expiry = expiries[args.expiry]
    expiry_calls = calls[calls["expiry"] == expiry].sort_values("strike")

    print(f"Expiry: {expiry}")
    print(f"Calls for this expiry: {len(expiry_calls)}")
    print(f"\nStrike/Price data:")
    print(expiry_calls[["strike", "bid", "ask", "last", "volume", "open_interest"]].to_string())

    # Step 4: Smoothing check
    print(f"\n{'='*60}")
    print("STEP 4: Smoothing check")
    print(f"{'='*60}")

    spot = client.get_spot_price(args.symbol, trade_date)
    if spot is None:
        spot = expiry_calls["strike"].median()
        print(f"Could not fetch spot, using median strike: {spot:.2f}")
    else:
        print(f"Spot price: {spot:.2f}")

    smoother = CallPriceSmoother()
    try:
        smooth_result = smoother.fit(expiry_calls, spot)
        print(f"Smoothing successful!")
        print(f"Strike range: {smooth_result.strike_min:.0f} - {smooth_result.strike_max:.0f}")
        print(f"Spline knots: {len(smooth_result.spline.get_knots())}")
    except Exception as e:
        print(f"Smoothing failed: {e}")
        return

    # Step 5: Plot raw vs smoothed
    print(f"\n{'='*60}")
    print("STEP 5: Plotting raw vs smoothed prices")
    print(f"{'='*60}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{args.symbol} Options Debug - {trade_date} - Expiry {expiry.date() if hasattr(expiry, 'date') else expiry}")

    # Raw prices
    ax = axes[0, 0]
    mid_price = (expiry_calls["bid"] + expiry_calls["ask"]) / 2
    ax.scatter(expiry_calls["strike"], mid_price, alpha=0.6, label="Mid price")
    ax.scatter(expiry_calls["strike"], expiry_calls["last"], alpha=0.6, marker="x", label="Last")
    ax.axvline(spot, color="red", linestyle="--", label=f"Spot={spot:.0f}")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Price")
    ax.set_title("Raw Call Prices")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Smoothed prices
    ax = axes[0, 1]
    strike_grid = smooth_result.get_strike_grid(200)
    smooth_prices = smooth_result.smooth_fn(strike_grid)
    ax.plot(strike_grid, smooth_prices, "b-", linewidth=2, label="Smoothed")
    ax.scatter(expiry_calls["strike"], mid_price, alpha=0.4, s=20, label="Raw mid")
    ax.axvline(spot, color="red", linestyle="--")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Price")
    ax.set_title("Smoothed Call Prices")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Volume by strike
    ax = axes[1, 0]
    ax.bar(expiry_calls["strike"], expiry_calls["volume"], width=5, alpha=0.7)
    ax.axvline(spot, color="red", linestyle="--")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Volume")
    ax.set_title("Volume by Strike")
    ax.grid(True, alpha=0.3)

    # Open interest by strike
    ax = axes[1, 1]
    ax.bar(expiry_calls["strike"], expiry_calls["open_interest"], width=5, alpha=0.7, color="orange")
    ax.axvline(spot, color="red", linestyle="--")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Open Interest")
    ax.set_title("Open Interest by Strike")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nDone! Close the plot window to exit.")


if __name__ == "__main__":
    main()
