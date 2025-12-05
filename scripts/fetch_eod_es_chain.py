#!/usr/bin/env python3
"""Fetch ES options chain from EOD Historical Data."""

import argparse
from datetime import date

from es_options.data_sources import EODClient
from es_options.io import FileStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ES options chain")
    parser.add_argument("--symbol", type=str, default="ES")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    args = parser.parse_args()

    trade_date = date.fromisoformat(args.date) if args.date else date.today()
    print(f"Fetching {args.symbol} options chain for {trade_date}")

    client = EODClient()
    chain = client.get_option_chain(args.symbol, trade_date)

    if chain.empty:
        print("No options data returned")
        return

    file_store = FileStore(format=args.format)
    path = file_store.save_chain(chain, args.symbol, trade_date, stage="raw")

    print(f"Fetched {len(chain)} options")
    print(f"Expiries: {chain['expiry'].nunique()}")
    print(f"Strikes: {chain['strike'].min():.0f} - {chain['strike'].max():.0f}")
    print(f"Saved to: {path}")


if __name__ == "__main__":
    main()
