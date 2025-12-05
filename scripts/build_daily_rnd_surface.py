#!/usr/bin/env python3
"""Build RND surface for ES options."""

import argparse
from datetime import date
from pathlib import Path

from es_options.data_sources import EODClient
from es_options.io import FileStore
from es_options.pipelines import RNDPipeline
from es_options.viz import DiagnosticsPlotter, RNDSurfacePlotter


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RND surface from ES options")
    parser.add_argument("--symbol", type=str, default="ES")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--rate", type=float, default=0.05)
    args = parser.parse_args()

    trade_date = date.fromisoformat(args.date) if args.date else date.today()
    print(f"Building RND surface for {args.symbol} on {trade_date}")

    client = EODClient()
    file_store = FileStore()
    pipeline = RNDPipeline(client=client, file_store=file_store, r=args.rate)

    result = pipeline.run(args.symbol, trade_date)

    if not result.success:
        print(f"Pipeline failed: {result.errors}")
        return

    print(f"\nExpiries processed: {result.num_expiries}")
    for rnd in result.rnd_results:
        dte = int(rnd.time_to_expiry * 365)
        print(f"  {rnd.expiry} ({dte}d): mean={rnd.mean:.1f}, std={rnd.std:.1f}, skew={rnd.skewness:.4f}")

    if args.plot and result.rnd_results:
        output_dir = file_store.base_path / "processed" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)

        plotter = RNDSurfacePlotter()

        # 3D surface (smoothed)
        plotter.plot_from_results(
            result.rnd_results,
            title=f"RND Surface - {args.symbol} - {trade_date}",
            save_path=output_dir / f"rnd_surface_{args.symbol}_{trade_date}.png",
            smooth_sigma=2.5,
        )

        # Heatmap with price on y-axis and dates on x-axis
        plotter.plot_density_heatmap(
            result.rnd_results,
            trade_date=trade_date,
            title=f"Market-Implied Price Distribution - {args.symbol} - {trade_date}",
            save_path=output_dir / f"rnd_heatmap_{args.symbol}_{trade_date}.png",
        )

        # Diagnostics for first expiry
        DiagnosticsPlotter().plot_rnd_sanity_check(
            result.rnd_results[0],
            save_path=output_dir / f"rnd_diagnostics_{args.symbol}_{trade_date}.png",
        )
        print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
