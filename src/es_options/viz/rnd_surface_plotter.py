"""3D Risk-Neutral Density surface plotting."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

if TYPE_CHECKING:
    from es_options.metrics.rnd_extractor import RNDResult


class RNDSurfacePlotter:
    """Plot 3D RND surfaces."""

    def __init__(self, figsize: tuple[int, int] = (14, 10)):
        self.figsize = figsize

    def plot_from_results(
        self,
        rnd_results: list[RNDResult],
        title: str = "Risk-Neutral Density Surface",
        save_path: Path | str | None = None,
        smooth_sigma: float = 2.0,
    ) -> plt.Figure:
        """Plot 3D surface from list of RNDResult objects."""
        if not rnd_results:
            raise ValueError("No RND results to plot")

        spot = rnd_results[0].spot_price

        # Create a common strike grid centered around spot
        all_strikes = np.concatenate([r.strikes for r in rnd_results])
        strike_min = max(spot * 0.7, all_strikes.min())
        strike_max = min(spot * 1.3, all_strikes.max())
        strike_grid = np.linspace(strike_min, strike_max, 200)

        # Get all DTEs
        dtes = sorted([int(r.time_to_expiry * 365) for r in rnd_results])
        dte_grid = np.linspace(min(dtes), max(dtes), 100)

        # Build interpolated surface
        X, Y = np.meshgrid(strike_grid, dte_grid)
        Z = np.zeros_like(X)

        for i, dte in enumerate(dte_grid):
            # Find two nearest expiries to interpolate
            rnd_by_dte = {int(r.time_to_expiry * 365): r for r in rnd_results}
            actual_dtes = sorted(rnd_by_dte.keys())

            if dte <= actual_dtes[0]:
                rnd = rnd_by_dte[actual_dtes[0]]
                Z[i, :] = np.interp(strike_grid, rnd.strikes, rnd.density)
            elif dte >= actual_dtes[-1]:
                rnd = rnd_by_dte[actual_dtes[-1]]
                Z[i, :] = np.interp(strike_grid, rnd.strikes, rnd.density)
            else:
                # Linear interpolation between two expiries
                lower_dte = max(d for d in actual_dtes if d <= dte)
                upper_dte = min(d for d in actual_dtes if d >= dte)
                if lower_dte == upper_dte:
                    rnd = rnd_by_dte[lower_dte]
                    Z[i, :] = np.interp(strike_grid, rnd.strikes, rnd.density)
                else:
                    weight = (dte - lower_dte) / (upper_dte - lower_dte)
                    rnd_low = rnd_by_dte[lower_dte]
                    rnd_high = rnd_by_dte[upper_dte]
                    z_low = np.interp(strike_grid, rnd_low.strikes, rnd_low.density)
                    z_high = np.interp(strike_grid, rnd_high.strikes, rnd_high.density)
                    Z[i, :] = (1 - weight) * z_low + weight * z_high

        # Apply Gaussian smoothing for terrain-like appearance
        Z = gaussian_filter(Z, sigma=smooth_sigma)

        # Plot
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            X, Y, Z,
            cmap="viridis",
            alpha=0.9,
            linewidth=0,
            antialiased=True,
            rstride=2,
            cstride=2,
        )
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Density")

        # Spot price line
        ax.plot(
            [spot, spot],
            [dte_grid.min(), dte_grid.max()],
            [0, 0],
            "r--", linewidth=2, label=f"Spot = {spot:.0f}"
        )
        ax.legend()

        ax.set_xlabel("Strike")
        ax.set_ylabel("Days to Expiry")
        ax.set_zlabel("Density")
        ax.set_title(title)
        ax.view_init(elev=30, azim=-45)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_density_heatmap(
        self,
        rnd_results: list[RNDResult],
        trade_date: date,
        title: str = "Market-Implied Price Distribution",
        save_path: Path | str | None = None,
        spot_history: pd.DataFrame | None = None,
    ) -> plt.Figure:
        """
        Plot RND as heatmap with dates on x-axis and price on y-axis.

        Args:
            rnd_results: List of RND results
            trade_date: Current trade date
            title: Plot title
            save_path: Path to save figure
            spot_history: DataFrame with 'date' and 'close' columns for historical prices
        """
        if not rnd_results:
            raise ValueError("No RND results to plot")

        spot = rnd_results[0].spot_price

        # Create price grid (y-axis)
        all_strikes = np.concatenate([r.strikes for r in rnd_results])
        price_min = max(spot * 0.75, all_strikes.min())
        price_max = min(spot * 1.25, all_strikes.max())
        price_grid = np.linspace(price_min, price_max, 200)

        # Create date grid (x-axis) - from trade_date into future
        max_dte = max(int(r.time_to_expiry * 365) for r in rnd_results)
        date_grid = [trade_date + timedelta(days=d) for d in range(max_dte + 1)]
        num_dates = len(date_grid)

        # Build density matrix
        Z = np.zeros((len(price_grid), num_dates))

        rnd_by_dte = {int(r.time_to_expiry * 365): r for r in rnd_results}
        actual_dtes = sorted(rnd_by_dte.keys())

        for i, d in enumerate(range(num_dates)):
            if d <= actual_dtes[0]:
                rnd = rnd_by_dte[actual_dtes[0]]
                Z[:, i] = np.interp(price_grid, rnd.strikes, rnd.density)
            elif d >= actual_dtes[-1]:
                rnd = rnd_by_dte[actual_dtes[-1]]
                Z[:, i] = np.interp(price_grid, rnd.strikes, rnd.density)
            else:
                lower_dte = max(dte for dte in actual_dtes if dte <= d)
                upper_dte = min(dte for dte in actual_dtes if dte >= d)
                if lower_dte == upper_dte:
                    rnd = rnd_by_dte[lower_dte]
                    Z[:, i] = np.interp(price_grid, rnd.strikes, rnd.density)
                else:
                    weight = (d - lower_dte) / (upper_dte - lower_dte)
                    rnd_low = rnd_by_dte[lower_dte]
                    rnd_high = rnd_by_dte[upper_dte]
                    z_low = np.interp(price_grid, rnd_low.strikes, rnd_low.density)
                    z_high = np.interp(price_grid, rnd_high.strikes, rnd_high.density)
                    Z[:, i] = (1 - weight) * z_low + weight * z_high

        # Smooth
        Z = gaussian_filter(Z, sigma=1.5)

        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Heatmap
        extent = [0, num_dates, price_min, price_max]
        im = ax.imshow(
            Z,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="YlOrRd",
            alpha=0.8,
        )
        cbar = fig.colorbar(im, ax=ax, label="Probability Density")

        # Current spot line
        ax.axhline(spot, color="blue", linewidth=2, linestyle="-", label=f"Current Spot = {spot:.0f}")

        # Plot quantile bands
        q05 = []
        q25 = []
        q50 = []
        q75 = []
        q95 = []

        for i in range(num_dates):
            density = Z[:, i]
            if density.sum() > 0:
                density_norm = density / density.sum()
                cdf = np.cumsum(density_norm)
                q05.append(np.interp(0.05, cdf, price_grid))
                q25.append(np.interp(0.25, cdf, price_grid))
                q50.append(np.interp(0.50, cdf, price_grid))
                q75.append(np.interp(0.75, cdf, price_grid))
                q95.append(np.interp(0.95, cdf, price_grid))
            else:
                q05.append(np.nan)
                q25.append(np.nan)
                q50.append(np.nan)
                q75.append(np.nan)
                q95.append(np.nan)

        days = np.arange(num_dates)
        ax.fill_between(days, q05, q95, alpha=0.2, color="blue", label="90% CI")
        ax.fill_between(days, q25, q75, alpha=0.3, color="blue", label="50% CI")
        ax.plot(days, q50, "b--", linewidth=1.5, label="Median")

        # Historical spot if provided
        if spot_history is not None and not spot_history.empty:
            hist = spot_history[spot_history["date"] <= trade_date].tail(60)
            if not hist.empty:
                # Plot to the left (negative days)
                hist_days = [(d - trade_date).days for d in hist["date"]]
                ax.plot(hist_days, hist["close"], "k-", linewidth=2, label="Historical")

        # X-axis labels (show some dates)
        tick_positions = np.linspace(0, num_dates - 1, 6).astype(int)
        tick_labels = [date_grid[i].strftime("%b %d") for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.set_title(title)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_single_expiry(
        self,
        rnd_result: RNDResult,
        title: str | None = None,
        save_path: Path | str | None = None,
    ) -> plt.Figure:
        """Plot RND for a single expiry (2D)."""
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.fill_between(rnd_result.strikes, rnd_result.density, alpha=0.3)
        ax.plot(rnd_result.strikes, rnd_result.density, linewidth=2)
        ax.axvline(rnd_result.spot_price, color="red", linestyle="--", label=f"Spot = {rnd_result.spot_price:.0f}")

        quantiles = rnd_result.get_quantiles([0.05, 0.50, 0.95])
        for q, val in quantiles.items():
            ax.axvline(val, color="green", linestyle=":", alpha=0.7, label=f"Q{int(q*100)} = {val:.0f}")

        ax.set_xlabel("Strike")
        ax.set_ylabel("Probability Density")
        ax.set_title(title or f"RND - {rnd_result.expiry} ({int(rnd_result.time_to_expiry * 365)}d)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
