"""Diagnostic plots for options analytics."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from es_options.metrics.rnd_extractor import RNDResult


class DiagnosticsPlotter:
    """Diagnostic plots for validating RND extraction."""

    def __init__(self, figsize: tuple[int, int] = (12, 8)):
        self.figsize = figsize

    def plot_rnd_sanity_check(
        self,
        rnd_result: RNDResult,
        title: str | None = None,
        save_path: Path | str | None = None,
    ) -> plt.Figure:
        """RND sanity check plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title or f"RND Sanity Check - {rnd_result.expiry}", fontsize=14)

        # 1. Density with quantiles
        ax = axes[0, 0]
        ax.fill_between(rnd_result.strikes, rnd_result.density, alpha=0.3)
        ax.plot(rnd_result.strikes, rnd_result.density, linewidth=2)
        ax.axvline(rnd_result.spot_price, color="red", linestyle="--", label="Spot")

        quantiles = rnd_result.get_quantiles([0.05, 0.50, 0.95])
        for q, val in quantiles.items():
            ax.axvline(val, color="green", linestyle=":", alpha=0.7)

        ax.set_xlabel("Strike")
        ax.set_ylabel("Density")
        ax.set_title("Risk-Neutral Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. CDF
        ax = axes[0, 1]
        cdf = np.cumsum(rnd_result.density) * (rnd_result.strikes[1] - rnd_result.strikes[0])
        ax.plot(rnd_result.strikes, cdf, linewidth=2)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(rnd_result.spot_price, color="red", linestyle="--")
        ax.set_xlabel("Strike")
        ax.set_ylabel("CDF")
        ax.set_title("Cumulative Distribution")
        ax.grid(True, alpha=0.3)

        # 3. Log density
        ax = axes[1, 0]
        log_density = np.log(np.maximum(rnd_result.density, 1e-10))
        ax.plot(rnd_result.strikes, log_density, linewidth=2)
        ax.axvline(rnd_result.spot_price, color="red", linestyle="--")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Log Density")
        ax.set_title("Log Density (Tail Behavior)")
        ax.grid(True, alpha=0.3)

        # 4. Statistics and Data Quality
        ax = axes[1, 1]
        ax.axis("off")

        stats = f"""RND Statistics
        ─────────────────────
        Expiry: {rnd_result.expiry}
        DTE: {int(rnd_result.time_to_expiry * 365)} days
        Spot: {rnd_result.spot_price:.2f}

        Mean: {rnd_result.mean:.2f}
        Std: {rnd_result.std:.2f}
        Skew: {rnd_result.skewness:.4f}
        Kurt: {rnd_result.kurtosis:.4f}

        Q5: {quantiles[0.05]:.2f}
        Q50: {quantiles[0.50]:.2f}
        Q95: {quantiles[0.95]:.2f}

        Integral: {np.trapz(rnd_result.density, rnd_result.strikes):.6f}
        """

        # Add data quality info if available
        if rnd_result.data_quality is not None:
            dq = rnd_result.data_quality
            quality_color = "green" if dq.quality_score >= 0.7 else "orange" if dq.quality_score >= 0.4 else "red"
            stats += f"""
        Data Quality: {dq.quality_label} ({dq.quality_score:.0%})
        ─────────────────────
        Options: {dq.num_otm_options} ({dq.num_calls}C + {dq.num_puts_synthetic}P→C)
        Strikes: {dq.strike_range[0]:.0f} - {dq.strike_range[1]:.0f}
        Coverage: {dq.strike_coverage:.0%} of spot
        Avg Gap: {dq.avg_strike_gap:.1f}
        Max Gap: {dq.max_strike_gap:.1f}
        Moneyness: {dq.moneyness_range[0]:.2f} - {dq.moneyness_range[1]:.2f}
        """

        ax.text(0.05, 0.95, stats, transform=ax.transAxes, verticalalignment="top",
                fontsize=10, fontfamily="monospace")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
