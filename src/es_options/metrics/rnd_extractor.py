"""Risk-Neutral Density extraction using Breeden-Litzenberger."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from es_options.processing.smoothing import SmoothingResult
from es_options.utils.math_utils import compute_density_moments, compute_quantiles, normalize_density


@dataclass
class DataQuality:
    """Data quality metrics for an RND extraction."""
    num_raw_options: int  # Total options before filtering
    num_otm_options: int  # OTM options used (calls + synthetic from puts)
    num_calls: int  # Actual OTM calls
    num_puts_synthetic: int  # Puts converted to synthetic calls
    strike_range: tuple[float, float]  # (min, max) strike
    strike_coverage: float  # strike_range / spot as percentage
    avg_strike_gap: float  # Average gap between strikes
    max_strike_gap: float  # Largest gap (problem area)
    moneyness_range: tuple[float, float]  # (min K/S, max K/S)
    quality_score: float  # 0-1 overall quality score

    def __str__(self) -> str:
        return (
            f"Options: {self.num_otm_options} ({self.num_calls}C + {self.num_puts_synthetic}P→C)\n"
            f"Strikes: {self.strike_range[0]:.0f}-{self.strike_range[1]:.0f} "
            f"({self.strike_coverage:.0%} of spot)\n"
            f"Gaps: avg={self.avg_strike_gap:.1f}, max={self.max_strike_gap:.1f}\n"
            f"Moneyness: {self.moneyness_range[0]:.2f}-{self.moneyness_range[1]:.2f}\n"
            f"Quality: {self.quality_score:.0%}"
        )

    @property
    def quality_label(self) -> str:
        if self.quality_score >= 0.8:
            return "HIGH"
        elif self.quality_score >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"


@dataclass
class RNDResult:
    """Result of RND extraction for a single expiry."""
    strikes: NDArray[np.float64]
    density: NDArray[np.float64]
    expiry: date
    time_to_expiry: float
    spot_price: float
    mean: float
    std: float
    skewness: float
    kurtosis: float
    data_quality: DataQuality | None = None

    def get_quantiles(self, quantiles: list[float] = [0.05, 0.25, 0.50, 0.75, 0.95]) -> dict[float, float]:
        values = compute_quantiles(self.density, self.strikes, quantiles)
        return dict(zip(quantiles, values))

    def probability_below(self, strike: float) -> float:
        mask = self.strikes <= strike
        if not mask.any():
            return 0.0
        return float(np.trapz(self.density[mask], self.strikes[mask]))

    def probability_above(self, strike: float) -> float:
        return 1.0 - self.probability_below(strike)


class RNDExtractor:
    """
    Extract Risk-Neutral Density using Breeden-Litzenberger formula.

    f_Q(K) = e^(rT) * d²C/dK²
    """

    def __init__(self, r: float = 0.05, num_points: int = 500):
        self.r = r
        self.num_points = num_points

    def extract(
        self,
        smooth_result: SmoothingResult,
        expiry: date,
        trade_date: date,
    ) -> RNDResult:
        """Extract RND from smoothed call prices."""
        T = (expiry - trade_date).days / 365.0
        if T <= 0:
            raise ValueError(f"Expiry {expiry} must be after trade date {trade_date}")

        strikes = smooth_result.get_strike_grid(self.num_points)
        density = self._compute_density(smooth_result.smooth_fn, strikes, T, smooth_result)
        moments = compute_density_moments(density, strikes)

        return RNDResult(
            strikes=strikes,
            density=density,
            expiry=expiry,
            time_to_expiry=T,
            spot_price=smooth_result.spot_price,
            mean=moments["mean"],
            std=moments["std"],
            skewness=moments["skewness"],
            kurtosis=moments["kurtosis"],
        )

    def _compute_density(
        self,
        C: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        strikes: NDArray[np.float64],
        T: float,
        smooth_result: SmoothingResult | None = None,
    ) -> NDArray[np.float64]:
        """Compute RND using Breeden-Litzenberger: f_Q(K) = e^(rT) * d²C/dK²"""
        # Try to use spline's derivative if available (more accurate)
        if smooth_result is not None and smooth_result.spline is not None:
            try:
                log_strikes = np.log(strikes)
                # Get second derivative of spline in log-strike space
                # Need chain rule: d²C/dK² = (d²C/d(logK)² - dC/d(logK)) / K²
                spline = smooth_result.spline
                d1 = spline.derivative(1)(log_strikes)  # dC/d(logK)
                d2 = spline.derivative(2)(log_strikes)  # d²C/d(logK)²
                # Chain rule for second derivative
                d2C = (d2 - d1) / (strikes ** 2)
            except Exception:
                # Fall back to finite differences
                d2C = self._finite_diff_2nd(C, strikes)
        else:
            d2C = self._finite_diff_2nd(C, strikes)

        density = np.exp(self.r * T) * d2C
        return normalize_density(density, strikes)

    def _finite_diff_2nd(
        self,
        C: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        strikes: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute second derivative using finite differences."""
        dK = strikes[1] - strikes[0]
        prices = C(strikes)

        d2C = np.zeros_like(prices)
        d2C[1:-1] = (prices[2:] - 2 * prices[1:-1] + prices[:-2]) / (dK**2)
        d2C[0] = (prices[2] - 2 * prices[1] + prices[0]) / (dK**2)
        d2C[-1] = (prices[-1] - 2 * prices[-2] + prices[-3]) / (dK**2)
        return d2C
