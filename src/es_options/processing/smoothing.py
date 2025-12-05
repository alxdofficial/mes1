"""Smoothing for option prices using smoothing splines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.interpolate import UnivariateSpline


@dataclass
class SmoothingResult:
    """Result of call price smoothing."""
    strikes: NDArray[np.float64]
    prices: NDArray[np.float64]
    smooth_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    spot_price: float
    strike_min: float = field(default=0.0)
    strike_max: float = field(default=0.0)
    spline: UnivariateSpline = field(default=None)

    def __post_init__(self):
        if self.strike_min == 0.0:
            self.strike_min = float(self.strikes.min())
        if self.strike_max == 0.0:
            self.strike_max = float(self.strikes.max())

    def get_strike_grid(self, num_points: int = 500) -> NDArray[np.float64]:
        return np.linspace(self.strike_min, self.strike_max, num_points)


class CallPriceSmoother:
    """Smooth call prices using smoothing splines in log-strike space."""

    def __init__(self, smoothing_factor: float | None = None, min_points: int = 10):
        """
        Args:
            smoothing_factor: Controls smoothness. None = auto (recommended).
                Higher = smoother but less fit to data.
            min_points: Minimum data points required.
        """
        self.smoothing_factor = smoothing_factor
        self.min_points = min_points

    def fit(
        self,
        chain: pd.DataFrame,
        spot_price: float | None = None,
        price_col: str = "mid",
    ) -> SmoothingResult:
        """Fit a smooth call price function C(K)."""
        # Get calls only
        if "right" in chain.columns:
            calls = chain[chain["right"] == "C"].copy()
        else:
            calls = chain.copy()

        # Compute mid price if not present
        if price_col == "mid" and "mid" not in calls.columns:
            calls["mid"] = (calls["bid"] + calls["ask"]) / 2

        if len(calls) < self.min_points:
            raise ValueError(f"Need at least {self.min_points} calls, got {len(calls)}")

        calls = calls.sort_values("strike")
        strikes = calls["strike"].values.astype(np.float64)
        prices = calls[price_col].values.astype(np.float64)

        # Remove invalid prices
        valid = ~np.isnan(prices) & (prices > 0) & ~np.isnan(strikes) & (strikes > 0)
        strikes = strikes[valid]
        prices = prices[valid]

        # Remove duplicates (average prices at same strike)
        unique_strikes, indices = np.unique(strikes, return_inverse=True)
        unique_prices = np.zeros(len(unique_strikes))
        counts = np.zeros(len(unique_strikes))
        for i, idx in enumerate(indices):
            unique_prices[idx] += prices[i]
            counts[idx] += 1
        unique_prices /= counts
        strikes = unique_strikes
        prices = unique_prices

        if len(strikes) < self.min_points:
            raise ValueError(f"Not enough valid prices after cleaning: {len(strikes)}")

        if spot_price is None:
            spot_price = float(strikes[len(strikes) // 2])

        # Work in log-strike space for better numerical properties
        log_strikes = np.log(strikes)

        # Use smoothing spline - key difference from CubicSpline
        # s controls smoothness: higher = smoother
        if self.smoothing_factor is not None:
            s = self.smoothing_factor * len(strikes)
        else:
            # Auto smoothing: fairly aggressive to avoid oscillations
            # Scale by variance of prices to be scale-independent
            price_var = np.var(prices)
            s = len(strikes) * price_var * 0.01

        spline = UnivariateSpline(log_strikes, prices, k=3, s=s, ext=3)  # ext=3 = constant extrapolation

        # Store bounds for extrapolation control
        log_k_min = log_strikes.min()
        log_k_max = log_strikes.max()

        def smooth_fn(K: NDArray[np.float64]) -> NDArray[np.float64]:
            K = np.atleast_1d(K).astype(np.float64)
            log_K = np.log(np.maximum(K, 1e-10))
            # Clip to avoid extrapolation issues
            log_K = np.clip(log_K, log_k_min, log_k_max)
            result = spline(log_K)
            return np.maximum(result, 0)

        return SmoothingResult(
            strikes=strikes,
            prices=prices,
            smooth_fn=smooth_fn,
            spot_price=spot_price,
            strike_min=float(strikes.min()),
            strike_max=float(strikes.max()),
            spline=spline,
        )
