"""Arbitrage checks for options prices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


@dataclass
class ArbitrageCheckResult:
    """Result of arbitrage checks."""
    is_valid: bool
    num_violations: int
    monotonicity_ok: bool
    convexity_ok: bool


class ArbitrageChecker:
    """
    Check call prices for arbitrage violations.

    Two conditions for arbitrage-free call prices:
    1. Monotonicity: C(K) is non-increasing in K
    2. Convexity: d²C/dK² >= 0
    """

    def __init__(self, monotonicity_tol: float = 1e-6, convexity_tol: float = -1e-6):
        self.monotonicity_tol = monotonicity_tol
        self.convexity_tol = convexity_tol

    def check(
        self,
        smooth_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        strikes: NDArray[np.float64],
    ) -> ArbitrageCheckResult:
        """Check smoothed call prices for arbitrage violations."""
        prices = smooth_fn(strikes)

        # Check monotonicity: dC/dK should be <= 0
        dC = np.diff(prices)
        dK = np.diff(strikes)
        first_deriv = dC / dK
        mono_violations = np.sum(first_deriv > self.monotonicity_tol)
        mono_ok = mono_violations == 0

        # Check convexity: d²C/dK² should be >= 0
        dK_uniform = strikes[1] - strikes[0]
        second_deriv = np.zeros_like(prices)
        second_deriv[1:-1] = (prices[2:] - 2 * prices[1:-1] + prices[:-2]) / (dK_uniform**2)
        conv_violations = np.sum(second_deriv[1:-1] < self.convexity_tol)
        conv_ok = conv_violations == 0

        return ArbitrageCheckResult(
            is_valid=mono_ok and conv_ok,
            num_violations=int(mono_violations + conv_violations),
            monotonicity_ok=mono_ok,
            convexity_ok=conv_ok,
        )
