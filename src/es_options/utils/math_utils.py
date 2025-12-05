"""Math utilities for RND extraction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def central_diff_2nd(
    f: NDArray[np.float64],
    h: float,
) -> NDArray[np.float64]:
    """
    Compute second derivative using central finite differences.
    f''(x) = (f(x+h) - 2*f(x) + f(x-h)) / h^2
    """
    n = len(f)
    d2f = np.zeros(n)

    # Interior points
    d2f[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / (h**2)

    # Edges
    if n >= 3:
        d2f[0] = (f[2] - 2 * f[1] + f[0]) / (h**2)
        d2f[-1] = (f[-1] - 2 * f[-2] + f[-3]) / (h**2)

    return d2f


def normalize_density(
    density: NDArray[np.float64],
    x: NDArray[np.float64],
    min_value: float = 1e-10,
) -> NDArray[np.float64]:
    """Normalize density to integrate to 1, clipping negative values."""
    density = np.maximum(density, min_value)
    integral = np.trapz(density, x)
    if integral > 0:
        return density / integral
    return np.ones_like(density) / (x[-1] - x[0])


def compute_density_moments(
    density: NDArray[np.float64],
    x: NDArray[np.float64],
) -> dict[str, float]:
    """Compute mean, std, skewness, kurtosis of a density."""
    mean = np.trapz(x * density, x)
    variance = np.trapz((x - mean) ** 2 * density, x)
    std = np.sqrt(max(variance, 0))

    if std > 0:
        skewness = np.trapz((x - mean) ** 3 * density, x) / (std**3)
        kurtosis = np.trapz((x - mean) ** 4 * density, x) / (std**4) - 3.0
    else:
        skewness = 0.0
        kurtosis = 0.0

    return {
        "mean": float(mean),
        "std": float(std),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
    }


def compute_quantiles(
    density: NDArray[np.float64],
    x: NDArray[np.float64],
    quantiles: list[float],
) -> NDArray[np.float64]:
    """Compute quantiles from a density function."""
    dx = x[1] - x[0]
    cdf = np.cumsum(density) * dx
    cdf = cdf / cdf[-1] if cdf[-1] > 0 else cdf
    return np.interp(quantiles, cdf, x)
