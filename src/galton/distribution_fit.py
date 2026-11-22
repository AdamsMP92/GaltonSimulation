"""Descriptive comparison of detector samples with a Gaussian distribution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import gaussian_kde, kurtosis, norm, skew


@dataclass(frozen=True)
class GaussianKDEComparison:
    """Numerical result of a Gaussian-versus-KDE density comparison."""

    n_samples: int
    mean: float
    standard_deviation: float
    kde_bandwidth: float
    skewness: float
    excess_kurtosis: float
    ks_distance: float
    wasserstein_distance: float
    normalized_wasserstein_distance: float
    density_l1_distance: float
    grid: np.ndarray
    kde_density: np.ndarray
    gaussian_density: np.ndarray

    @property
    def total_variation_distance(self) -> float:
        """Approximate density distance, one half of the L1 distance."""

        return 0.5 * self.density_l1_distance

    def to_dict(self) -> dict[str, int | float]:
        """Return scalar diagnostics suitable for JSON output."""

        return {
            "n_samples": self.n_samples,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "kde_bandwidth": self.kde_bandwidth,
            "skewness": self.skewness,
            "excess_kurtosis": self.excess_kurtosis,
            "ks_distance": self.ks_distance,
            "wasserstein_distance": self.wasserstein_distance,
            "normalized_wasserstein_distance": self.normalized_wasserstein_distance,
            "density_l1_distance": self.density_l1_distance,
            "total_variation_distance": self.total_variation_distance,
        }


def compare_gaussian_and_kde(
    samples: np.ndarray,
    *,
    kde_bandwidth: str | float = "scott",
    grid_points: int = 1200,
) -> GaussianKDEComparison:
    """Fit a Gaussian and a KDE to the same finite one-dimensional samples.

    The Gaussian uses maximum-likelihood estimates for its mean and standard
    deviation.  The KDE is descriptive and uses SciPy's bandwidth convention.
    No histogram bins enter any of the returned distance measures.
    """

    values = np.asarray(samples, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("at least two finite detector samples are required")
    if float(np.ptp(values)) <= 0.0:
        raise ValueError("detector samples must not all be identical")
    if grid_points < 100:
        raise ValueError("grid_points must be at least 100")

    mean, standard_deviation = norm.fit(values)
    mean = float(mean)
    standard_deviation = float(standard_deviation)
    if standard_deviation <= 0.0:
        raise ValueError("the fitted Gaussian must have positive width")

    kde = gaussian_kde(values, bw_method=kde_bandwidth)
    sample_span = float(np.ptp(values))
    lower = min(float(np.min(values)) - 0.1 * sample_span, mean - 4.5 * standard_deviation)
    upper = max(float(np.max(values)) + 0.1 * sample_span, mean + 4.5 * standard_deviation)
    grid = np.linspace(lower, upper, grid_points)
    kde_density = np.asarray(kde(grid), dtype=float)
    gaussian_density = np.asarray(
        norm.pdf(grid, loc=mean, scale=standard_deviation),
        dtype=float,
    )

    sorted_values = np.sort(values)
    n_samples = int(values.size)
    gaussian_cdf = norm.cdf(
        sorted_values,
        loc=mean,
        scale=standard_deviation,
    )
    empirical_upper = np.arange(1, n_samples + 1, dtype=float) / n_samples
    empirical_lower = np.arange(n_samples, dtype=float) / n_samples
    ks_distance = float(
        max(
            np.max(empirical_upper - gaussian_cdf),
            np.max(gaussian_cdf - empirical_lower),
        )
    )

    gaussian_quantiles = norm.ppf(
        (np.arange(n_samples, dtype=float) + 0.5) / n_samples,
        loc=mean,
        scale=standard_deviation,
    )
    wasserstein_distance = float(np.mean(np.abs(sorted_values - gaussian_quantiles)))
    density_l1_distance = float(
        np.trapezoid(np.abs(kde_density - gaussian_density), grid)
    )

    return GaussianKDEComparison(
        n_samples=n_samples,
        mean=mean,
        standard_deviation=standard_deviation,
        kde_bandwidth=float(np.sqrt(kde.covariance[0, 0])),
        skewness=float(skew(values, bias=False)),
        excess_kurtosis=float(kurtosis(values, fisher=True, bias=False)),
        ks_distance=ks_distance,
        wasserstein_distance=wasserstein_distance,
        normalized_wasserstein_distance=wasserstein_distance / standard_deviation,
        density_l1_distance=density_l1_distance,
        grid=grid,
        kde_density=kde_density,
        gaussian_density=gaussian_density,
    )
