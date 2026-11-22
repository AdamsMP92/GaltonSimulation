from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from galton import compare_gaussian_and_kde, plot_gaussian_kde_comparison


def test_gaussian_quantiles_are_recognized_as_nearly_gaussian() -> None:
    probabilities = (np.arange(1000, dtype=float) + 0.5) / 1000
    samples = norm.ppf(probabilities, loc=1.5, scale=2.0)

    comparison = compare_gaussian_and_kde(samples)

    assert comparison.n_samples == 1000
    assert comparison.mean == pytest.approx(1.5, abs=1e-12)
    assert comparison.standard_deviation == pytest.approx(2.0, rel=0.01)
    assert abs(comparison.skewness) < 1e-12
    assert comparison.ks_distance < 0.002
    assert comparison.normalized_wasserstein_distance < 0.002


def test_comparison_plot_contains_both_fitted_curves() -> None:
    samples = np.linspace(-2.0, 2.0, 101) ** 3

    figure, axes, comparison = plot_gaussian_kde_comparison(samples)

    labels = {line.get_label() for line in axes[0].lines}
    assert any(label.startswith("KDE") for label in labels)
    assert any(label.startswith("Gauß-Fit") for label in labels)
    assert comparison.n_samples == samples.size
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.close(figure)


def test_comparison_rejects_constant_samples() -> None:
    with pytest.raises(ValueError, match="must not all be identical"):
        compare_gaussian_and_kde(np.ones(10))
