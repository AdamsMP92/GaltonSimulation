from __future__ import annotations

import numpy as np

from galton import SimulationConfig, scan_rho_delta


def test_small_rho_delta_scan_has_expected_shapes() -> None:
    config = SimulationConfig(n_rows=2, n_starts=4, max_events=5, max_time=10.0)
    result = scan_rho_delta(
        rho_values=[0.5, 0.6],
        delta_values=[2.4, 2.8, 3.2],
        base_config=config,
        n_starts=4,
    )
    assert result.survival_fraction.shape == (2, 3)
    assert result.collision_count_mean_all.shape == (2, 3)
    assert np.all((0.0 <= result.survival_fraction) & (result.survival_fraction <= 1.0))
