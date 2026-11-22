from __future__ import annotations

import json

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from galton import (
    SimulationConfig,
    count_separated_endpoints,
    plot_start_grid_scan,
    run_start_grid_scan,
)


def free_fall_config() -> SimulationConfig:
    return SimulationConfig(
        n_rows=0,
        start_width=1.0,
        peg_radius=0.1,
        ball_radius=0.1,
        horizontal_spacing=1.0,
        vertical_spacing=1.0,
        release_height=1.0,
        detector_y=-2.0,
        initial_velocity_x=0.2,
        boundary_mode="fixed_walls",
        fixed_wall_half_width=10.0,
        n_starts=4,
    )


def test_start_grid_scan_preserves_distinct_linear_free_fall_endpoints(
    tmp_path,
) -> None:
    result = run_start_grid_scan(
        free_fall_config(),
        [4, 8],
        endpoint_tolerance_ratio=0.01,
        workers=1,
        output_directory=tmp_path,
        make_plot=True,
    )

    assert [level.n_starts for level in result.levels] == [4, 8]
    for level in result.levels:
        assert level.detected_count == level.n_starts
        assert level.exact_endpoint_count == level.n_starts
        assert level.resolved_endpoint_count == level.n_starts
        assert level.stalled_count == 0
    assert result.levels[-1].ks_distance_to_finest == 0.0
    assert result.levels[-1].wasserstein_distance_to_finest == 0.0
    assert (tmp_path / "start_grid_scan.png").is_file()
    assert (tmp_path / "start_grid_scan.npz").is_file()
    metadata = json.loads(
        (tmp_path / "start_grid_scan.json").read_text(encoding="utf-8")
    )
    assert metadata["grid_sizes"] == [4, 8]


def test_endpoint_group_count_does_not_depend_on_a_bin_origin() -> None:
    values = np.array([0.0, 0.004, 0.009, 0.03, 0.035])
    assert count_separated_endpoints(values, tolerance=0.01) == 2


def test_start_grid_scan_plot_has_four_bin_free_panels() -> None:
    result = run_start_grid_scan(
        free_fall_config(),
        [4, 8],
        endpoint_tolerance_ratio=0.01,
    )
    figure, axes = plot_start_grid_scan(result)
    assert axes.shape == (2, 2)
    assert "Abbildung" in axes[0, 0].get_title()
    assert "Bin-freie" in axes[1, 1].get_title()
    plt.close(figure)


@pytest.mark.parametrize(
    "grid_sizes",
    [[], [1, 2], [4, 4], [8, 4]],
)
def test_start_grid_scan_rejects_invalid_grid_sizes(grid_sizes) -> None:
    with pytest.raises(ValueError, match="grid_sizes"):
        run_start_grid_scan(
            free_fall_config(),
            grid_sizes,
            endpoint_tolerance_ratio=0.01,
        )
