from __future__ import annotations

from math import sqrt

import pytest

from galton import simple_galton_config
from galton.geometry import triangular_peg_centers


def example_config(**changes):
    parameters = {
        "rows": 8,
        "peg_diameter_ratio": 0.8,
        "ball_diameter_ratio": 0.6,
        "peg_spacing_ratio": 2.6,
        "release_height_ratio": 1.2,
        "detector_gap_ratio": 2.0,
        "detector_width_ratio": 19.6,
        "boundary_behavior": "reflecting",
        "starts": 2000,
    }
    parameters.update(changes)
    return simple_galton_config(**parameters)


def test_simple_config_derives_equilateral_geometry_and_zero_detector() -> None:
    config = example_config()

    assert config.start_width == 1.0
    assert config.initial_velocity_x == 0.0
    assert config.initial_velocity_y == 0.0
    assert config.peg_radius == pytest.approx(0.4)
    assert config.ball_radius == pytest.approx(0.3)
    assert config.effective_radius == pytest.approx(0.7)
    assert config.vertical_spacing == pytest.approx(sqrt(3.0) * 2.6 / 2.0)
    assert config.resolved_detector_y == 0.0
    assert config.bottom_row_y - config.effective_radius == pytest.approx(2.0)
    assert config.detector_width == pytest.approx(19.6)
    assert config.boundary_mode == "funnel_walls"
    assert config.boundary_behavior == "reflecting"
    assert config.wall_restitution == 1.0
    assert config.kinetic_energy_retention == 1.0
    assert config.n_starts == 2000
    assert config.max_events == 1000
    dimensionless = config.dimensionless_parameters()
    assert dimensionless["peg_diameter_over_width"] == pytest.approx(0.8)
    assert dimensionless["ball_diameter_over_width"] == pytest.approx(0.6)
    assert dimensionless["collision_diameter_over_width"] == pytest.approx(1.4)

    centers = triangular_peg_centers(config)
    top = centers[0]
    second_row_left = centers[1]
    nearest_neighbor_distance = float(((top - second_row_left) ** 2).sum() ** 0.5)
    assert nearest_neighbor_distance == pytest.approx(2.6)


def test_simple_config_accepts_explicit_initial_velocity() -> None:
    config = example_config(initial_velocity=(0.2, -0.1))
    assert config.initial_velocity_x == 0.2
    assert config.initial_velocity_y == -0.1


def test_simple_config_exposes_uniform_collision_energy_retention() -> None:
    config = example_config(kinetic_energy_retention=0.93)
    assert config.kinetic_energy_retention == pytest.approx(0.93)
    assert config.restitution == 1.0
    assert config.wall_restitution == 1.0


@pytest.mark.parametrize("value", [0.0, -0.1, 1.01])
def test_simple_config_rejects_invalid_energy_retention(value: float) -> None:
    with pytest.raises(ValueError, match="kinetic_energy_retention"):
        example_config(kinetic_energy_retention=value)


def test_simple_config_has_no_hidden_geometry_ratio_defaults() -> None:
    with pytest.raises(TypeError):
        simple_galton_config()  # type: ignore[call-arg]


def test_funnel_rejects_detector_width_that_cuts_into_grid() -> None:
    with pytest.raises(ValueError, match="detector_width is too small"):
        example_config(detector_width_ratio=19.5)
