from __future__ import annotations

import json
from dataclasses import replace
from math import sqrt
from pathlib import Path

import numpy as np
import pytest

from galton import (
    SimulationConfig,
    run_simulation,
    simple_galton_config,
    trace_trajectory,
)
from galton.observables import summarize_trajectories


def free_fall_config(**changes) -> SimulationConfig:
    config = SimulationConfig(
        n_rows=0,
        start_width=1.0,
        peg_radius=0.1,
        ball_radius=0.1,
        horizontal_spacing=1.0,
        vertical_spacing=1.0,
        release_height=1.0,
        detector_y=-2.0,
        gravity=1.0,
        boundary_mode="fixed_walls",
        fixed_wall_half_width=10.0,
        n_starts=8,
        max_time=20.0,
    )
    return replace(config, **changes)


def test_continuous_detector_returns_real_impact_coordinate() -> None:
    config = free_fall_config(initial_velocity_x=0.25)
    result = trace_trajectory(config, x0=0.125)
    expected_time = sqrt(6.0)

    assert result.status == "detected_inside"
    assert result.x_detector == pytest.approx(0.125 + 0.25 * expected_time)
    assert isinstance(result.x_detector, float)
    assert result.n_collisions == 0


def test_collision_wins_when_it_precedes_detector() -> None:
    config = SimulationConfig(
        n_rows=1,
        start_width=1.0,
        peg_radius=0.1,
        ball_radius=0.1,
        horizontal_spacing=1.0,
        vertical_spacing=1.0,
        release_height=0.5,
        detector_y=-1.0,
        boundary_mode="fixed_walls",
        fixed_wall_half_width=10.0,
        n_starts=2,
        max_events=1,
    )
    result = trace_trajectory(config, x0=0.0)

    assert result.status == "max_events"
    assert result.n_collisions == 1
    assert result.collisions[0].velocity_before[1] < 0.0
    assert result.collisions[0].velocity_after[1] > 0.0
    assert result.collisions[0].energy_error < 1.0e-12


def test_peg_collision_removes_configured_kinetic_energy_fraction() -> None:
    config = SimulationConfig(
        n_rows=1,
        start_width=1.0,
        peg_radius=0.1,
        ball_radius=0.1,
        horizontal_spacing=1.0,
        vertical_spacing=1.0,
        release_height=0.5,
        detector_y=-1.0,
        kinetic_energy_retention=0.64,
        boundary_mode="fixed_walls",
        fixed_wall_half_width=10.0,
        n_starts=2,
        max_events=1,
    )
    result = trace_trajectory(config, x0=0.0)

    assert result.n_collisions == 1
    collision = result.collisions[0]
    assert collision.kinetic_energy_after == pytest.approx(
        0.64 * collision.kinetic_energy_before
    )
    assert collision.energy_loss == pytest.approx(
        0.36 * collision.kinetic_energy_before
    )
    assert result.total_collision_energy_loss == pytest.approx(
        collision.energy_loss
    )
    assert collision.energy_error < 1.0e-12


def test_saved_run_contains_collision_energy_data(tmp_path: Path) -> None:
    config = SimulationConfig(
        n_rows=1,
        start_width=1.0,
        peg_radius=0.1,
        ball_radius=0.1,
        horizontal_spacing=1.0,
        vertical_spacing=1.0,
        release_height=0.5,
        detector_y=-1.0,
        kinetic_energy_retention=0.64,
        boundary_mode="fixed_walls",
        fixed_wall_half_width=10.0,
        n_starts=1,
        max_events=1,
    )
    run = run_simulation(config, output_directory=tmp_path)

    metadata = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert run.simulation_seconds > 0.0
    assert metadata["timing"]["simulation_seconds"] == pytest.approx(
        run.simulation_seconds
    )
    assert metadata["timing"]["workers"] == 1
    with np.load(tmp_path / "run.npz") as saved:
        assert float(saved["simulation_seconds"]) == pytest.approx(
            run.simulation_seconds
        )
        assert int(saved["workers"]) == 1
        assert saved["collision_kinetic_energy_before"].shape == (1,)
        assert saved["collision_kinetic_energy_after"][0] == pytest.approx(
            0.64 * saved["collision_kinetic_energy_before"][0]
        )
        assert saved["collision_energy_loss_total"][0] == pytest.approx(
            run.trajectories[0].total_collision_energy_loss
        )


def test_max_events_is_not_counted_as_survival() -> None:
    config = SimulationConfig(
        n_rows=1,
        peg_radius=0.1,
        ball_radius=0.1,
        horizontal_spacing=1.0,
        vertical_spacing=1.0,
        release_height=0.5,
        detector_y=-1.0,
        boundary_mode="fixed_walls",
        fixed_wall_half_width=10.0,
        n_starts=2,
        max_events=1,
    )
    trajectory = trace_trajectory(config, x0=0.0)
    summary = summarize_trajectories([trajectory], collision_threshold=1)
    assert summary.survival_fraction == 0.0
    assert summary.failure_fraction == 1.0


def test_initial_overlap_is_reported() -> None:
    config = SimulationConfig(
        n_rows=1,
        peg_radius=0.1,
        ball_radius=0.1,
        horizontal_spacing=1.0,
        vertical_spacing=1.0,
        release_height=0.1,
        detector_y=-1.0,
        boundary_mode="fixed_walls",
        fixed_wall_half_width=10.0,
        n_starts=2,
    )
    result = trace_trajectory(config, x0=0.0)
    assert result.status == "invalid_initial_state"


def test_mirror_symmetric_starts_have_mirrored_results() -> None:
    config = SimulationConfig(
        n_rows=2,
        start_width=1.0,
        peg_radius=0.15,
        ball_radius=0.1,
        horizontal_spacing=1.0,
        vertical_spacing=0.8,
        release_height=0.6,
        detector_y=-2.0,
        boundary_mode="fixed_walls",
        fixed_wall_half_width=10.0,
        n_starts=2,
        max_events=20,
        max_time=20.0,
    )
    left = trace_trajectory(config, x0=-0.12)
    right = trace_trajectory(config, x0=0.12)

    assert left.status == right.status
    assert left.n_collisions == right.n_collisions
    if left.x_detector is not None and right.x_detector is not None:
        assert left.x_detector == pytest.approx(-right.x_detector, abs=1.0e-9)
    for left_collision, right_collision in zip(
        left.collisions, right.collisions, strict=True
    ):
        assert left_collision.position[0] == pytest.approx(
            -right_collision.position[0], abs=1.0e-9
        )
        assert left_collision.position[1] == pytest.approx(
            right_collision.position[1], abs=1.0e-9
        )


def test_only_sum_of_radii_controls_contacts() -> None:
    first = SimulationConfig(n_starts=2, peg_radius=0.4, ball_radius=0.2)
    second = replace(first, peg_radius=0.1, ball_radius=0.5)
    first_result = trace_trajectory(first, x0=0.2)
    second_result = trace_trajectory(second, x0=0.2)

    assert first_result.status == second_result.status
    assert first_result.n_collisions == second_result.n_collisions
    assert first_result.x_detector == pytest.approx(second_result.x_detector)


def test_free_fall_is_scale_invariant() -> None:
    base = free_fall_config(initial_velocity_x=0.3, fixed_wall_half_width=10.0)
    scale = 4.0
    scaled = replace(
        base,
        start_width=base.start_width * scale,
        peg_radius=base.peg_radius * scale,
        ball_radius=base.ball_radius * scale,
        horizontal_spacing=base.horizontal_spacing * scale,
        vertical_spacing=base.vertical_spacing * scale,
        release_height=base.release_height * scale,
        detector_y=base.resolved_detector_y * scale,
        initial_velocity_x=base.initial_velocity_x * sqrt(scale),
        fixed_wall_half_width=base.fixed_wall_half_width * scale,
        max_time=base.max_time * sqrt(scale),
    )

    base_result = trace_trajectory(base, x0=0.2)
    scaled_result = trace_trajectory(scaled, x0=0.2 * scale)
    assert base_result.status == scaled_result.status == "detected_inside"
    assert scaled_result.total_time == pytest.approx(base_result.total_time * sqrt(scale))
    assert scaled_result.x_detector / scale == pytest.approx(base_result.x_detector)


def test_one_call_runs_deterministic_midpoint_ensemble() -> None:
    config = free_fall_config(n_starts=6)
    first = run_simulation(config)
    second = run_simulation(config)
    np.testing.assert_allclose(first.start_positions, second.start_positions)
    np.testing.assert_array_equal(first.start_positions, -first.start_positions[::-1])
    assert first.summary.to_dict() == second.summary.to_dict()
    assert first.summary.survival_fraction == 1.0


def test_two_workers_match_serial_run() -> None:
    config = free_fall_config(n_starts=6, store_trajectories=False)
    serial = run_simulation(config, workers=1)
    parallel = run_simulation(config, workers=2)

    np.testing.assert_array_equal(serial.start_positions, parallel.start_positions)
    assert serial.summary.to_dict() == parallel.summary.to_dict()
    assert [trajectory.status for trajectory in serial.trajectories] == [
        trajectory.status for trajectory in parallel.trajectories
    ]
    assert serial.workers == 1
    assert parallel.workers == 2


def test_runner_rejects_more_than_two_workers() -> None:
    with pytest.raises(ValueError, match="workers must be 1 or 2"):
        run_simulation(free_fall_config(n_starts=2), workers=4)


def test_odd_midpoint_grid_is_exactly_symmetric() -> None:
    result = run_simulation(free_fall_config(n_starts=5))
    np.testing.assert_array_equal(result.start_positions, -result.start_positions[::-1])
    assert result.start_positions[2] == 0.0


def test_reflecting_vertical_wall_keeps_ball_in_system() -> None:
    config = SimulationConfig(
        n_rows=0,
        start_width=1.0,
        peg_radius=0.1,
        ball_radius=0.1,
        horizontal_spacing=1.0,
        vertical_spacing=1.0,
        release_height=1.0,
        detector_y=-2.0,
        gravity=1.0,
        initial_velocity_x=0.6,
        boundary_mode="funnel_walls",
        boundary_behavior="reflecting",
        detector_width=2.0,
        wall_restitution=1.0,
        n_starts=2,
    )
    result = trace_trajectory(config, x0=0.0)

    assert result.status == "detected_inside"
    assert result.n_wall_collisions == 1
    wall_collision = result.wall_collisions[0]
    assert wall_collision.side == "right"
    assert wall_collision.velocity_before[0] > 0.0
    assert wall_collision.velocity_after[0] < 0.0
    assert wall_collision.energy_error < 1.0e-12


def test_wall_collision_uses_same_kinetic_energy_retention() -> None:
    config = SimulationConfig(
        n_rows=0,
        start_width=1.0,
        peg_radius=0.1,
        ball_radius=0.1,
        horizontal_spacing=1.0,
        vertical_spacing=1.0,
        release_height=1.0,
        detector_y=-2.0,
        gravity=1.0,
        initial_velocity_x=0.6,
        kinetic_energy_retention=0.64,
        boundary_mode="funnel_walls",
        boundary_behavior="reflecting",
        detector_width=2.0,
        n_starts=2,
    )
    result = trace_trajectory(config, x0=0.0)

    assert result.n_wall_collisions == 1
    collision = result.wall_collisions[0]
    assert collision.kinetic_energy_after == pytest.approx(
        0.64 * collision.kinetic_energy_before
    )
    assert collision.energy_loss == pytest.approx(
        0.36 * collision.kinetic_energy_before
    )
    assert collision.energy_error < 1.0e-12


def test_unresolved_inelastic_contact_is_filtered_as_stalled() -> None:
    config = simple_galton_config(
        rows=16,
        peg_diameter_ratio=0.1,
        ball_diameter_ratio=0.3,
        peg_spacing_ratio=0.8,
        release_height_ratio=1.0,
        detector_gap_ratio=2.0,
        detector_width_ratio=13.0,
        boundary_behavior="reflecting",
        starts=1,
        kinetic_energy_retention=0.25,
        max_events=1000,
        max_time=200.0,
    )
    result = trace_trajectory(config, x0=0.00025)
    summary = summarize_trajectories([result], collision_threshold=1)

    assert result.status == "stalled"
    assert result.n_collisions < config.max_events
    assert summary.survival_fraction == 0.0
    assert summary.stalled_fraction == 1.0


def test_absorbing_funnel_still_reports_escape() -> None:
    config = SimulationConfig(
        n_rows=0,
        start_width=1.0,
        peg_radius=0.1,
        ball_radius=0.1,
        horizontal_spacing=1.0,
        vertical_spacing=1.0,
        release_height=1.0,
        detector_y=-2.0,
        gravity=1.0,
        initial_velocity_x=0.6,
        boundary_mode="funnel_walls",
        boundary_behavior="absorbing",
        detector_width=2.0,
        n_starts=2,
    )
    result = trace_trajectory(config, x0=0.0)
    assert result.status == "escaped_right"
    assert result.n_wall_collisions == 0


def test_sloped_funnel_wall_reflects_with_correct_normal() -> None:
    config = SimulationConfig(
        n_rows=2,
        start_width=1.0,
        peg_radius=0.05,
        ball_radius=0.05,
        horizontal_spacing=1.0,
        vertical_spacing=sqrt(3.0) / 2.0,
        y_top=0.0,
        release_height=0.01,
        detector_y=-2.0,
        gravity=1.0,
        initial_velocity_x=0.2,
        boundary_mode="funnel_walls",
        boundary_behavior="reflecting",
        detector_width=2.0,
        wall_restitution=1.0,
        n_starts=2,
        max_events=50,
    )
    result = trace_trajectory(config, x0=0.46)

    assert result.status == "detected_inside"
    assert result.n_wall_collisions == 1
    collision = result.wall_collisions[0]
    np.testing.assert_allclose(collision.normal, [sqrt(3.0) / 2.0, 0.5])
    assert collision.energy_error < 1.0e-12
