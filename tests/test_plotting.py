from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from galton import SimulationConfig, plot_simulation_overview, run_simulation


def test_overview_contains_trajectory_and_detector_panels() -> None:
    config = SimulationConfig(
        n_rows=0,
        start_width=1.0,
        peg_radius=0.1,
        ball_radius=0.1,
        horizontal_spacing=1.0,
        vertical_spacing=1.0,
        release_height=1.0,
        detector_y=-2.0,
        boundary_mode="fixed_walls",
        fixed_wall_half_width=10.0,
        n_starts=12,
    )
    run = run_simulation(config)
    figure, axes = plot_simulation_overview(run, max_trajectories=6)

    assert len(axes) == 2
    assert "Trajektorien" in axes[0].get_title()
    assert "Detektorstatistik" in axes[1].get_title()
    assert axes[1].patches  # histogram bins
    assert axes[1].lines  # KDE curve
    plt.close(figure)


def test_overview_plots_every_trajectory_and_detector_hit_by_default() -> None:
    config = SimulationConfig(
        n_rows=0,
        start_width=1.0,
        peg_radius=0.1,
        ball_radius=0.1,
        horizontal_spacing=1.0,
        vertical_spacing=1.0,
        release_height=1.0,
        detector_y=-2.0,
        boundary_mode="fixed_walls",
        fixed_wall_half_width=10.0,
        n_starts=12,
    )
    run = run_simulation(config)
    figure, axes = plot_simulation_overview(run)

    detector_marks = [
        collection
        for collection in axes[0].collections
        if collection.get_label() == "alle Detektortreffer"
    ]
    assert len(detector_marks) == 1
    assert len(detector_marks[0].get_offsets()) == config.n_starts
    assert "alle 12 Trajektorien dargestellt" in axes[0].get_title()
    plt.close(figure)


def test_overview_shows_escaped_paths_transparently() -> None:
    config = SimulationConfig(
        n_rows=0,
        start_width=1.0,
        peg_radius=0.1,
        ball_radius=0.1,
        horizontal_spacing=1.0,
        vertical_spacing=1.0,
        release_height=1.0,
        detector_y=-2.0,
        boundary_mode="fixed_walls",
        fixed_wall_half_width=0.3,
        n_starts=12,
    )
    run = run_simulation(config)
    figure, axes = plot_simulation_overview(
        run,
        max_trajectories=6,
        max_escaped_per_side=3,
    )

    labelled_lines = {
        line.get_label(): line
        for line in axes[0].lines
        if not line.get_label().startswith("_")
    }
    assert "am Detektor" in labelled_lines
    assert "links ausgetreten" in labelled_lines
    assert "rechts ausgetreten" in labelled_lines
    assert labelled_lines["am Detektor"].get_alpha() == 0.78
    assert labelled_lines["links ausgetreten"].get_alpha() == 0.16
    assert labelled_lines["rechts ausgetreten"].get_alpha() == 0.16
    plt.close(figure)
