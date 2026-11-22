"""Small, human-facing configuration builders."""

from __future__ import annotations

from math import sqrt
from typing import Literal

from .config import SimulationConfig


def simple_galton_config(
    *,
    rows: int,
    peg_diameter_ratio: float,
    ball_diameter_ratio: float,
    peg_spacing_ratio: float,
    release_height_ratio: float,
    detector_gap_ratio: float,
    detector_width_ratio: float,
    boundary_behavior: Literal["absorbing", "reflecting"],
    starts: int,
    start_width: float = 1.0,
    initial_velocity: tuple[float, float] = (0.0, 0.0),
    kinetic_energy_retention: float = 1.0,
    max_events: int = 1000,
    max_time: float = 100.0,
) -> SimulationConfig:
    """Build a readable reference configuration from a few quantities.

    Every argument ending in ``_ratio`` is a dimensionless length divided by
    ``start_width``.  These physical ratios are deliberately required: this
    wrapper does not invent a peg/ball split or a release-height ratio.

    The vertical spacing is derived as ``sqrt(3)/2`` times the horizontal
    spacing for an equilateral triangular (hexagonal) lattice.  The detector
    is placed at ``y=0``.  ``detector_width_ratio`` is the width available to
    ball centers between the two lower vertical bands.  From the lowest row
    upwards, the side walls follow the grid slope; above the top row a short
    vertical inlet keeps rebounds inside.  Initial velocity defaults to zero.
    ``kinetic_energy_retention`` is the common factor ``w_e`` for every peg
    and wall impact: the reflected velocity is scaled by ``sqrt(w_e)``.
    """

    if rows <= 0:
        raise ValueError("rows must be positive")
    for name, value in (
        ("peg_diameter_ratio", peg_diameter_ratio),
        ("ball_diameter_ratio", ball_diameter_ratio),
        ("peg_spacing_ratio", peg_spacing_ratio),
        ("release_height_ratio", release_height_ratio),
        ("detector_gap_ratio", detector_gap_ratio),
        ("detector_width_ratio", detector_width_ratio),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if starts <= 0:
        raise ValueError("starts must be positive")
    if start_width <= 0.0:
        raise ValueError("start_width must be positive")
    if len(initial_velocity) != 2:
        raise ValueError("initial_velocity must contain (v_x, v_y)")
    if boundary_behavior not in {"absorbing", "reflecting"}:
        raise ValueError("boundary_behavior must be 'absorbing' or 'reflecting'")
    if not 0.0 < kinetic_energy_retention <= 1.0:
        raise ValueError("kinetic_energy_retention must lie in (0, 1]")

    peg_radius = 0.5 * peg_diameter_ratio * start_width
    ball_radius = 0.5 * ball_diameter_ratio * start_width
    collision_radius = peg_radius + ball_radius
    peg_spacing = peg_spacing_ratio * start_width
    release_height = release_height_ratio * start_width
    detector_gap = detector_gap_ratio * start_width
    detector_width = detector_width_ratio * start_width

    vertical_spacing = sqrt(3.0) * peg_spacing / 2.0
    bottom_row_y = detector_gap + collision_radius
    y_top = bottom_row_y + (rows - 1) * vertical_spacing
    return SimulationConfig(
        n_rows=rows,
        start_width=start_width,
        peg_radius=peg_radius,
        ball_radius=ball_radius,
        horizontal_spacing=peg_spacing,
        vertical_spacing=vertical_spacing,
        y_top=y_top,
        release_height=release_height,
        detector_y=0.0,
        detector_gap=detector_gap,
        gravity=1.0,
        initial_velocity_x=float(initial_velocity[0]),
        initial_velocity_y=float(initial_velocity[1]),
        restitution=1.0,
        kinetic_energy_retention=kinetic_energy_retention,
        boundary_mode="funnel_walls",
        boundary_behavior=boundary_behavior,
        detector_width=detector_width,
        wall_restitution=1.0,
        n_starts=starts,
        collision_threshold=3,
        max_events=max_events,
        max_time=max_time,
    )
