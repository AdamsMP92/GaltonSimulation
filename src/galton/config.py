"""Configuration for the Galton-board model.

All lengths may be given in any one consistent unit.  The start width is the
natural length scale; ``gravity`` fixes the corresponding time scale
``sqrt(start_width / gravity)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Literal


BoundaryMode = Literal[
    "collision_envelope",
    "center_envelope",
    "fixed_walls",
    "funnel_walls",
]
BoundaryBehavior = Literal["absorbing", "reflecting"]


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for one deterministic ensemble run.

    The defaults are dimensionless: ``start_width = gravity = 1``.  They are
    intended as a reproducible example, not as a validated experimental
    geometry.
    """

    n_rows: int = 8
    start_width: float = 1.0
    peg_radius: float = 0.38
    ball_radius: float = 0.28
    horizontal_spacing: float = 2.8
    vertical_spacing: float = 2.45
    y_top: float = 0.0
    release_height: float = 1.5
    detector_y: float | None = None
    detector_gap: float = 1.0
    gravity: float = 1.0
    initial_velocity_x: float = 0.0
    initial_velocity_y: float = 0.0
    # Advanced normal restitution coefficients can be combined with the
    # common whole-velocity energy-retention factor below.
    restitution: float = 1.0
    kinetic_energy_retention: float = 1.0

    boundary_mode: BoundaryMode = "collision_envelope"
    boundary_behavior: BoundaryBehavior = "absorbing"
    fixed_wall_half_width: float | None = None
    detector_width: float | None = None
    wall_restitution: float = 1.0

    n_starts: int = 128
    collision_threshold: int = 3
    max_events: int = 100
    max_time: float = 100.0
    store_trajectories: bool = True

    # Dimensionless numerical tolerances.  Length, time, and velocity
    # tolerances are scaled by their natural units before use.
    position_tolerance: float = 1.0e-9
    time_tolerance: float = 1.0e-10
    velocity_tolerance: float = 1.0e-10
    root_imag_tolerance: float = 1.0e-9
    simultaneous_time_tolerance: float = 1.0e-8
    energy_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.n_rows < 0:
            raise ValueError("n_rows must be non-negative")
        for name in (
            "start_width",
            "horizontal_spacing",
            "vertical_spacing",
            "release_height",
            "detector_gap",
            "gravity",
            "max_time",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.peg_radius < 0.0 or self.ball_radius < 0.0:
            raise ValueError("radii must be non-negative")
        if self.effective_radius <= 0.0:
            raise ValueError("peg_radius + ball_radius must be positive")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("restitution must lie in [0, 1]")
        if not 0.0 < self.kinetic_energy_retention <= 1.0:
            raise ValueError("kinetic_energy_retention must lie in (0, 1]")
        if not 0.0 <= self.wall_restitution <= 1.0:
            raise ValueError("wall_restitution must lie in [0, 1]")
        if self.n_starts <= 0:
            raise ValueError("n_starts must be positive")
        if self.collision_threshold < 0:
            raise ValueError("collision_threshold must be non-negative")
        if self.max_events <= 0:
            raise ValueError("max_events must be positive")
        if self.boundary_mode not in {
            "collision_envelope",
            "center_envelope",
            "fixed_walls",
            "funnel_walls",
        }:
            raise ValueError(f"unknown boundary_mode: {self.boundary_mode}")
        if self.boundary_behavior not in {"absorbing", "reflecting"}:
            raise ValueError(f"unknown boundary_behavior: {self.boundary_behavior}")
        if self.fixed_wall_half_width is not None and self.fixed_wall_half_width <= 0.0:
            raise ValueError("fixed_wall_half_width must be positive")
        if self.detector_width is not None and self.detector_width <= 0.0:
            raise ValueError("detector_width must be positive")
        if self.boundary_mode == "funnel_walls":
            if self.detector_width is None:
                raise ValueError("funnel_walls requires detector_width")
            top_half_width = (
                self.detector_width / 2.0
                - max(self.n_rows - 1, 0) * self.horizontal_spacing / 2.0
            )
            required_top_half_width = max(
                self.start_width / 2.0,
                self.effective_radius if self.n_rows else 0.0,
            )
            if top_half_width < required_top_half_width - self.length_tolerance:
                minimum_width = 2.0 * (
                    required_top_half_width
                    + max(self.n_rows - 1, 0) * self.horizontal_spacing / 2.0
                )
                raise ValueError(
                    "detector_width is too small for the start opening and grid; "
                    f"it must be at least {minimum_width:g}"
                )

        bottom = self.bottom_row_y
        if self.resolved_detector_y >= bottom - self.effective_radius:
            raise ValueError(
                "detector_y must be below the lowest collision circle "
                "(bottom_row_y - effective_radius)"
            )

        for name in (
            "position_tolerance",
            "time_tolerance",
            "velocity_tolerance",
            "root_imag_tolerance",
            "simultaneous_time_tolerance",
            "energy_tolerance",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    @property
    def effective_radius(self) -> float:
        return self.peg_radius + self.ball_radius

    @property
    def release_y(self) -> float:
        return self.y_top + self.release_height

    @property
    def bottom_row_y(self) -> float:
        if self.n_rows == 0:
            return self.y_top
        return self.y_top - (self.n_rows - 1) * self.vertical_spacing

    @property
    def resolved_detector_y(self) -> float:
        if self.detector_y is not None:
            return self.detector_y
        return self.bottom_row_y - self.effective_radius - self.detector_gap

    @property
    def natural_time(self) -> float:
        return sqrt(self.start_width / self.gravity)

    @property
    def natural_velocity(self) -> float:
        return sqrt(self.gravity * self.start_width)

    @property
    def length_tolerance(self) -> float:
        return self.position_tolerance * self.start_width

    @property
    def scaled_time_tolerance(self) -> float:
        return self.time_tolerance * self.natural_time

    @property
    def scaled_velocity_tolerance(self) -> float:
        return self.velocity_tolerance * self.natural_velocity

    @property
    def simultaneous_tolerance(self) -> float:
        return self.simultaneous_time_tolerance * self.natural_time

    def dimensionless_parameters(self) -> dict[str, float | int | str]:
        width = self.start_width
        return {
            "alpha": self.peg_radius / width,
            "beta": self.ball_radius / width,
            "rho": self.effective_radius / width,
            "peg_diameter_over_width": 2.0 * self.peg_radius / width,
            "ball_diameter_over_width": 2.0 * self.ball_radius / width,
            "collision_diameter_over_width": 2.0 * self.effective_radius / width,
            "delta": self.horizontal_spacing / width,
            "eta": self.vertical_spacing / width,
            "h_over_d": self.vertical_spacing / self.horizontal_spacing,
            "release_height_over_width": self.release_height / width,
            "detector_y_over_width": (self.resolved_detector_y - self.y_top) / width,
            "initial_velocity_x_scaled": self.initial_velocity_x / self.natural_velocity,
            "initial_velocity_y_scaled": self.initial_velocity_y / self.natural_velocity,
            "n_rows": self.n_rows,
            "boundary_mode": self.boundary_mode,
            "boundary_behavior": self.boundary_behavior,
            "detector_width_over_width": (
                None if self.detector_width is None else self.detector_width / width
            ),
            "wall_restitution": self.wall_restitution,
            "kinetic_energy_retention": self.kinetic_energy_retention,
        }

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = asdict(self)
        result["effective_radius"] = self.effective_radius
        result["release_y"] = self.release_y
        result["resolved_detector_y"] = self.resolved_detector_y
        result["natural_time"] = self.natural_time
        result["natural_velocity"] = self.natural_velocity
        result["dimensionless"] = self.dimensionless_parameters()
        return result
