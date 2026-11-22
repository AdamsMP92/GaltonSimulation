"""Peg-array and side-envelope geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import SimulationConfig


FloatArray = NDArray[np.float64]


def triangular_peg_centers(config: SimulationConfig) -> FloatArray:
    """Return peg centers in row-major order.

    Row ``i`` contains ``i + 1`` pegs at
    ``x = (j - i/2) d`` and ``y = y_top - i h``.
    """

    centers = [
        (
            (column - row / 2.0) * config.horizontal_spacing,
            config.y_top - row * config.vertical_spacing,
        )
        for row in range(config.n_rows)
        for column in range(row + 1)
    ]
    if not centers:
        return np.empty((0, 2), dtype=float)
    return np.asarray(centers, dtype=float)


@dataclass(frozen=True)
class SideEnvelope:
    """Symmetric envelope ``|x| <= half_width(y)`` for ``y <= y_top``.

    ``transition_y`` optionally changes the expanding triangular part into a
    vertical lower corridor of width ``2 * lower_half_width``.
    """

    y_top: float
    half_width_at_top: float
    expansion_per_drop: float
    transition_y: float | None = None
    lower_half_width: float | None = None
    upper_half_width: float | None = None

    def half_width(self, y: float | FloatArray) -> float | FloatArray:
        sloped_width = self.half_width_at_top + self.expansion_per_drop * (
            self.y_top - y
        )
        if self.upper_half_width is not None:
            sloped_width = np.where(y > self.y_top, self.upper_half_width, sloped_width)
        if self.transition_y is None:
            return sloped_width
        assert self.lower_half_width is not None
        return np.where(y < self.transition_y, self.lower_half_width, sloped_width)

    def local_slope(self, y: float) -> float:
        if self.upper_half_width is not None and y > self.y_top:
            return 0.0
        if self.transition_y is not None and y < self.transition_y:
            return 0.0
        return self.expansion_per_drop

    def outward_normal(self, side: str, y: float) -> FloatArray:
        slope = self.local_slope(y)
        normal = np.array([1.0 if side == "right" else -1.0, slope])
        return normal / np.linalg.norm(normal)

    def side_outside(self, position: FloatArray, tolerance: float = 0.0) -> str | None:
        x, y = float(position[0]), float(position[1])
        if self.upper_half_width is None and y > self.y_top + tolerance:
            return None
        half_width = float(self.half_width(y))
        if x > half_width + tolerance:
            return "right"
        if x < -half_width - tolerance:
            return "left"
        return None


def make_side_envelope(config: SimulationConfig) -> SideEnvelope:
    if config.boundary_mode == "collision_envelope":
        top_width = config.effective_radius
        slope = config.horizontal_spacing / (2.0 * config.vertical_spacing)
    elif config.boundary_mode == "center_envelope":
        top_width = 0.0
        slope = config.horizontal_spacing / (2.0 * config.vertical_spacing)
    elif config.boundary_mode == "funnel_walls":
        assert config.detector_width is not None
        lower_width = config.detector_width / 2.0
        slope = config.horizontal_spacing / (2.0 * config.vertical_spacing)
        top_width = lower_width - max(config.n_rows - 1, 0) * (
            config.horizontal_spacing / 2.0
        )
        return SideEnvelope(
            y_top=config.y_top,
            half_width_at_top=top_width,
            expansion_per_drop=slope,
            transition_y=config.bottom_row_y,
            lower_half_width=lower_width,
            upper_half_width=top_width,
        )
    else:
        if config.fixed_wall_half_width is None:
            bottom_center = max(config.n_rows - 1, 0) * config.horizontal_spacing / 2.0
            top_width = max(config.start_width / 2.0, bottom_center + config.effective_radius)
        else:
            top_width = config.fixed_wall_half_width
        slope = 0.0
    return SideEnvelope(config.y_top, top_width, slope)
