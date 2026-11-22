"""Analytic first-exit detection for the piecewise-linear side envelope."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .ballistics import position_at, real_polynomial_roots
from .geometry import SideEnvelope


@dataclass(frozen=True)
class BoundaryExit:
    time: float
    side: str
    normal: np.ndarray


def _unique_sorted(values: list[float], tolerance: float) -> list[float]:
    result: list[float] = []
    for value in sorted(values):
        if not result or abs(value - result[-1]) > tolerance:
            result.append(value)
    return result


def first_boundary_exit(
    position: ArrayLike,
    velocity: ArrayLike,
    acceleration: ArrayLike,
    envelope: SideEnvelope,
    *,
    maximum_time: float,
    time_tolerance: float,
    position_tolerance: float,
    imag_tolerance: float,
) -> BoundaryExit | None:
    """Return the first time the ballistic segment is outside the envelope.

    Roots of the top-entry condition and both straight side boundaries divide
    the segment into intervals.  Testing each interval detects an exit even if
    the parabola later re-enters before the next peg contact.
    """

    p = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)
    a = np.asarray(acceleration, dtype=float)
    if maximum_time <= 0.0:
        return None

    immediate_side = envelope.side_outside(p, position_tolerance)
    if immediate_side is not None:
        return BoundaryExit(
            0.0,
            immediate_side,
            envelope.outward_normal(immediate_side, float(p[1])),
        )

    slope = envelope.expansion_per_drop
    constant = envelope.half_width_at_top + slope * envelope.y_top

    # Descending-order quadratic coefficients for y-y_top, x-right(y), and
    # left(y)-x.  Positive side-polynomial values mean "outside".
    top_coefficients = (0.5 * a[1], v[1], p[1] - envelope.y_top)
    right_coefficients = (
        0.5 * (a[0] + slope * a[1]),
        v[0] + slope * v[1],
        p[0] + slope * p[1] - constant,
    )
    left_coefficients = (
        0.5 * (-a[0] + slope * a[1]),
        -v[0] + slope * v[1],
        -p[0] + slope * p[1] - constant,
    )

    breakpoints = [0.0, maximum_time]
    coefficient_sets = [top_coefficients, right_coefficients, left_coefficients]
    if envelope.upper_half_width is not None:
        upper_right_coefficients = (
            0.5 * a[0],
            v[0],
            p[0] - envelope.upper_half_width,
        )
        upper_left_coefficients = (
            -0.5 * a[0],
            -v[0],
            -p[0] - envelope.upper_half_width,
        )
        coefficient_sets.extend(
            (upper_right_coefficients, upper_left_coefficients)
        )
    if envelope.transition_y is not None:
        assert envelope.lower_half_width is not None
        transition_coefficients = (
            0.5 * a[1],
            v[1],
            p[1] - envelope.transition_y,
        )
        lower_right_coefficients = (
            0.5 * a[0],
            v[0],
            p[0] - envelope.lower_half_width,
        )
        lower_left_coefficients = (
            -0.5 * a[0],
            -v[0],
            -p[0] - envelope.lower_half_width,
        )
        coefficient_sets.extend(
            (
                transition_coefficients,
                lower_right_coefficients,
                lower_left_coefficients,
            )
        )

    for coefficients in coefficient_sets:
        for root in real_polynomial_roots(coefficients, imag_tolerance=imag_tolerance):
            root_value = float(root)
            if 0.0 < root_value < maximum_time:
                breakpoints.append(root_value)
    breakpoints = _unique_sorted(breakpoints, time_tolerance)

    for start, end in zip(breakpoints[:-1], breakpoints[1:], strict=True):
        if end - start <= time_tolerance:
            continue
        midpoint = 0.5 * (start + end)
        midpoint_position = position_at(p, v, a, midpoint)
        side = envelope.side_outside(midpoint_position, position_tolerance)
        if side is not None:
            return BoundaryExit(
                start,
                side,
                envelope.outward_normal(side, float(midpoint_position[1])),
            )
    return None
