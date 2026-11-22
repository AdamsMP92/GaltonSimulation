"""Analytic ballistic motion and polynomial event times."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def position_at(
    position: ArrayLike,
    velocity: ArrayLike,
    acceleration: ArrayLike,
    time: float,
) -> FloatArray:
    position_array = np.asarray(position, dtype=float)
    velocity_array = np.asarray(velocity, dtype=float)
    acceleration_array = np.asarray(acceleration, dtype=float)
    return position_array + velocity_array * time + 0.5 * acceleration_array * time**2


def velocity_at(velocity: ArrayLike, acceleration: ArrayLike, time: float) -> FloatArray:
    return np.asarray(velocity, dtype=float) + np.asarray(acceleration, dtype=float) * time


def real_polynomial_roots(
    coefficients: ArrayLike,
    *,
    imag_tolerance: float = 1.0e-9,
) -> FloatArray:
    """Return sorted nearly-real roots of a descending-order polynomial."""

    coefficients_array = np.asarray(coefficients, dtype=float)
    if coefficients_array.ndim != 1:
        raise ValueError("coefficients must be one-dimensional")
    scale = float(np.max(np.abs(coefficients_array), initial=0.0))
    if scale == 0.0:
        return np.empty(0, dtype=float)
    threshold = 32.0 * np.finfo(float).eps * scale
    nonzero = np.flatnonzero(np.abs(coefficients_array) > threshold)
    if nonzero.size == 0:
        return np.empty(0, dtype=float)
    trimmed = coefficients_array[nonzero[0] :] / scale
    if trimmed.size <= 1:
        return np.empty(0, dtype=float)

    roots = np.roots(trimmed)
    real_roots = [
        float(root.real)
        for root in roots
        if abs(float(root.imag)) <= imag_tolerance * max(1.0, abs(float(root.real)))
    ]
    return np.asarray(sorted(real_roots), dtype=float)


def detector_crossing_time(
    position: ArrayLike,
    velocity: ArrayLike,
    gravity: float,
    detector_y: float,
    *,
    minimum_time: float,
    velocity_tolerance: float,
    imag_tolerance: float,
    maximum_time: float = np.inf,
) -> float | None:
    """First positive, downward crossing of the horizontal detector plane."""

    position_array = np.asarray(position, dtype=float)
    velocity_array = np.asarray(velocity, dtype=float)
    roots = real_polynomial_roots(
        (-0.5 * gravity, velocity_array[1], position_array[1] - detector_y),
        imag_tolerance=imag_tolerance,
    )
    candidates = [
        float(time)
        for time in roots
        if minimum_time < time <= maximum_time
        and velocity_array[1] - gravity * time < -velocity_tolerance
    ]
    return min(candidates, default=None)


def specific_energy(position: ArrayLike, velocity: ArrayLike, gravity: float) -> float:
    position_array = np.asarray(position, dtype=float)
    velocity_array = np.asarray(velocity, dtype=float)
    return 0.5 * float(np.dot(velocity_array, velocity_array)) + gravity * float(
        position_array[1]
    )
