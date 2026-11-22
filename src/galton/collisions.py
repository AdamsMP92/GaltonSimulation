"""Quartic peg-contact search and frictionless impact response."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .ballistics import position_at, real_polynomial_roots, velocity_at


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CollisionCandidate:
    time: float
    peg_indices: tuple[int, ...]


def _ballistic_coordinate_bounds(
    position: FloatArray,
    velocity: FloatArray,
    acceleration: FloatArray,
    minimum_time: float,
    maximum_time: float,
) -> tuple[FloatArray, FloatArray]:
    """Bounding box of one ballistic segment over a finite time interval."""

    lower = np.empty_like(position)
    upper = np.empty_like(position)
    for axis in range(position.size):
        times = [minimum_time, maximum_time]
        if acceleration[axis] != 0.0:
            extremum_time = -velocity[axis] / acceleration[axis]
            if minimum_time < extremum_time < maximum_time:
                times.append(float(extremum_time))
        values = [
            position[axis]
            + velocity[axis] * time
            + 0.5 * acceleration[axis] * time**2
            for time in times
        ]
        lower[axis] = min(values)
        upper[axis] = max(values)
    return lower, upper


def peg_contact_times(
    position: ArrayLike,
    velocity: ArrayLike,
    acceleration: ArrayLike,
    peg_center: ArrayLike,
    effective_radius: float,
    *,
    minimum_time: float,
    maximum_time: float,
    imag_tolerance: float,
    position_tolerance: float,
    velocity_tolerance: float,
) -> list[float]:
    """All admissible incoming contact times with one circular peg."""

    position_array = np.asarray(position, dtype=float)
    velocity_array = np.asarray(velocity, dtype=float)
    acceleration_array = np.asarray(acceleration, dtype=float)
    center_array = np.asarray(peg_center, dtype=float)

    q = position_array - center_array
    b = velocity_array
    c = 0.5 * acceleration_array
    coefficients = (
        np.dot(c, c),
        2.0 * np.dot(b, c),
        np.dot(b, b) + 2.0 * np.dot(q, c),
        2.0 * np.dot(q, b),
        np.dot(q, q) - effective_radius**2,
    )

    roots = real_polynomial_roots(coefficients, imag_tolerance=imag_tolerance)
    result: list[float] = []
    for root in roots:
        time = float(root)
        if not minimum_time < time <= maximum_time:
            continue
        contact_position = position_at(position_array, velocity_array, acceleration_array, time)
        offset = contact_position - center_array
        distance = float(np.linalg.norm(offset))
        if abs(distance - effective_radius) > position_tolerance:
            continue
        normal = offset / distance
        contact_velocity = velocity_at(velocity_array, acceleration_array, time)
        if float(np.dot(normal, contact_velocity)) < -velocity_tolerance:
            result.append(time)
    return result


def find_next_collision(
    position: ArrayLike,
    velocity: ArrayLike,
    acceleration: ArrayLike,
    peg_centers: FloatArray,
    effective_radius: float,
    *,
    minimum_time: float,
    maximum_time: float,
    imag_tolerance: float,
    position_tolerance: float,
    velocity_tolerance: float,
    simultaneous_tolerance: float,
) -> CollisionCandidate | None:
    """Find the earliest admissible peg contact.

    Nearly simultaneous contacts with different pegs are returned together so
    that the tracer can flag the unresolved rigid multi-contact problem.
    """

    if maximum_time <= minimum_time or peg_centers.size == 0:
        return None

    position_array = np.asarray(position, dtype=float)
    velocity_array = np.asarray(velocity, dtype=float)
    acceleration_array = np.asarray(acceleration, dtype=float)
    candidate_indices = np.arange(len(peg_centers), dtype=int)
    if np.isfinite(maximum_time):
        lower, upper = _ballistic_coordinate_bounds(
            position_array,
            velocity_array,
            acceleration_array,
            minimum_time,
            maximum_time,
        )
        margin = effective_radius + position_tolerance
        possible = np.all(
            (peg_centers >= lower - margin)
            & (peg_centers <= upper + margin),
            axis=1,
        )
        candidate_indices = np.flatnonzero(possible)

    hits: list[tuple[float, int]] = []
    for peg_index in candidate_indices:
        times = peg_contact_times(
            position_array,
            velocity_array,
            acceleration_array,
            peg_centers[peg_index],
            effective_radius,
            minimum_time=minimum_time,
            maximum_time=maximum_time,
            imag_tolerance=imag_tolerance,
            position_tolerance=position_tolerance,
            velocity_tolerance=velocity_tolerance,
        )
        if times:
            hits.append((min(times), int(peg_index)))
    if not hits:
        return None

    first_time = min(time for time, _ in hits)
    indices = tuple(
        peg_index
        for time, peg_index in hits
        if abs(time - first_time) <= simultaneous_tolerance
    )
    return CollisionCandidate(first_time, indices)


def collision_normal(contact_position: ArrayLike, peg_center: ArrayLike) -> FloatArray:
    offset = np.asarray(contact_position, dtype=float) - np.asarray(peg_center, dtype=float)
    norm = float(np.linalg.norm(offset))
    if norm == 0.0:
        raise ValueError("contact position cannot equal peg center")
    return offset / norm


def reflect_velocity(
    incoming_velocity: ArrayLike,
    normal: ArrayLike,
    restitution: float = 1.0,
    kinetic_energy_retention: float = 1.0,
) -> FloatArray:
    """Reflect a velocity and optionally retain a fixed kinetic-energy share.

    For a specular collision (``restitution=1``), the returned velocity obeys
    ``|v_after|^2 = kinetic_energy_retention * |v_before|^2``.
    """

    if not 0.0 < kinetic_energy_retention <= 1.0:
        raise ValueError("kinetic_energy_retention must lie in (0, 1]")
    incoming = np.asarray(incoming_velocity, dtype=float)
    unit_normal = np.asarray(normal, dtype=float)
    unit_normal = unit_normal / np.linalg.norm(unit_normal)
    reflected = incoming - (1.0 + restitution) * float(
        np.dot(incoming, unit_normal)
    ) * unit_normal
    return np.sqrt(kinetic_energy_retention) * reflected
