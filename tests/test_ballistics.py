from __future__ import annotations

from math import sqrt

import numpy as np

from galton.ballistics import detector_crossing_time, position_at
from galton.collisions import peg_contact_times, reflect_velocity


def test_free_fall_detector_crossing_matches_analytic_solution() -> None:
    position = np.array([0.25, 2.0])
    velocity = np.array([0.4, 0.3])
    gravity = 1.5
    detector_y = -1.0
    expected = (velocity[1] + sqrt(velocity[1] ** 2 + 2 * gravity * 3.0)) / gravity

    actual = detector_crossing_time(
        position,
        velocity,
        gravity,
        detector_y,
        minimum_time=1.0e-12,
        velocity_tolerance=1.0e-12,
        imag_tolerance=1.0e-10,
    )

    assert actual is not None
    assert actual == pytest_approx(expected)
    detector_position = position_at(position, velocity, [0.0, -gravity], actual)
    assert detector_position[1] == pytest_approx(detector_y)


def test_central_elastic_collision_reverses_normal_velocity() -> None:
    incoming = np.array([0.0, -2.0])
    outgoing = reflect_velocity(incoming, [0.0, 1.0], restitution=1.0)
    np.testing.assert_allclose(outgoing, [0.0, 2.0])
    assert np.linalg.norm(outgoing) == pytest_approx(np.linalg.norm(incoming))


def test_oblique_collision_preserves_tangent_and_reverses_normal() -> None:
    incoming = np.array([3.0, -2.0])
    outgoing = reflect_velocity(incoming, [0.0, 1.0], restitution=1.0)
    np.testing.assert_allclose(outgoing, [3.0, 2.0])


def test_energy_retention_scales_complete_reflected_velocity() -> None:
    incoming = np.array([3.0, -4.0])
    outgoing = reflect_velocity(
        incoming,
        [0.0, 1.0],
        restitution=1.0,
        kinetic_energy_retention=0.64,
    )

    np.testing.assert_allclose(outgoing, [2.4, 3.2])
    assert float(np.dot(outgoing, outgoing)) == pytest_approx(
        0.64 * float(np.dot(incoming, incoming))
    )


def test_clear_flyby_does_not_create_collision() -> None:
    times = peg_contact_times(
        position=[-2.0, 2.0],
        velocity=[1.0, 0.0],
        acceleration=[0.0, 0.0],
        peg_center=[0.0, 0.0],
        effective_radius=0.5,
        minimum_time=1.0e-12,
        maximum_time=10.0,
        imag_tolerance=1.0e-10,
        position_tolerance=1.0e-9,
        velocity_tolerance=1.0e-12,
    )
    assert times == []


def pytest_approx(value: float):
    # Keeping pytest imports out of the production modules also makes these
    # equations readable as plain Python examples.
    import pytest

    return pytest.approx(value, rel=1.0e-11, abs=1.0e-11)
