from __future__ import annotations

import numpy as np
import pytest

from galton.boundaries import first_boundary_exit
from galton.geometry import SideEnvelope


def test_boundary_is_checked_during_flight_not_only_at_collisions() -> None:
    envelope = SideEnvelope(y_top=0.0, half_width_at_top=0.5, expansion_per_drop=0.0)
    event = first_boundary_exit(
        position=[0.0, 0.5],
        velocity=[1.0, 0.0],
        acceleration=[0.0, -1.0],
        envelope=envelope,
        maximum_time=3.0,
        time_tolerance=1.0e-12,
        position_tolerance=1.0e-12,
        imag_tolerance=1.0e-10,
    )

    assert event is not None
    assert event.side == "right"
    # The path first enters the monitored y <= y_top region at t=1 and is
    # already outside there.
    assert event.time == pytest.approx(1.0)


def test_tangent_touch_without_outside_interval_is_not_escape() -> None:
    envelope = SideEnvelope(y_top=1.0, half_width_at_top=0.5, expansion_per_drop=0.0)
    event = first_boundary_exit(
        position=[0.0, 0.0],
        velocity=[1.0, 0.0],
        acceleration=[-2.0, 0.0],
        envelope=envelope,
        maximum_time=1.0,
        time_tolerance=1.0e-12,
        position_tolerance=1.0e-12,
        imag_tolerance=1.0e-10,
    )
    # x=t-t^2 reaches x=0.25, so the trajectory remains inside.
    assert event is None


def test_immediate_outside_state_reports_direction() -> None:
    envelope = SideEnvelope(y_top=0.0, half_width_at_top=0.5, expansion_per_drop=0.0)
    event = first_boundary_exit(
        position=np.array([-0.6, -1.0]),
        velocity=np.zeros(2),
        acceleration=np.array([0.0, -1.0]),
        envelope=envelope,
        maximum_time=1.0,
        time_tolerance=1.0e-12,
        position_tolerance=1.0e-12,
        imag_tolerance=1.0e-10,
    )
    assert event is not None
    assert event.time == 0.0
    assert event.side == "left"


def test_piecewise_funnel_uses_sloped_then_vertical_boundary() -> None:
    envelope = SideEnvelope(
        y_top=1.0,
        half_width_at_top=0.5,
        expansion_per_drop=0.5,
        transition_y=0.0,
        lower_half_width=1.0,
    )
    assert float(envelope.half_width(0.5)) == pytest.approx(0.75)
    assert float(envelope.half_width(0.0)) == pytest.approx(1.0)
    assert float(envelope.half_width(-0.5)) == pytest.approx(1.0)

    sloped_contact = first_boundary_exit(
        position=[0.0, 0.5],
        velocity=[2.0, 0.0],
        acceleration=[0.0, 0.0],
        envelope=envelope,
        maximum_time=1.0,
        time_tolerance=1.0e-12,
        position_tolerance=1.0e-12,
        imag_tolerance=1.0e-10,
    )
    assert sloped_contact is not None
    assert sloped_contact.time == pytest.approx(0.375)
    np.testing.assert_allclose(
        sloped_contact.normal,
        np.array([1.0, 0.5]) / np.sqrt(1.25),
    )

    vertical_contact = first_boundary_exit(
        position=[0.0, -0.5],
        velocity=[2.0, 0.0],
        acceleration=[0.0, 0.0],
        envelope=envelope,
        maximum_time=1.0,
        time_tolerance=1.0e-12,
        position_tolerance=1.0e-12,
        imag_tolerance=1.0e-10,
    )
    assert vertical_contact is not None
    assert vertical_contact.time == pytest.approx(0.5)
    np.testing.assert_allclose(vertical_contact.normal, [1.0, 0.0])


def test_funnel_can_include_vertical_entry_chute() -> None:
    envelope = SideEnvelope(
        y_top=1.0,
        half_width_at_top=0.5,
        expansion_per_drop=0.5,
        transition_y=0.0,
        lower_half_width=1.0,
        upper_half_width=0.5,
    )
    event = first_boundary_exit(
        position=[0.0, 1.5],
        velocity=[1.0, 0.0],
        acceleration=[0.0, 0.0],
        envelope=envelope,
        maximum_time=1.0,
        time_tolerance=1.0e-12,
        position_tolerance=1.0e-12,
        imag_tolerance=1.0e-10,
    )
    assert event is not None
    assert event.time == pytest.approx(0.5)
    assert event.side == "right"
    np.testing.assert_allclose(event.normal, [1.0, 0.0])
