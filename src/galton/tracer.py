"""Event-driven trajectory tracing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .ballistics import (
    detector_crossing_time,
    position_at,
    specific_energy,
    velocity_at,
)
from .boundaries import first_boundary_exit
from .collisions import (
    collision_normal,
    find_next_collision,
    reflect_velocity,
)
from .config import SimulationConfig
from .geometry import SideEnvelope, make_side_envelope, triangular_peg_centers


FloatArray = NDArray[np.float64]
TrajectoryStatus = Literal[
    "detected_inside",
    "escaped_left",
    "escaped_right",
    "invalid_initial_state",
    "stalled",
    "max_events",
    "max_time",
    "numerical_failure",
]


@dataclass(frozen=True)
class TrajectorySegment:
    start_time: float
    duration: float
    start_position: FloatArray
    start_velocity: FloatArray
    end_position: FloatArray
    end_velocity: FloatArray
    event: str


@dataclass(frozen=True)
class CollisionRecord:
    time: float
    peg_index: int
    position: FloatArray
    velocity_before: FloatArray
    velocity_after: FloatArray
    normal: FloatArray
    incoming_normal_velocity: float
    kinetic_energy_before: float
    kinetic_energy_after: float
    energy_loss: float
    energy_error: float


@dataclass(frozen=True)
class WallCollisionRecord:
    time: float
    side: str
    position: FloatArray
    velocity_before: FloatArray
    velocity_after: FloatArray
    normal: FloatArray
    outgoing_normal_velocity: float
    kinetic_energy_before: float
    kinetic_energy_after: float
    energy_loss: float
    energy_error: float


@dataclass(frozen=True)
class TrajectoryResult:
    x0: float
    status: TrajectoryStatus
    x_detector: float | None
    total_time: float
    collisions: tuple[CollisionRecord, ...]
    wall_collisions: tuple[WallCollisionRecord, ...]
    segments: tuple[TrajectorySegment, ...]
    max_energy_error: float
    message: str | None = None

    @property
    def n_collisions(self) -> int:
        return len(self.collisions)

    @property
    def n_wall_collisions(self) -> int:
        return len(self.wall_collisions)

    @property
    def total_collision_energy_loss(self) -> float:
        """Total specific kinetic energy removed by peg and wall impacts."""

        return sum(collision.energy_loss for collision in self.collisions) + sum(
            collision.energy_loss for collision in self.wall_collisions
        )


def _make_segment(
    start_time: float,
    duration: float,
    position: FloatArray,
    velocity: FloatArray,
    acceleration: FloatArray,
    event: str,
) -> TrajectorySegment:
    return TrajectorySegment(
        start_time=start_time,
        duration=duration,
        start_position=position.copy(),
        start_velocity=velocity.copy(),
        end_position=position_at(position, velocity, acceleration, duration),
        end_velocity=velocity_at(velocity, acceleration, duration),
        event=event,
    )


def trace_trajectory(
    config: SimulationConfig,
    x0: float,
    *,
    peg_centers: FloatArray | None = None,
    envelope: SideEnvelope | None = None,
) -> TrajectoryResult:
    """Trace one ball until detection, escape, or an explicit failure limit."""

    if peg_centers is None:
        peg_centers = triangular_peg_centers(config)
    if envelope is None:
        envelope = make_side_envelope(config)

    position = np.array([x0, config.release_y], dtype=float)
    velocity = np.array(
        [config.initial_velocity_x, config.initial_velocity_y], dtype=float
    )
    acceleration = np.array([0.0, -config.gravity], dtype=float)
    radius = config.effective_radius
    length_tolerance = config.length_tolerance
    time_tolerance = config.scaled_time_tolerance
    velocity_tolerance = config.scaled_velocity_tolerance
    simultaneous_tolerance = config.simultaneous_tolerance
    contact_position_tolerance = max(100.0 * length_tolerance, 1.0e-11 * radius)

    if peg_centers.size:
        distances = np.linalg.norm(peg_centers - position, axis=1)
        if bool(np.any(distances <= radius + length_tolerance)):
            return TrajectoryResult(
                x0=x0,
                status="invalid_initial_state",
                x_detector=None,
                total_time=0.0,
                collisions=(),
                wall_collisions=(),
                segments=(),
                max_energy_error=0.0,
                message="initial ball center lies on or inside a collision circle",
            )

    initial_energy = specific_energy(position, velocity, config.gravity)
    expected_energy = initial_energy
    energy_scale = max(1.0, abs(initial_energy), config.gravity * config.start_width)
    allowed_energy_error = config.energy_tolerance * energy_scale
    elapsed = 0.0
    collisions: list[CollisionRecord] = []
    wall_collisions: list[WallCollisionRecord] = []
    segments: list[TrajectorySegment] = []
    max_energy_error = 0.0
    event_count = 0
    low_energy_peg: int | None = None
    low_energy_position: FloatArray | None = None
    low_energy_duration: float | None = None
    low_energy_plateau_count = 0
    stall_energy_threshold = config.gravity * length_tolerance

    def is_stalled_peg_contact(
        peg_index: int,
        contact_position: FloatArray,
        kinetic_energy_after: float,
        flight_duration: float,
    ) -> bool:
        """Recognize a numerically unresolved inelastic limit cycle."""

        nonlocal low_energy_peg, low_energy_position
        nonlocal low_energy_duration, low_energy_plateau_count
        if kinetic_energy_after > stall_energy_threshold:
            low_energy_peg = None
            low_energy_position = None
            low_energy_duration = None
            low_energy_plateau_count = 0
            return False

        same_unresolved_contact = (
            low_energy_peg == peg_index
            and low_energy_position is not None
            and float(np.linalg.norm(contact_position - low_energy_position))
            <= 10.0 * length_tolerance
        )
        duration_has_plateaued = (
            same_unresolved_contact
            and low_energy_duration is not None
            and abs(flight_duration - low_energy_duration)
            <= 0.05 * max(flight_duration, low_energy_duration)
        )
        low_energy_plateau_count = (
            low_energy_plateau_count + 1 if duration_has_plateaued else 0
        )
        low_energy_peg = peg_index
        low_energy_position = contact_position.copy()
        low_energy_duration = flight_duration
        return low_energy_plateau_count >= 4

    def finish(
        status: TrajectoryStatus,
        *,
        detector_x: float | None = None,
        message: str | None = None,
    ) -> TrajectoryResult:
        return TrajectoryResult(
            x0=x0,
            status=status,
            x_detector=detector_x,
            total_time=elapsed,
            collisions=tuple(collisions),
            wall_collisions=tuple(wall_collisions),
            segments=tuple(segments),
            max_energy_error=max_energy_error,
            message=message,
        )

    while True:
        remaining_time = config.max_time - elapsed
        if remaining_time <= time_tolerance:
            return finish("max_time", message="maximum total flight time reached")

        detector_time = detector_crossing_time(
            position,
            velocity,
            config.gravity,
            config.resolved_detector_y,
            minimum_time=time_tolerance,
            velocity_tolerance=velocity_tolerance,
            imag_tolerance=config.root_imag_tolerance,
            maximum_time=remaining_time,
        )
        boundary_exit = first_boundary_exit(
            position,
            velocity,
            acceleration,
            envelope,
            maximum_time=remaining_time,
            time_tolerance=time_tolerance,
            position_tolerance=length_tolerance,
            imag_tolerance=config.root_imag_tolerance,
        )
        # A peg contact after the first free-flight detector or boundary event
        # is irrelevant.  Keeping the simultaneity margin preserves the event
        # precedence below while sharply reducing the set of possible pegs.
        collision_maximum_time = remaining_time
        if detector_time is not None:
            collision_maximum_time = min(
                collision_maximum_time,
                detector_time + simultaneous_tolerance,
            )
        if boundary_exit is not None:
            collision_maximum_time = min(
                collision_maximum_time,
                boundary_exit.time + simultaneous_tolerance,
            )
        collision = find_next_collision(
            position,
            velocity,
            acceleration,
            peg_centers,
            radius,
            minimum_time=time_tolerance,
            maximum_time=collision_maximum_time,
            imag_tolerance=config.root_imag_tolerance,
            position_tolerance=contact_position_tolerance,
            velocity_tolerance=velocity_tolerance,
            simultaneous_tolerance=simultaneous_tolerance,
        )

        candidate_times = [remaining_time]
        if collision is not None:
            candidate_times.append(collision.time)
        if detector_time is not None:
            candidate_times.append(detector_time)
        if boundary_exit is not None:
            candidate_times.append(boundary_exit.time)
        first_time = min(candidate_times)

        # A contact at the side envelope can prevent the free-flight path from
        # crossing it, so collisions take precedence within the simultaneity
        # tolerance.  A detector touch is inside unless it has already crossed.
        if collision is not None and collision.time <= first_time + simultaneous_tolerance:
            event = "collision"
            event_time = collision.time
        elif detector_time is not None and detector_time <= first_time + simultaneous_tolerance:
            event = "detector"
            event_time = detector_time
        elif boundary_exit is not None and boundary_exit.time <= first_time + simultaneous_tolerance:
            event = "boundary"
            event_time = boundary_exit.time
        else:
            event = "max_time"
            event_time = remaining_time

        segment = _make_segment(
            elapsed,
            event_time,
            position,
            velocity,
            acceleration,
            event,
        )
        if config.store_trajectories:
            segments.append(segment)
        elapsed += event_time

        if event == "max_time":
            position = segment.end_position
            velocity = segment.end_velocity
            return finish("max_time", message="maximum total flight time reached")

        if event == "detector":
            # If the detector and a genuine outside interval coincide within
            # tolerance, an already-outside point is still an escape.
            side = envelope.side_outside(segment.end_position, length_tolerance)
            if side is not None:
                return finish(f"escaped_{side}")  # type: ignore[arg-type]
            return finish("detected_inside", detector_x=float(segment.end_position[0]))

        if event == "boundary":
            assert boundary_exit is not None
            if config.boundary_behavior == "absorbing":
                return finish(f"escaped_{boundary_exit.side}")  # type: ignore[arg-type]

            contact_position = segment.end_position.copy()
            contact_position[0] = (
                float(envelope.half_width(float(contact_position[1])))
                * (1.0 if boundary_exit.side == "right" else -1.0)
            )
            velocity_before = segment.end_velocity
            outward_normal_velocity = float(
                np.dot(velocity_before, boundary_exit.normal)
            )
            if outward_normal_velocity <= velocity_tolerance:
                return finish(
                    "numerical_failure",
                    message="selected wall contact is not outward directed",
                )
            energy_before = specific_energy(
                contact_position,
                velocity_before,
                config.gravity,
            )
            kinetic_energy_before = 0.5 * float(
                np.dot(velocity_before, velocity_before)
            )
            undamped_velocity_after = reflect_velocity(
                velocity_before,
                boundary_exit.normal,
                restitution=config.wall_restitution,
            )
            velocity_after = reflect_velocity(
                velocity_before,
                boundary_exit.normal,
                restitution=config.wall_restitution,
                kinetic_energy_retention=config.kinetic_energy_retention,
            )
            expected_kinetic_energy_after = (
                config.kinetic_energy_retention
                * 0.5
                * float(np.dot(undamped_velocity_after, undamped_velocity_after))
            )
            kinetic_energy_after = 0.5 * float(
                np.dot(velocity_after, velocity_after)
            )
            modeled_energy_loss = (
                kinetic_energy_before - expected_kinetic_energy_after
            )
            energy_loss = max(0.0, kinetic_energy_before - kinetic_energy_after)
            expected_energy_after = expected_energy - modeled_energy_loss
            energy_after = specific_energy(
                contact_position,
                velocity_after,
                config.gravity,
            )
            energy_error = max(
                abs(energy_before - expected_energy),
                abs(kinetic_energy_after - expected_kinetic_energy_after),
                abs(energy_after - expected_energy_after),
            )
            max_energy_error = max(max_energy_error, energy_error)
            wall_collisions.append(
                WallCollisionRecord(
                    time=elapsed,
                    side=boundary_exit.side,
                    position=contact_position.copy(),
                    velocity_before=velocity_before.copy(),
                    velocity_after=velocity_after.copy(),
                    normal=boundary_exit.normal.copy(),
                    outgoing_normal_velocity=outward_normal_velocity,
                    kinetic_energy_before=kinetic_energy_before,
                    kinetic_energy_after=kinetic_energy_after,
                    energy_loss=energy_loss,
                    energy_error=energy_error,
                )
            )
            low_energy_peg = None
            low_energy_position = None
            low_energy_duration = None
            low_energy_plateau_count = 0
            expected_energy = expected_energy_after
            event_count += 1
            if energy_error > allowed_energy_error:
                return finish(
                    "numerical_failure",
                    message=(
                        "wall-impact energy check failed: "
                        f"error={energy_error:.3e}, "
                        f"tolerance={allowed_energy_error:.3e}"
                    ),
                )
            if event_count >= config.max_events:
                return finish("max_events", message="maximum collision count reached")
            position = contact_position
            velocity = velocity_after
            continue

        assert collision is not None
        if len(collision.peg_indices) != 1:
            return finish(
                "numerical_failure",
                message=(
                    "unresolved simultaneous contact with pegs "
                    f"{collision.peg_indices}"
                ),
            )

        peg_index = collision.peg_indices[0]
        contact_position = segment.end_position
        velocity_before = segment.end_velocity
        normal = collision_normal(contact_position, peg_centers[peg_index])
        incoming_normal_velocity = float(np.dot(velocity_before, normal))
        if incoming_normal_velocity >= -velocity_tolerance:
            return finish(
                "numerical_failure",
                message="selected contact is not an incoming collision",
            )
        energy_before = specific_energy(
            contact_position,
            velocity_before,
            config.gravity,
        )
        kinetic_energy_before = 0.5 * float(
            np.dot(velocity_before, velocity_before)
        )
        undamped_velocity_after = reflect_velocity(
            velocity_before,
            normal,
            restitution=config.restitution,
        )
        velocity_after = reflect_velocity(
            velocity_before,
            normal,
            restitution=config.restitution,
            kinetic_energy_retention=config.kinetic_energy_retention,
        )
        expected_kinetic_energy_after = (
            config.kinetic_energy_retention
            * 0.5
            * float(np.dot(undamped_velocity_after, undamped_velocity_after))
        )
        kinetic_energy_after = 0.5 * float(np.dot(velocity_after, velocity_after))
        modeled_energy_loss = kinetic_energy_before - expected_kinetic_energy_after
        energy_loss = max(0.0, kinetic_energy_before - kinetic_energy_after)
        expected_energy_after = expected_energy - modeled_energy_loss
        energy_after = specific_energy(contact_position, velocity_after, config.gravity)
        energy_error = max(
            abs(energy_before - expected_energy),
            abs(kinetic_energy_after - expected_kinetic_energy_after),
            abs(energy_after - expected_energy_after),
        )
        max_energy_error = max(max_energy_error, energy_error)
        collisions.append(
            CollisionRecord(
                time=elapsed,
                peg_index=peg_index,
                position=contact_position.copy(),
                velocity_before=velocity_before.copy(),
                velocity_after=velocity_after.copy(),
                normal=normal.copy(),
                incoming_normal_velocity=incoming_normal_velocity,
                kinetic_energy_before=kinetic_energy_before,
                kinetic_energy_after=kinetic_energy_after,
                energy_loss=energy_loss,
                energy_error=energy_error,
            )
        )
        expected_energy = expected_energy_after
        event_count += 1

        if energy_error > allowed_energy_error:
            return finish(
                "numerical_failure",
                message=(
                    "peg-impact energy check failed: "
                    f"error={energy_error:.3e}, tolerance={allowed_energy_error:.3e}"
                ),
            )
        if is_stalled_peg_contact(
            peg_index,
            contact_position,
            kinetic_energy_after,
            event_time,
        ):
            return finish(
                "stalled",
                message=(
                    "inelastic collapse below spatial resolution at peg "
                    f"{peg_index}"
                ),
            )
        if event_count >= config.max_events:
            return finish("max_events", message="maximum collision count reached")

        position = contact_position
        velocity = velocity_after
