"""Ensemble observables defined in the model specification."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .tracer import TrajectoryResult


FAILURE_STATUSES = {"stalled", "max_events", "max_time", "numerical_failure"}


@dataclass(frozen=True)
class SimulationSummary:
    n_trajectories: int
    n_valid_initial_states: int
    survival_fraction: float
    collision_count_mean_all: float
    collision_count_mean_survivors: float
    wall_collision_count_mean_all: float
    wall_collision_count_mean_survivors: float
    collision_energy_loss_mean_all: float
    collision_energy_loss_mean_survivors: float
    fraction_with_ge_k_collisions: float
    collision_threshold: int
    escape_left_fraction: float
    escape_right_fraction: float
    failure_fraction: float
    stalled_fraction: float
    invalid_initial_fraction: float
    detector_x: np.ndarray
    status_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        def finite_or_none(value: float) -> float | None:
            return value if isfinite(value) else None

        return {
            "n_trajectories": self.n_trajectories,
            "n_valid_initial_states": self.n_valid_initial_states,
            "survival_fraction": self.survival_fraction,
            "collision_count_mean_all": finite_or_none(
                self.collision_count_mean_all
            ),
            "collision_count_mean_survivors": finite_or_none(
                self.collision_count_mean_survivors
            ),
            "wall_collision_count_mean_all": finite_or_none(
                self.wall_collision_count_mean_all
            ),
            "wall_collision_count_mean_survivors": finite_or_none(
                self.wall_collision_count_mean_survivors
            ),
            "collision_energy_loss_mean_all": finite_or_none(
                self.collision_energy_loss_mean_all
            ),
            "collision_energy_loss_mean_survivors": finite_or_none(
                self.collision_energy_loss_mean_survivors
            ),
            "fraction_with_ge_k_collisions": self.fraction_with_ge_k_collisions,
            "collision_threshold": self.collision_threshold,
            "escape_left_fraction": self.escape_left_fraction,
            "escape_right_fraction": self.escape_right_fraction,
            "failure_fraction": self.failure_fraction,
            "stalled_fraction": self.stalled_fraction,
            "invalid_initial_fraction": self.invalid_initial_fraction,
            "detector_x": self.detector_x.tolist(),
            "status_counts": self.status_counts,
        }


def summarize_trajectories(
    trajectories: list[TrajectoryResult] | tuple[TrajectoryResult, ...],
    collision_threshold: int,
) -> SimulationSummary:
    if not trajectories:
        raise ValueError("at least one trajectory is required")

    total = len(trajectories)
    statuses = [trajectory.status for trajectory in trajectories]
    status_counts = {status: statuses.count(status) for status in sorted(set(statuses))}
    valid = [
        trajectory
        for trajectory in trajectories
        if trajectory.status != "invalid_initial_state"
    ]
    survivors = [
        trajectory for trajectory in trajectories if trajectory.status == "detected_inside"
    ]
    detector_x = np.asarray(
        [trajectory.x_detector for trajectory in survivors], dtype=float
    )

    collision_count_mean_all = (
        float(np.mean([trajectory.n_collisions for trajectory in valid]))
        if valid
        else float("nan")
    )
    collision_count_mean_survivors = (
        float(np.mean([trajectory.n_collisions for trajectory in survivors]))
        if survivors
        else float("nan")
    )
    wall_collision_count_mean_all = (
        float(np.mean([trajectory.n_wall_collisions for trajectory in valid]))
        if valid
        else float("nan")
    )
    wall_collision_count_mean_survivors = (
        float(np.mean([trajectory.n_wall_collisions for trajectory in survivors]))
        if survivors
        else float("nan")
    )
    collision_energy_loss_mean_all = (
        float(
            np.mean(
                [trajectory.total_collision_energy_loss for trajectory in valid]
            )
        )
        if valid
        else float("nan")
    )
    collision_energy_loss_mean_survivors = (
        float(
            np.mean(
                [
                    trajectory.total_collision_energy_loss
                    for trajectory in survivors
                ]
            )
        )
        if survivors
        else float("nan")
    )
    denominator_valid = len(valid)
    fraction_with_threshold = (
        sum(
            trajectory.n_collisions >= collision_threshold
            for trajectory in valid
        )
        / denominator_valid
        if denominator_valid
        else 0.0
    )

    return SimulationSummary(
        n_trajectories=total,
        n_valid_initial_states=denominator_valid,
        survival_fraction=len(survivors) / total,
        collision_count_mean_all=collision_count_mean_all,
        collision_count_mean_survivors=collision_count_mean_survivors,
        wall_collision_count_mean_all=wall_collision_count_mean_all,
        wall_collision_count_mean_survivors=wall_collision_count_mean_survivors,
        collision_energy_loss_mean_all=collision_energy_loss_mean_all,
        collision_energy_loss_mean_survivors=(
            collision_energy_loss_mean_survivors
        ),
        fraction_with_ge_k_collisions=fraction_with_threshold,
        collision_threshold=collision_threshold,
        escape_left_fraction=status_counts.get("escaped_left", 0) / total,
        escape_right_fraction=status_counts.get("escaped_right", 0) / total,
        failure_fraction=sum(status_counts.get(status, 0) for status in FAILURE_STATUSES)
        / total,
        stalled_fraction=status_counts.get("stalled", 0) / total,
        invalid_initial_fraction=status_counts.get("invalid_initial_state", 0) / total,
        detector_x=detector_x,
        status_counts=status_counts,
    )
