"""High-level one-call simulation API and result persistence."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import SimulationConfig
from .geometry import SideEnvelope, make_side_envelope, triangular_peg_centers
from .observables import SimulationSummary, summarize_trajectories
from .tracer import TrajectoryResult, trace_trajectory


_WORKER_CONFIG: SimulationConfig | None = None
_WORKER_PEG_CENTERS: np.ndarray | None = None
_WORKER_ENVELOPE: SideEnvelope | None = None


def _initialize_trajectory_worker(
    config: SimulationConfig,
    peg_centers: np.ndarray,
    envelope: SideEnvelope,
) -> None:
    global _WORKER_CONFIG, _WORKER_PEG_CENTERS, _WORKER_ENVELOPE
    _WORKER_CONFIG = config
    _WORKER_PEG_CENTERS = peg_centers
    _WORKER_ENVELOPE = envelope


def _trace_start_in_worker(x0: float) -> TrajectoryResult:
    if (
        _WORKER_CONFIG is None
        or _WORKER_PEG_CENTERS is None
        or _WORKER_ENVELOPE is None
    ):
        raise RuntimeError("trajectory worker was not initialized")
    return trace_trajectory(
        _WORKER_CONFIG,
        x0,
        peg_centers=_WORKER_PEG_CENTERS,
        envelope=_WORKER_ENVELOPE,
    )


@dataclass(frozen=True)
class SimulationRun:
    config: SimulationConfig
    start_positions: np.ndarray
    trajectories: tuple[TrajectoryResult, ...]
    summary: SimulationSummary
    simulation_seconds: float
    workers: int


def midpoint_start_positions(config: SimulationConfig) -> np.ndarray:
    """Return midpoint samples with bit-exact mirror pairs."""

    half_count = config.n_starts // 2
    step = config.start_width / config.n_starts
    if config.n_starts % 2 == 0:
        positive = (np.arange(half_count, dtype=float) + 0.5) * step
        return np.concatenate((-positive[::-1], positive))

    positive = np.arange(1, half_count + 1, dtype=float) * step
    return np.concatenate((-positive[::-1], np.array([0.0]), positive))


def run_simulation(
    config: SimulationConfig | None = None,
    *,
    output_directory: str | Path | None = None,
    make_plot: bool = False,
    workers: int = 1,
) -> SimulationRun:
    """Run the complete deterministic start ensemble with one function call.

    If ``output_directory`` is given, the configuration, summary, and compact
    numerical results are written there.  ``make_plot=True`` additionally
    saves a trajectory overview without opening a GUI.
    """

    if config is None:
        config = SimulationConfig()
    if make_plot and output_directory is None:
        raise ValueError("make_plot=True requires output_directory")
    if workers not in {1, 2}:
        raise ValueError("workers must be 1 or 2")

    simulation_started = perf_counter()
    peg_centers = triangular_peg_centers(config)
    envelope = make_side_envelope(config)
    starts = midpoint_start_positions(config)
    if workers == 1:
        trajectories = tuple(
            trace_trajectory(
                config,
                float(x0),
                peg_centers=peg_centers,
                envelope=envelope,
            )
            for x0 in starts
        )
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_trajectory_worker,
            initargs=(config, peg_centers, envelope),
        ) as executor:
            trajectories = tuple(
                executor.map(
                    _trace_start_in_worker,
                    (float(x0) for x0 in starts),
                    chunksize=8,
                )
            )
    summary = summarize_trajectories(trajectories, config.collision_threshold)
    simulation_seconds = perf_counter() - simulation_started
    run = SimulationRun(
        config=config,
        start_positions=starts,
        trajectories=trajectories,
        summary=summary,
        simulation_seconds=simulation_seconds,
        workers=workers,
    )

    if output_directory is not None:
        output_path = Path(output_directory)
        save_simulation_run(run, output_path)
        if make_plot:
            from .plotting import plot_simulation_overview

            figure, _ = plot_simulation_overview(run)
            figure.savefig(output_path / "trajectories.png", dpi=180, bbox_inches="tight")
            import matplotlib.pyplot as plt

            plt.close(figure)
    return run


def save_simulation_run(run: SimulationRun, output_directory: str | Path) -> None:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": run.config.to_dict(),
        "summary": run.summary.to_dict(),
        "timing": {
            "simulation_seconds": run.simulation_seconds,
            "workers": run.workers,
        },
    }
    (output_path / "run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    statuses = np.asarray([trajectory.status for trajectory in run.trajectories])
    detector_x = np.asarray(
        [
            np.nan if trajectory.x_detector is None else trajectory.x_detector
            for trajectory in run.trajectories
        ],
        dtype=float,
    )
    collision_counts = np.asarray(
        [trajectory.n_collisions for trajectory in run.trajectories], dtype=int
    )
    wall_collision_counts = np.asarray(
        [trajectory.n_wall_collisions for trajectory in run.trajectories], dtype=int
    )
    collision_energy_loss_total = np.asarray(
        [
            trajectory.total_collision_energy_loss
            for trajectory in run.trajectories
        ],
        dtype=float,
    )
    collision_trajectory: list[int] = []
    collision_time: list[float] = []
    collision_peg: list[int] = []
    collision_position: list[np.ndarray] = []
    collision_kinetic_energy_before: list[float] = []
    collision_kinetic_energy_after: list[float] = []
    collision_energy_loss: list[float] = []
    wall_collision_trajectory: list[int] = []
    wall_collision_time: list[float] = []
    wall_collision_side: list[str] = []
    wall_collision_position: list[np.ndarray] = []
    wall_collision_kinetic_energy_before: list[float] = []
    wall_collision_kinetic_energy_after: list[float] = []
    wall_collision_energy_loss: list[float] = []
    for trajectory_index, trajectory in enumerate(run.trajectories):
        for collision in trajectory.collisions:
            collision_trajectory.append(trajectory_index)
            collision_time.append(collision.time)
            collision_peg.append(collision.peg_index)
            collision_position.append(collision.position)
            collision_kinetic_energy_before.append(collision.kinetic_energy_before)
            collision_kinetic_energy_after.append(collision.kinetic_energy_after)
            collision_energy_loss.append(collision.energy_loss)
        for collision in trajectory.wall_collisions:
            wall_collision_trajectory.append(trajectory_index)
            wall_collision_time.append(collision.time)
            wall_collision_side.append(collision.side)
            wall_collision_position.append(collision.position)
            wall_collision_kinetic_energy_before.append(
                collision.kinetic_energy_before
            )
            wall_collision_kinetic_energy_after.append(
                collision.kinetic_energy_after
            )
            wall_collision_energy_loss.append(collision.energy_loss)

    np.savez_compressed(
        output_path / "run.npz",
        start_positions=run.start_positions,
        statuses=statuses,
        detector_x=detector_x,
        simulation_seconds=np.asarray(run.simulation_seconds, dtype=float),
        workers=np.asarray(run.workers, dtype=int),
        collision_counts=collision_counts,
        wall_collision_counts=wall_collision_counts,
        collision_energy_loss_total=collision_energy_loss_total,
        collision_trajectory=np.asarray(collision_trajectory, dtype=int),
        collision_time=np.asarray(collision_time, dtype=float),
        collision_peg=np.asarray(collision_peg, dtype=int),
        collision_kinetic_energy_before=np.asarray(
            collision_kinetic_energy_before, dtype=float
        ),
        collision_kinetic_energy_after=np.asarray(
            collision_kinetic_energy_after, dtype=float
        ),
        collision_energy_loss=np.asarray(collision_energy_loss, dtype=float),
        collision_position=(
            np.asarray(collision_position, dtype=float).reshape(-1, 2)
            if collision_position
            else np.empty((0, 2), dtype=float)
        ),
        wall_collision_trajectory=np.asarray(wall_collision_trajectory, dtype=int),
        wall_collision_time=np.asarray(wall_collision_time, dtype=float),
        wall_collision_side=np.asarray(wall_collision_side),
        wall_collision_kinetic_energy_before=np.asarray(
            wall_collision_kinetic_energy_before, dtype=float
        ),
        wall_collision_kinetic_energy_after=np.asarray(
            wall_collision_kinetic_energy_after, dtype=float
        ),
        wall_collision_energy_loss=np.asarray(
            wall_collision_energy_loss, dtype=float
        ),
        wall_collision_position=(
            np.asarray(wall_collision_position, dtype=float).reshape(-1, 2)
            if wall_collision_position
            else np.empty((0, 2), dtype=float)
        ),
    )
