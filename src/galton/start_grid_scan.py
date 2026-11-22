"""Convergence scan over deterministic, equidistant initial-position grids."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from .config import SimulationConfig
from .simulation import run_simulation


@dataclass(frozen=True)
class StartGridLevel:
    """Results for one equidistant midpoint grid of initial positions."""

    n_starts: int
    start_spacing: float
    start_positions: np.ndarray
    detector_x: np.ndarray
    statuses: np.ndarray
    detected_count: int
    stalled_count: int
    exact_endpoint_count: int
    resolved_endpoint_count: int
    median_endpoint_gap: float
    largest_endpoint_gap: float
    simulation_seconds: float
    ks_distance_to_finest: float = float("nan")
    wasserstein_distance_to_finest: float = float("nan")

    @property
    def detected_x(self) -> np.ndarray:
        return self.detector_x[np.isfinite(self.detector_x)]

    def to_dict(self) -> dict[str, object]:
        def finite_or_none(value: float) -> float | None:
            return value if isfinite(value) else None

        return {
            "n_starts": self.n_starts,
            "start_spacing": self.start_spacing,
            "detected_count": self.detected_count,
            "stalled_count": self.stalled_count,
            "exact_endpoint_count": self.exact_endpoint_count,
            "resolved_endpoint_count": self.resolved_endpoint_count,
            "median_endpoint_gap": finite_or_none(self.median_endpoint_gap),
            "largest_endpoint_gap": finite_or_none(self.largest_endpoint_gap),
            "simulation_seconds": self.simulation_seconds,
            "ks_distance_to_finest": finite_or_none(
                self.ks_distance_to_finest
            ),
            "wasserstein_distance_to_finest": finite_or_none(
                self.wasserstein_distance_to_finest
            ),
        }


@dataclass(frozen=True)
class StartGridScanResult:
    """Results of successively refined deterministic start grids."""

    base_config: SimulationConfig
    levels: tuple[StartGridLevel, ...]
    endpoint_tolerance_ratio: float
    workers: int
    total_seconds: float

    @property
    def endpoint_tolerance(self) -> float:
        return self.endpoint_tolerance_ratio * self.base_config.start_width

    def to_dict(self) -> dict[str, object]:
        return {
            "base_config": self.base_config.to_dict(),
            "grid_sizes": [level.n_starts for level in self.levels],
            "endpoint_tolerance_ratio": self.endpoint_tolerance_ratio,
            "endpoint_tolerance": self.endpoint_tolerance,
            "workers": self.workers,
            "total_seconds": self.total_seconds,
            "levels": [level.to_dict() for level in self.levels],
        }


def count_separated_endpoints(values: np.ndarray, tolerance: float) -> int:
    """Count ordered endpoint groups of diameter at most ``tolerance``.

    Unlike histogram occupancy, this count has no bin origin.
    """

    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.size == 0:
        return 0

    count = 1
    group_start = float(ordered[0])
    for value in ordered[1:]:
        if float(value) - group_start > tolerance:
            count += 1
            group_start = float(value)
    return count


def run_start_grid_scan(
    base_config: SimulationConfig,
    grid_sizes: tuple[int, ...] | list[int],
    *,
    endpoint_tolerance_ratio: float,
    workers: int = 1,
    output_directory: str | Path | None = None,
    make_plot: bool = False,
) -> StartGridScanResult:
    """Run increasingly fine, equidistant midpoint grids over the start width."""

    sizes = tuple(int(size) for size in grid_sizes)
    if not sizes or any(size <= 1 for size in sizes):
        raise ValueError("grid_sizes must contain integers greater than one")
    if sizes != tuple(sorted(set(sizes))):
        raise ValueError("grid_sizes must be unique and strictly increasing")
    if endpoint_tolerance_ratio <= 0.0:
        raise ValueError("endpoint_tolerance_ratio must be positive")
    if make_plot and output_directory is None:
        raise ValueError("make_plot=True requires output_directory")

    scan_started = perf_counter()
    endpoint_tolerance = endpoint_tolerance_ratio * base_config.start_width
    levels: list[StartGridLevel] = []
    for n_starts in sizes:
        config = replace(
            base_config,
            n_starts=n_starts,
            store_trajectories=False,
        )
        run = run_simulation(config, workers=workers)
        detector_x = np.asarray(
            [
                np.nan if trajectory.x_detector is None else trajectory.x_detector
                for trajectory in run.trajectories
            ],
            dtype=float,
        )
        statuses = np.asarray(
            [trajectory.status for trajectory in run.trajectories]
        )
        detected_x = detector_x[np.isfinite(detector_x)]
        ordered = np.sort(detected_x)
        gaps = np.diff(ordered)
        levels.append(
            StartGridLevel(
                n_starts=n_starts,
                start_spacing=base_config.start_width / n_starts,
                start_positions=run.start_positions.copy(),
                detector_x=detector_x,
                statuses=statuses,
                detected_count=int(detected_x.size),
                stalled_count=int(np.count_nonzero(statuses == "stalled")),
                exact_endpoint_count=int(np.unique(detected_x).size),
                resolved_endpoint_count=count_separated_endpoints(
                    detected_x,
                    endpoint_tolerance,
                ),
                median_endpoint_gap=(
                    float(np.median(gaps)) if gaps.size else float("nan")
                ),
                largest_endpoint_gap=(
                    float(np.max(gaps)) if gaps.size else float("nan")
                ),
                simulation_seconds=run.simulation_seconds,
            )
        )

    finest = levels[-1].detected_x
    for index, level in enumerate(levels):
        detected_x = level.detected_x
        if detected_x.size and finest.size:
            ks_distance = (
                0.0
                if index == len(levels) - 1
                else float(ks_2samp(detected_x, finest).statistic)
            )
            wasserstein = (
                0.0
                if index == len(levels) - 1
                else float(wasserstein_distance(detected_x, finest))
                / base_config.start_width
            )
        else:
            ks_distance = float("nan")
            wasserstein = float("nan")
        levels[index] = replace(
            level,
            ks_distance_to_finest=ks_distance,
            wasserstein_distance_to_finest=wasserstein,
        )

    result = StartGridScanResult(
        base_config=base_config,
        levels=tuple(levels),
        endpoint_tolerance_ratio=endpoint_tolerance_ratio,
        workers=workers,
        total_seconds=perf_counter() - scan_started,
    )
    if output_directory is not None:
        output_path = Path(output_directory)
        save_start_grid_scan(result, output_path)
        if make_plot:
            from .plotting import plot_start_grid_scan

            figure, _ = plot_start_grid_scan(result)
            figure.savefig(
                output_path / "start_grid_scan.png",
                dpi=180,
                bbox_inches="tight",
            )
            import matplotlib.pyplot as plt

            plt.close(figure)
    return result


def save_start_grid_scan(
    result: StartGridScanResult,
    output_directory: str | Path,
) -> None:
    """Persist scan metadata and aligned start/end coordinates."""

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "start_grid_scan.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    arrays: dict[str, np.ndarray] = {}
    for level in result.levels:
        suffix = str(level.n_starts)
        arrays[f"start_positions_{suffix}"] = level.start_positions
        arrays[f"detector_x_{suffix}"] = level.detector_x
        arrays[f"statuses_{suffix}"] = level.statuses
    np.savez_compressed(output_path / "start_grid_scan.npz", **arrays)
