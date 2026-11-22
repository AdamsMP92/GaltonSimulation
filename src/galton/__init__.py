"""Public API for the deterministic Galton-board simulation."""

from .config import BoundaryBehavior, BoundaryMode, SimulationConfig
from .distribution_fit import GaussianKDEComparison, compare_gaussian_and_kde
from .observables import SimulationSummary
from .plotting import (
    plot_detector_statistics,
    plot_gaussian_kde_comparison,
    plot_simulation_overview,
    plot_start_grid_scan,
    plot_trajectories,
)
from .presets import simple_galton_config
from .scans import ScanResult, scan_rho_delta
from .simulation import SimulationRun, run_simulation, save_simulation_run
from .start_grid_scan import (
    StartGridLevel,
    StartGridScanResult,
    count_separated_endpoints,
    run_start_grid_scan,
    save_start_grid_scan,
)
from .tracer import (
    CollisionRecord,
    TrajectoryResult,
    TrajectorySegment,
    WallCollisionRecord,
    trace_trajectory,
)

__all__ = [
    "BoundaryBehavior",
    "BoundaryMode",
    "CollisionRecord",
    "GaussianKDEComparison",
    "ScanResult",
    "SimulationConfig",
    "SimulationRun",
    "SimulationSummary",
    "StartGridLevel",
    "StartGridScanResult",
    "TrajectoryResult",
    "TrajectorySegment",
    "WallCollisionRecord",
    "compare_gaussian_and_kde",
    "plot_detector_statistics",
    "plot_gaussian_kde_comparison",
    "plot_simulation_overview",
    "plot_start_grid_scan",
    "plot_trajectories",
    "run_simulation",
    "run_start_grid_scan",
    "save_simulation_run",
    "scan_rho_delta",
    "save_start_grid_scan",
    "simple_galton_config",
    "trace_trajectory",
    "count_separated_endpoints",
]
