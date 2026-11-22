"""Deterministic convergence scan over equidistant initial-position grids."""

from __future__ import annotations

import json

from galton import run_start_grid_scan
from run_single import EXAMPLE_CONFIG, WORKERS


GRID_SIZES = (100, 200, 400, 800, 1600)
ENDPOINT_TOLERANCE_RATIO = 0.01
OUTPUT_DIRECTORY = "results/start_grid_scan"


if __name__ == "__main__":
    result = run_start_grid_scan(
        EXAMPLE_CONFIG,
        GRID_SIZES,
        endpoint_tolerance_ratio=ENDPOINT_TOLERANCE_RATIO,
        workers=WORKERS,
        output_directory=OUTPUT_DIRECTORY,
        make_plot=True,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
