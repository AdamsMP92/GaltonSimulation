"""Reproducible reference run."""

from __future__ import annotations

import json

from galton import run_simulation, simple_galton_config


WORKERS = 2  # Erlaubt sind 1 oder 2.


# Alle *_ratio-Werte sind explizite Längenverhältnisse zur Startbreite W.
# Der Wrapper setzt nur v0=0 und W=1 als geometrisch-dynamische Standards.
EXAMPLE_CONFIG = simple_galton_config(
    rows=36,
    peg_diameter_ratio=0.1,
    ball_diameter_ratio=0.3,
    peg_spacing_ratio=0.8,
    release_height_ratio=1.0,
    detector_gap_ratio=2.0,
    detector_width_ratio=29.0,
    boundary_behavior="reflecting",
    starts=3000,
    kinetic_energy_retention=0.99,  # w_e: 1 elastisch, < 1 Energieverlust
    max_events=10000,
    max_time=200,
)


if __name__ == "__main__":
    result = run_simulation(
        EXAMPLE_CONFIG,
        output_directory="results/latest",
        make_plot=True,
        workers=WORKERS,
    )
    summary = result.summary.to_dict()
    summary["detector_x"] = f"{len(result.summary.detector_x)} values saved in run.npz"
    summary["simulation_seconds"] = result.simulation_seconds
    summary["workers"] = result.workers
    print(json.dumps(summary, indent=2, sort_keys=True))
