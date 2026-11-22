"""Deterministic scans in effective radius rho and spacing delta."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike

from .config import SimulationConfig
from .simulation import run_simulation


@dataclass(frozen=True)
class ScanResult:
    rho: np.ndarray
    delta: np.ndarray
    survival_fraction: np.ndarray
    collision_count_mean_all: np.ndarray
    fraction_with_ge_k_collisions: np.ndarray
    failure_fraction: np.ndarray
    keep_h_over_d: bool
    base_config: SimulationConfig


def scan_rho_delta(
    rho_values: ArrayLike,
    delta_values: ArrayLike,
    base_config: SimulationConfig | None = None,
    *,
    n_starts: int | None = None,
    keep_h_over_d: bool = True,
) -> ScanResult:
    """Scan ``rho=(R_peg+R_ball)/W`` and ``delta=d/W`` sequentially.

    ``keep_h_over_d=True`` is the reference slice requested in the model
    specification.  Setting it to ``False`` keeps ``eta=h/W`` fixed instead.
    """

    if base_config is None:
        base_config = SimulationConfig()
    rho = np.asarray(rho_values, dtype=float)
    delta = np.asarray(delta_values, dtype=float)
    if rho.ndim != 1 or delta.ndim != 1 or rho.size == 0 or delta.size == 0:
        raise ValueError("rho_values and delta_values must be non-empty 1D arrays")
    if np.any(rho <= 0.0) or np.any(delta <= 0.0):
        raise ValueError("rho and delta values must be positive")

    shape = (rho.size, delta.size)
    survival = np.empty(shape, dtype=float)
    collision_mean = np.empty(shape, dtype=float)
    collision_fraction = np.empty(shape, dtype=float)
    failure = np.empty(shape, dtype=float)
    radius_split = base_config.peg_radius / base_config.effective_radius
    h_over_d = base_config.vertical_spacing / base_config.horizontal_spacing

    for rho_index, rho_value in enumerate(rho):
        effective_radius = float(rho_value) * base_config.start_width
        for delta_index, delta_value in enumerate(delta):
            spacing = float(delta_value) * base_config.start_width
            vertical_spacing = (
                h_over_d * spacing if keep_h_over_d else base_config.vertical_spacing
            )
            config = replace(
                base_config,
                peg_radius=radius_split * effective_radius,
                ball_radius=(1.0 - radius_split) * effective_radius,
                horizontal_spacing=spacing,
                vertical_spacing=vertical_spacing,
                n_starts=base_config.n_starts if n_starts is None else n_starts,
                store_trajectories=False,
            )
            summary = run_simulation(config).summary
            survival[rho_index, delta_index] = summary.survival_fraction
            collision_mean[rho_index, delta_index] = summary.collision_count_mean_all
            collision_fraction[rho_index, delta_index] = (
                summary.fraction_with_ge_k_collisions
            )
            failure[rho_index, delta_index] = summary.failure_fraction

    return ScanResult(
        rho=rho,
        delta=delta,
        survival_fraction=survival,
        collision_count_mean_all=collision_mean,
        fraction_with_ge_k_collisions=collision_fraction,
        failure_fraction=failure,
        keep_h_over_d=keep_h_over_d,
        base_config=base_config,
    )
