"""Plotting helpers kept separate from the simulation dynamics."""

from __future__ import annotations

from collections.abc import Collection, Mapping

import numpy as np

from .distribution_fit import GaussianKDEComparison, compare_gaussian_and_kde
from .geometry import make_side_envelope, triangular_peg_centers
from .scans import ScanResult
from .simulation import SimulationRun
from .start_grid_scan import StartGridScanResult


STATUS_COLORS = {
    "detected_inside": "tab:blue",
    "escaped_left": "tab:orange",
    "escaped_right": "tab:orange",
    "invalid_initial_state": "tab:red",
    "stalled": "tab:olive",
    "max_events": "tab:purple",
    "max_time": "tab:brown",
    "numerical_failure": "tab:red",
}

STATUS_LABELS = {
    "detected_inside": "am Detektor",
    "escaped_left": "links ausgetreten",
    "escaped_right": "rechts ausgetreten",
    "invalid_initial_state": "ungültiger Start",
    "stalled": "inelastisch festgelaufen",
    "max_events": "Ereignisgrenze",
    "max_time": "Zeitgrenze",
    "numerical_failure": "numerischer Fehler",
}

BOUNDARY_LABELS = {
    "collision_envelope": "Kollisionshülle",
    "center_envelope": "Mittelpunkthülle",
    "fixed_walls": "Referenzgrenze",
    "funnel_walls": "seitliche Bande",
}


def plot_trajectories(
    run: SimulationRun,
    *,
    max_trajectories: int | None = None,
    max_trajectories_by_status: Mapping[str, int | None] | None = None,
    trajectory_alpha_by_status: Mapping[str, float] | None = None,
    trajectory_linewidth_by_status: Mapping[str, float] | None = None,
    samples_per_segment: int = 30,
    trajectory_statuses: Collection[str] | None = None,
    axes=None,
):
    """Plot selected or all analytic trajectory segments and the geometry."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    if axes is None:
        figure, axes = plt.subplots(figsize=(8, 8))
    else:
        figure = axes.figure

    config = run.config
    acceleration = np.array([0.0, -config.gravity])
    peg_centers = triangular_peg_centers(config)
    envelope = make_side_envelope(config)

    eligible_trajectories = [
        trajectory
        for trajectory in run.trajectories
        if trajectory_statuses is None or trajectory.status in trajectory_statuses
    ]
    if max_trajectories_by_status is None:
        if max_trajectories is None:
            trajectories = eligible_trajectories
        else:
            count = min(max_trajectories, len(eligible_trajectories))
            indices = (
                np.unique(
                    np.linspace(
                        0,
                        len(eligible_trajectories) - 1,
                        count,
                        dtype=int,
                    )
                )
                if count
                else np.empty(0, dtype=int)
            )
            trajectories = [eligible_trajectories[int(index)] for index in indices]
    else:
        trajectories = []
        for status, maximum in max_trajectories_by_status.items():
            status_trajectories = [
                trajectory
                for trajectory in eligible_trajectories
                if trajectory.status == status
            ]
            count = (
                len(status_trajectories)
                if maximum is None
                else min(maximum, len(status_trajectories))
            )
            if count == 0:
                continue
            indices = np.unique(
                np.linspace(0, len(status_trajectories) - 1, count, dtype=int)
            )
            trajectories.extend(status_trajectories[int(index)] for index in indices)

    if trajectories:
        separator = np.full((1, 2), np.nan)
        for status in STATUS_COLORS:
            status_trajectories = [
                trajectory
                for trajectory in trajectories
                if trajectory.status == status
            ]
            pieces: list[np.ndarray] = []
            for trajectory in status_trajectories:
                for segment in trajectory.segments:
                    times = np.linspace(0.0, segment.duration, samples_per_segment)
                    points = (
                        segment.start_position[None, :]
                        + times[:, None] * segment.start_velocity[None, :]
                        + 0.5 * times[:, None] ** 2 * acceleration[None, :]
                    )
                    pieces.extend((points, separator))
            if not pieces:
                continue
            alpha = (
                trajectory_alpha_by_status.get(status, 0.65)
                if trajectory_alpha_by_status is not None
                else 0.65
            )
            linewidth = (
                trajectory_linewidth_by_status.get(status, 0.8)
                if trajectory_linewidth_by_status is not None
                else 0.8
            )
            points = np.concatenate(pieces)
            axes.plot(
                points[:, 0],
                points[:, 1],
                color=STATUS_COLORS[status],
                linewidth=linewidth,
                alpha=alpha,
                zorder=2.5 if status == "detected_inside" else 1.0,
                label=STATUS_LABELS[status],
            )

    detected_x = [
        trajectory.x_detector
        for trajectory in trajectories
        if trajectory.status == "detected_inside" and trajectory.x_detector is not None
    ]
    if detected_x:
        all_hits_are_shown = len(detected_x) == run.summary.status_counts.get(
            "detected_inside", 0
        )
        axes.scatter(
            detected_x,
            np.full(len(detected_x), config.resolved_detector_y),
            marker="|",
            s=30,
            color="tab:green",
            linewidths=0.8,
            alpha=0.3 if len(detected_x) > 500 else 0.8,
            zorder=5.0,
            label=(
                "alle Detektortreffer"
                if all_hits_are_shown
                else "Treffer dargestellter Bahnen"
            ),
        )

    wall_contact_positions = np.asarray(
        [
            collision.position
            for trajectory in trajectories
            for collision in trajectory.wall_collisions
        ],
        dtype=float,
    )
    if wall_contact_positions.size:
        wall_contact_positions = wall_contact_positions.reshape(-1, 2)
        axes.scatter(
            wall_contact_positions[:, 0],
            wall_contact_positions[:, 1],
            marker="o",
            s=9,
            color="tab:purple",
            alpha=0.65,
            linewidths=0.0,
            zorder=5.0,
            label="Bandenkontakt",
        )

    for center in peg_centers:
        axes.add_patch(
            Circle(
                center,
                config.peg_radius,
                facecolor="0.35",
                edgecolor="none",
                alpha=0.8,
                zorder=3.0,
            )
        )
        axes.add_patch(
            Circle(
                center,
                config.effective_radius,
                facecolor="none",
                edgecolor="0.6",
                linewidth=0.45,
                linestyle=":",
                zorder=3.1,
            )
        )

    boundary_top_y = (
        config.release_y if envelope.upper_half_width is not None else config.y_top
    )
    y_values = np.linspace(boundary_top_y, config.resolved_detector_y, 240)
    widths = np.asarray(envelope.half_width(y_values))
    boundary_linestyle = "-" if config.boundary_behavior == "reflecting" else "--"
    boundary_linewidth = 2.0 if config.boundary_behavior == "reflecting" else 1.0
    axes.plot(
        widths,
        y_values,
        color="0.15",
        linestyle=boundary_linestyle,
        linewidth=boundary_linewidth,
        zorder=4.0,
        label=BOUNDARY_LABELS[config.boundary_mode],
    )
    axes.plot(
        -widths,
        y_values,
        color="0.15",
        linestyle=boundary_linestyle,
        linewidth=boundary_linewidth,
        zorder=4.0,
    )
    axes.axhline(
        config.resolved_detector_y,
        color="tab:green",
        linewidth=1.2,
        zorder=4.0,
    )
    axes.plot(
        [-config.start_width / 2.0, config.start_width / 2.0],
        [config.release_y, config.release_y],
        color="tab:cyan",
        linewidth=2.0,
        zorder=4.0,
    )
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_title(
        "Deterministisches Galtonbrett\n"
        f"S={run.summary.survival_fraction:.3f}, "
        f"mittlere Kontakte={run.summary.collision_count_mean_all:.2f}"
    )
    axes.set_aspect("equal", adjustable="box")
    axes.legend(loc="best", fontsize="small")
    return figure, axes


def plot_detector_statistics(
    run: SimulationRun,
    *,
    bins: str | int | np.ndarray = "fd",
    kde_bandwidth: str | float = "scott",
    axes=None,
):
    """Plot the continuous detector coordinates as density histogram and KDE.

    The KDE is a descriptive envelope only.  It is not a Gaussian fit and no
    distributional model enters the simulation dynamics.
    """

    import matplotlib.pyplot as plt

    if axes is None:
        figure, axes = plt.subplots(figsize=(7, 6))
    else:
        figure = axes.figure

    detector_x = np.asarray(run.summary.detector_x, dtype=float)
    if detector_x.size == 0:
        axes.text(
            0.5,
            0.5,
            "Keine Detektortreffer",
            ha="center",
            va="center",
            transform=axes.transAxes,
        )
    else:
        axes.hist(
            detector_x,
            bins=bins,
            density=True,
            color="tab:blue",
            alpha=0.35,
            edgecolor="tab:blue",
            linewidth=0.8,
            label="Histogramm",
        )
        if detector_x.size >= 2 and float(np.ptp(detector_x)) > 0.0:
            from scipy.stats import gaussian_kde

            padding = 0.05 * float(np.ptp(detector_x))
            grid = np.linspace(
                float(np.min(detector_x)) - padding,
                float(np.max(detector_x)) + padding,
                500,
            )
            density = gaussian_kde(detector_x, bw_method=kde_bandwidth)(grid)
            axes.fill_between(
                grid,
                0.0,
                density,
                color="tab:orange",
                alpha=0.22,
                label=f"KDE ({kde_bandwidth})",
            )
            axes.plot(grid, density, color="tab:orange", linewidth=1.8)
        axes.legend(loc="best", fontsize="small")

    axes.set_xlabel(r"kontinuierliche Auftreffposition $x_{det}$")
    axes.set_ylabel("Wahrscheinlichkeitsdichte")
    axes.set_title(
        "Detektorstatistik\n"
        f"{detector_x.size} Treffer aus {run.summary.n_trajectories} Starts"
    )
    axes.grid(axis="y", alpha=0.2)
    return figure, axes


def plot_gaussian_kde_comparison(
    detector_x: np.ndarray,
    *,
    bins: str | int | np.ndarray = "fd",
    kde_bandwidth: str | float = "scott",
    title: str | None = None,
):
    """Compare a fitted Gaussian and KDE for one detector sample."""

    import matplotlib.pyplot as plt

    comparison = compare_gaussian_and_kde(
        detector_x,
        kde_bandwidth=kde_bandwidth,
    )
    values = np.asarray(detector_x, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": (3.0, 1.0)},
        constrained_layout=True,
    )
    density_axes, difference_axes = axes
    density_axes.hist(
        values,
        bins=bins,
        density=True,
        color="tab:blue",
        alpha=0.28,
        edgecolor="tab:blue",
        linewidth=0.75,
        label="Histogramm (nur zur Orientierung)",
    )
    density_axes.plot(
        comparison.grid,
        comparison.kde_density,
        color="tab:orange",
        linewidth=2.2,
        label=(
            f"KDE ({kde_bandwidth}, "
            f"h={comparison.kde_bandwidth:.3g})"
        ),
    )
    density_axes.plot(
        comparison.grid,
        comparison.gaussian_density,
        color="tab:red",
        linestyle="--",
        linewidth=2.2,
        label=(
            "Gauß-Fit "
            f"(μ={comparison.mean:.3g}, σ={comparison.standard_deviation:.3g})"
        ),
    )
    density_axes.set_ylabel("Wahrscheinlichkeitsdichte")
    density_axes.set_title(
        title
        if title is not None
        else f"Gauß- und KDE-Fit für {comparison.n_samples} Detektortreffer"
    )
    density_axes.grid(axis="y", alpha=0.2)
    density_axes.legend(loc="best")
    diagnostics = (
        f"Schiefe = {comparison.skewness:.3f}\n"
        f"Exzess = {comparison.excess_kurtosis:.3f}\n"
        f"KS-D = {comparison.ks_distance:.3f}\n"
        f"W₁/σ = {comparison.normalized_wasserstein_distance:.3f}\n"
        f"TV(KDE, Gauß) ≈ {comparison.total_variation_distance:.3f}"
    )
    density_axes.text(
        0.015,
        0.97,
        diagnostics,
        transform=density_axes.transAxes,
        ha="left",
        va="top",
        fontsize="small",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.88},
    )

    difference = comparison.kde_density - comparison.gaussian_density
    difference_axes.axhline(0.0, color="0.25", linewidth=0.9)
    difference_axes.fill_between(
        comparison.grid,
        0.0,
        difference,
        where=difference >= 0.0,
        color="tab:orange",
        alpha=0.35,
        interpolate=True,
        label="KDE dichter",
    )
    difference_axes.fill_between(
        comparison.grid,
        0.0,
        difference,
        where=difference < 0.0,
        color="tab:red",
        alpha=0.28,
        interpolate=True,
        label="Gauß dichter",
    )
    difference_axes.set_xlabel(r"kontinuierliche Auftreffposition $x_{det}$")
    difference_axes.set_ylabel("KDE − Gauß")
    difference_axes.grid(axis="y", alpha=0.2)
    difference_axes.legend(loc="best", ncols=2, fontsize="small")
    return figure, axes, comparison


def plot_simulation_overview(
    run: SimulationRun,
    *,
    max_trajectories: int | None = None,
    max_escaped_per_side: int | None = None,
    bins: str | int | np.ndarray = "fd",
    kde_bandwidth: str | float = "scott",
):
    """Place trajectory outcomes and detector statistics side by side."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 7.5),
        gridspec_kw={"width_ratios": (1.45, 1.0)},
        constrained_layout=True,
    )
    show_all = max_trajectories is None and max_escaped_per_side is None
    maximum_by_status = None if show_all else {
        "escaped_left": max_escaped_per_side,
        "escaped_right": max_escaped_per_side,
        "detected_inside": max_trajectories,
        "stalled": max_escaped_per_side,
        "max_events": max_escaped_per_side,
        "max_time": max_escaped_per_side,
        "numerical_failure": max_escaped_per_side,
        "invalid_initial_state": max_escaped_per_side,
    }
    trajectory_count = len(run.trajectories)
    detected_alpha = (
        max(0.08, min(0.78, 40.0 / max(trajectory_count, 1)))
        if show_all
        else 0.78
    )
    detected_linewidth = 0.35 if show_all and trajectory_count > 500 else 0.9
    plot_trajectories(
        run,
        max_trajectories=max_trajectories,
        max_trajectories_by_status=maximum_by_status,
        trajectory_alpha_by_status={
            "escaped_left": 0.16,
            "escaped_right": 0.16,
            "detected_inside": detected_alpha,
            "stalled": 0.3,
            "max_events": 0.3,
            "max_time": 0.3,
            "numerical_failure": 0.3,
        },
        trajectory_linewidth_by_status={
            "escaped_left": 0.55,
            "escaped_right": 0.55,
            "detected_inside": detected_linewidth,
            "stalled": 0.55,
            "max_events": 0.55,
            "max_time": 0.55,
            "numerical_failure": 0.55,
        },
        trajectory_statuses=None if show_all else set(maximum_by_status),
        axes=axes[0],
    )
    display_description = (
        f"alle {run.summary.n_trajectories} Trajektorien dargestellt"
        if show_all
        else "Auswahl von Trajektorien dargestellt"
    )
    if run.config.boundary_behavior == "reflecting":
        axes[0].set_title(
            "Trajektorien mit reflektierenden Banden\n"
            f"{display_description}; "
            f"{run.summary.status_counts.get('detected_inside', 0)} von "
            f"{run.summary.n_trajectories} erreichen den Detektor; "
            f"Ø {run.summary.wall_collision_count_mean_all:.2f} Bandenkontakte"
        )
    else:
        axes[0].set_title(
            "Detektierte und ausgetretene Trajektorien\n"
            f"{display_description}; "
            f"{run.summary.status_counts.get('detected_inside', 0)} von "
            f"{run.summary.n_trajectories} erreichen den Detektor"
        )
    plot_detector_statistics(
        run,
        bins=bins,
        kde_bandwidth=kde_bandwidth,
        axes=axes[1],
    )
    parameters = run.config.dimensionless_parameters()
    figure.suptitle(
        "Ereignisgetriebenes Galtonbrett  ·  "
        f"Dₚ/W={parameters['peg_diameter_over_width']:.3g}, "
        f"Dᵦ/W={parameters['ball_diameter_over_width']:.3g}, "
        f"d/W={parameters['delta']:.3g}, "
        f"wₑ={parameters['kinetic_energy_retention']:.3g}, "
        f"S={run.summary.survival_fraction:.3f}, "
        f"tₛᵢₘ={run.simulation_seconds:.2f} s "
        f"({run.workers} Worker)"
    )
    return figure, axes


def plot_start_grid_scan(scan: StartGridScanResult):
    """Plot a bin-free convergence analysis of deterministic start grids."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    width = scan.base_config.start_width
    finest = scan.levels[-1]
    detected = np.isfinite(finest.detector_x)

    axes[0, 0].scatter(
        finest.start_positions[detected] / width,
        finest.detector_x[detected] / width,
        s=5,
        alpha=0.55,
        linewidths=0.0,
        label="detektiert",
    )
    failed = ~detected
    if bool(np.any(failed)):
        axes[0, 0].plot(
            finest.start_positions[failed] / width,
            np.full(np.count_nonzero(failed), 0.02),
            marker="x",
            markersize=4,
            linestyle="none",
            color="tab:red",
            transform=axes[0, 0].get_xaxis_transform(),
            label="nicht detektiert",
        )
    axes[0, 0].set_xlabel(r"Startposition $x_0/W$")
    axes[0, 0].set_ylabel(r"Detektorposition $x_{det}/W$")
    axes[0, 0].set_title(
        rf"Direkte Abbildung für {finest.n_starts} Startpunkte"
    )
    axes[0, 0].legend(loc="best", fontsize="small")
    axes[0, 0].grid(alpha=0.2)

    for row, level in enumerate(scan.levels):
        axes[0, 1].plot(
            level.detected_x / width,
            np.full(level.detected_count, row),
            marker="|",
            markersize=5,
            linestyle="none",
            alpha=0.35,
        )
    axes[0, 1].set_yticks(
        range(len(scan.levels)),
        [str(level.n_starts) for level in scan.levels],
    )
    axes[0, 1].set_xlabel(r"Detektorposition $x_{det}/W$")
    axes[0, 1].set_ylabel("Zahl der Startpunkte")
    axes[0, 1].set_title("Erreichbare Positionen bei Gitterverfeinerung")
    axes[0, 1].grid(axis="x", alpha=0.2)

    grid_sizes = np.asarray([level.n_starts for level in scan.levels])
    axes[1, 0].plot(
        grid_sizes,
        [level.detected_count for level in scan.levels],
        marker="o",
        label="Detektortreffer",
    )
    axes[1, 0].plot(
        grid_sizes,
        [level.exact_endpoint_count for level in scan.levels],
        marker="s",
        linestyle="--",
        label="exakt verschiedene Endpunkte",
    )
    axes[1, 0].plot(
        grid_sizes,
        [level.resolved_endpoint_count for level in scan.levels],
        marker="^",
        label=(
            "getrennt bei "
            rf"$\varepsilon={scan.endpoint_tolerance_ratio:g}\,W$"
        ),
    )
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_xticks(grid_sizes, [str(size) for size in grid_sizes])
    axes[1, 0].set_xlabel("Zahl der Startpunkte")
    axes[1, 0].set_ylabel("Zahl der Endpositionen")
    axes[1, 0].set_title("Wächst die Zahl erreichbarer Endpunkte weiter?")
    axes[1, 0].legend(loc="best", fontsize="small")
    axes[1, 0].grid(alpha=0.2)

    for level in scan.levels:
        ordered = np.sort(level.detected_x / width)
        if ordered.size == 0:
            continue
        cumulative = np.arange(1, ordered.size + 1) / ordered.size
        axes[1, 1].plot(
            ordered,
            cumulative,
            linewidth=1.2,
            label=str(level.n_starts),
        )
    axes[1, 1].set_xlabel(r"Detektorposition $x_{det}/W$")
    axes[1, 1].set_ylabel("Empirische Verteilungsfunktion")
    axes[1, 1].set_title("Bin-freie Konvergenz der Verteilung")
    axes[1, 1].legend(title="Startpunkte", fontsize="small")
    axes[1, 1].grid(alpha=0.2)

    parameters = scan.base_config.dimensionless_parameters()
    figure.suptitle(
        "Deterministischer Startgitter-Scan  ·  "
        f"N={scan.base_config.n_rows}, "
        f"Dₚ/W={parameters['peg_diameter_over_width']:.3g}, "
        f"Dᵦ/W={parameters['ball_diameter_over_width']:.3g}, "
        f"d/W={parameters['delta']:.3g}, "
        f"wₑ={parameters['kinetic_energy_retention']:.3g}, "
        f"t={scan.total_seconds:.2f} s ({scan.workers} Worker)"
    )
    return figure, axes


def plot_scan(scan: ScanResult):
    """Plot the four reference observables of a rho/delta scan."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    panels = (
        (scan.survival_fraction, "Überlebensanteil $S_M$"),
        (scan.collision_count_mean_all, "Mittlere Kontaktzahl"),
        (
            scan.fraction_with_ge_k_collisions,
            "Anteil mit mindestens k Kontakten",
        ),
        (scan.failure_fraction, "Fehler-/Abbruchanteil"),
    )
    for axes_item, (values, title) in zip(axes.flat, panels, strict=True):
        image = axes_item.pcolormesh(
            scan.delta,
            scan.rho,
            values,
            shading="nearest",
        )
        axes_item.set_xlabel(r"$\delta=d/W$")
        axes_item.set_ylabel(r"$\rho=(R_{peg}+R_{ball})/W$")
        axes_item.set_title(title)
        figure.colorbar(image, ax=axes_item)
    slice_description = "h/d konstant" if scan.keep_h_over_d else "h/W konstant"
    figure.suptitle(f"Deterministischer Parameterscan ({slice_description})")
    return figure, axes
