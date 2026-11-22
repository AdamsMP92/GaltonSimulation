"""Compare the saved detector sample with a Gaussian and a KDE fit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from galton import plot_gaussian_kde_comparison


RUN_DIRECTORY = Path("results/latest")
SOURCE_LABEL = "32_0p5"


if __name__ == "__main__":
    metadata = json.loads((RUN_DIRECTORY / "run.json").read_text(encoding="utf-8"))
    with np.load(RUN_DIRECTORY / "run.npz", allow_pickle=False) as saved:
        detector_x = np.asarray(saved["detector_x"], dtype=float)

    config = metadata["config"]
    title = (
        "Gauß-Fit versus KDE des gespeicherten Detektorsamples\n"
        f"n_rows={config['n_rows']}, "
        f"wₑ={config['kinetic_energy_retention']:.3g}, "
        f"n={np.count_nonzero(np.isfinite(detector_x))} Treffer"
    )
    figure, _, comparison = plot_gaussian_kde_comparison(
        detector_x,
        title=title,
    )

    image_path = RUN_DIRECTORY / f"gauss_vs_kde_{SOURCE_LABEL}.png"
    data_path = RUN_DIRECTORY / f"gauss_vs_kde_{SOURCE_LABEL}.json"
    figure.savefig(image_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    data_path.write_text(
        json.dumps(
            {
                "source_run": str(RUN_DIRECTORY / "run.npz"),
                "source_label": SOURCE_LABEL,
                "stored_config": {
                    "n_rows": config["n_rows"],
                    "kinetic_energy_retention": config[
                        "kinetic_energy_retention"
                    ],
                },
                "comparison": comparison.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison.to_dict(), indent=2, sort_keys=True))
    print(f"saved {image_path}")
    print(f"saved {data_path}")
