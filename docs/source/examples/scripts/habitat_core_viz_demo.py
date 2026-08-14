#!/usr/bin/env python
"""
Habitat-core figures on one demo_data subject (two-step + viz).

Accompanies ``docs/source/examples/visualization.rst``.
Run from the repository root::

    python docs/source/examples/scripts/habitat_core_viz_demo.py
"""

from __future__ import annotations

# BEGIN example
from pathlib import Path

import matplotlib.pyplot as plt

from habit import (
    cohort_from_directory,
    habitat_ith_dispersion,
    habitat_volume_fractions,
    ith_score,
    spatial_interaction_matrix,
    two_step_habitat,
)
from habit.viz import (
    plot_habitat_label_compare,
    plot_habitat_overlay,
    plot_habitat_volume_fractions,
    plot_ith_summary,
    plot_msi_matrix,
    plot_partition_triptych,
)

# Change DATA / MODALITIES / ROI to your preprocessed layout
DATA = "demo_data/preprocessed"
MODALITIES = ("LAP",)
ROI = "LAP"

cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
result = two_step_habitat(
    modalities=MODALITIES, n_supervoxels=12, n_habitats=3, random_seed=0, roi=ROI
).fit_predict(cohort)

subject = cohort[0]
image = subject.image(MODALITIES[0]).data
sv = result.units[0].label_array
hab = result.habitat_maps[0].label_array

Path("out").mkdir(exist_ok=True)


def _save(fig: object, name: str) -> None:
    """Save a figure under out/ and close it."""
    fig.savefig(f"out/{name}", dpi=150, bbox_inches="tight")
    plt.close(fig)


_save(plot_habitat_overlay(image, hab, axis=0), "habitat_overlay.png")
_save(plot_partition_triptych(image, sv, hab, axis=0), "habitat_triptych.png")

ids = tuple(sorted({int(v) for v in hab.ravel() if int(v) != 0}))
frac = habitat_volume_fractions(hab, ids)
msi = spatial_interaction_matrix(hab, n_classes=max(ids) + 1)
_save(plot_habitat_volume_fractions(frac), "habitat_volume_fractions.png")
_save(plot_msi_matrix(msi, habitat_ids=ids), "habitat_msi_matrix.png")
_save(
    plot_ith_summary(ith_score(hab), dispersion=habitat_ith_dispersion(hab)),
    "habitat_ith_summary.png",
)

# Optional: compare two label maps on the same grid
hab_b = hab.copy()
if len(ids) >= 2:
    hab_b[hab == ids[-1]] = ids[0]
_save(
    plot_habitat_label_compare(image, hab, hab_b, titles=("Fit", "Perturbed")),
    "habitat_label_compare.png",
)
# END example

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from habit.viz import plot_cluster_validation_from_report as _plot_report

    gallery = Path("docs/source/_static/images/examples")
    gallery.mkdir(parents=True, exist_ok=True)
    mapping = {
        "out/habitat_overlay.png": "habitat_core_overlay.png",
        "out/habitat_triptych.png": "habitat_core_triptych.png",
        "out/habitat_volume_fractions.png": "habitat_core_volume_fractions.png",
        "out/habitat_msi_matrix.png": "habitat_core_msi_matrix.png",
        "out/habitat_ith_summary.png": "habitat_core_ith_summary.png",
        "out/habitat_label_compare.png": "habitat_core_label_compare.png",
    }
    for src, name in mapping.items():
        Path(gallery / name).write_bytes(Path(src).read_bytes())

    report = None
    if result.habitat_model is not None:
        report = (result.habitat_model.preprocessing_state or {}).get(
            "selection_report"
        )
    if report is None:
        report = {
            "candidates": [2, 3, 4],
            "methods": ["elbow"],
            "scores": {"elbow": [0.2, 0.55, 0.4]},
            "selected": 3,
        }
    fig = _plot_report(report)
    fig.savefig(gallery / "habitat_core_cluster_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote out/ and gallery PNGs")
