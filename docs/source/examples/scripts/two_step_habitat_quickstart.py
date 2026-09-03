#!/usr/bin/env python
"""
Two-step habitat analysis, end to end, on demo_data (one subject is enough).

Accompanies ``docs/source/examples/two_step_habitat.rst``.
Run from the repository root::

    python docs/source/examples/scripts/two_step_habitat_quickstart.py
"""

from __future__ import annotations

# BEGIN example
from habit.spec import HabitatSpec, Spec, Stage
from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
import habit.recipes as recipes

# fetch_demo() downloads the official pack once and prints the folder tree.
# Your data: same images/<id>/<mod>/ + masks/<id>/<roi>/ layout; change DATA.
DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"

cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
print(f"Cohort: {len(cohort)} subjects -> {list(cohort.subject_ids)}")

# What can go in Spec("...")? See docs/source/how_to/habitat_components.rst
# or: list_plugins("voxel_feature_extractor") / HabitatModelFitterRegistry.constructor_signature("kmeans")
spec = HabitatSpec(
    name="habitat_two_step",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": list(MODALITIES)})),
        Stage("partition", Spec("kmeans", {"n_supervoxels": 8, "n_init": 5})),
        Stage("pool", Spec("pool")),
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 10,
                    "validation": "elbow",
                    "n_init": 5,
                },
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
        Stage("quantify2", Spec("msi")),
        Stage("quantify3", Spec("ith_score")),
        Stage("quantify4", Spec("non_radiomics")),
    ),
    random_seed=42,
)
print(f"Spec fingerprint: {spec.fingerprint()}")

result = recipes.Study(spec=spec).fit_predict(cohort)
print(result.habitat_model.summary())
print(f"Habitat maps: {len(result.habitat_maps)}")
print(f"Feature columns (first 6): {list(result.features.feature_columns)[:6]}")
print(result.manifest.describe_methods())

# Persist wherever you like — swap this path for your project
out_dir = result.save("out/two_step_demo")
print(f"Saved study to {out_dir}")
# END example

# BEGIN figures
# Paste after the Script block. Uses cohort, result, and ROI from above.
from pathlib import Path

import matplotlib.pyplot as plt

from habit.kernels import habitat_ith_dispersion, habitat_volume_fractions, ith_score, spatial_interaction_matrix
from habit.viz import (
    plot_cluster_validation_from_report,
    plot_habitat_overlay,
    plot_habitat_volume_fractions,
    plot_ith_summary,
    plot_msi_matrix,
    plot_partition_triptych,
)

# Figures: public habit.viz defaults. Edit the out/ filenames if you like.
# Overlay uses the 3-D default (three orthogonal panels). Pass ImageVolume
# and HabitatMap — not .data — so direction / spacing stay attached.
# Triptych is a single axial slice; axis=0 is the public default (shown).
Path("out").mkdir(exist_ok=True)


def _save(fig: object, name: str) -> None:
    """Write one PNG under out/ and close the figure."""
    fig.savefig(f"out/{name}", dpi=150, bbox_inches="tight")
    plt.close(fig)


subject = cohort[0]
habitat_map = result.habitat_maps[0]
_save(
    plot_habitat_overlay(subject.image(ROI), habitat_map, title="habitats"),
    "two_step_overlay.png",
)
_save(
    plot_partition_triptych(
        subject.image(ROI),
        result.units[0],
        habitat_map,
        axis=0,
    ),
    "two_step_triptych.png",
)

labels = habitat_map.label_array
ids = tuple(int(v) for v in habitat_map.habitat_ids)
if ids:
    _save(
        plot_habitat_volume_fractions(habitat_volume_fractions(labels, ids)),
        "two_step_volume_fractions.png",
    )
    n_classes = int(max(ids)) + 1
    msi = spatial_interaction_matrix(labels, n_classes=n_classes)
    _save(
        plot_msi_matrix(msi, habitat_ids=tuple(range(1, n_classes))),
        "two_step_msi_matrix.png",
    )
    _save(
        plot_ith_summary(ith_score(labels), dispersion=habitat_ith_dispersion(labels)),
        "two_step_ith_summary.png",
    )

report = None
if result.habitat_model is not None:
    report = (result.habitat_model.preprocessing_state or {}).get("selection_report")
if report:
    _save(
        plot_cluster_validation_from_report(report),
        "two_step_cluster_validation.png",
    )
print("Wrote figures under out/")
# END figures

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import copy_out_figures_to_gallery
    from _habitat_eye_check import eye_check_study

    # Gallery = copy of out/ from the visible block (same composition).
    copy_out_figures_to_gallery(
        (
            "two_step_overlay.png",
            "two_step_triptych.png",
            "two_step_volume_fractions.png",
            "two_step_msi_matrix.png",
            "two_step_ith_summary.png",
            "two_step_cluster_validation.png",
        )
    )
    eye_check_study(cohort, result)
