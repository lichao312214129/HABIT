#!/usr/bin/env python
"""
Direct-pooling habitat analysis on demo_data.

Accompanies ``docs/source/examples/direct_pooling_habitat.rst``.
Run from the repository root::

    python docs/source/examples/scripts/direct_pooling_habitat_demo.py
"""

from __future__ import annotations

# BEGIN example
from habit import HabitatSpec, Spec, Stage, cohort_from_directory
import habit.recipes as recipes

# Change DATA / MODALITIES / ROI to your preprocessed layout
DATA = "demo_data/preprocessed"
MODALITIES = ("LAP",)
ROI = "LAP"

cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
print(f"Cohort: {len(cohort)} subjects")

# What can go in Spec("...")? See docs/source/how_to/habitat_components.rst
# or: list_plugins("voxel_feature_extractor") / get_param_schema("kmeans", "habitat_model_fitter")
spec = HabitatSpec(
    name="habitat_direct_pooling",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": list(MODALITIES)})),
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

result = recipes.Study(spec=spec).fit_predict(cohort)
print(result.habitat_model.summary())
print(f"Habitat maps: {len(result.habitat_maps)}")
out_dir = result.save("out/direct_pooling_demo")
print(f"Saved study to {out_dir}")
# END example

# BEGIN figures
# Paste after the Script block. Uses cohort, result, and ROI from above.
from pathlib import Path

import matplotlib.pyplot as plt

from habit import (
    habitat_ith_dispersion,
    habitat_volume_fractions,
    ith_score,
    spatial_interaction_matrix,
)
from habit.viz import (
    plot_cluster_validation_from_report,
    plot_habitat_overlay,
    plot_habitat_volume_fractions,
    plot_ith_summary,
    plot_msi_matrix,
)

# Figures: public habit.viz defaults. Edit the out/ filenames if you like.
# Overlay uses the 3-D default (three orthogonal panels). Pass ImageVolume
# and HabitatMap — not .data — so direction / spacing stay attached.
Path("out").mkdir(exist_ok=True)


def _save(fig: object, name: str) -> None:
    """Write one PNG under out/ and close the figure."""
    fig.savefig(f"out/{name}", dpi=150, bbox_inches="tight")
    plt.close(fig)


subject = cohort[0]
habitat_map = result.habitat_maps[0]
_save(
    plot_habitat_overlay(subject.image(ROI), habitat_map, title="habitats"),
    "direct_pooling_overlay.png",
)

labels = habitat_map.label_array
ids = tuple(int(v) for v in habitat_map.habitat_ids)
if ids:
    _save(
        plot_habitat_volume_fractions(habitat_volume_fractions(labels, ids)),
        "direct_pooling_volume_fractions.png",
    )
    n_classes = int(max(ids)) + 1
    msi = spatial_interaction_matrix(labels, n_classes=n_classes)
    _save(
        plot_msi_matrix(msi, habitat_ids=tuple(range(1, n_classes))),
        "direct_pooling_msi_matrix.png",
    )
    _save(
        plot_ith_summary(ith_score(labels), dispersion=habitat_ith_dispersion(labels)),
        "direct_pooling_ith_summary.png",
    )

report = None
if result.habitat_model is not None:
    report = (result.habitat_model.preprocessing_state or {}).get("selection_report")
if report:
    _save(
        plot_cluster_validation_from_report(report),
        "direct_pooling_cluster_validation.png",
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
            "direct_pooling_overlay.png",
            "direct_pooling_volume_fractions.png",
            "direct_pooling_msi_matrix.png",
            "direct_pooling_ith_summary.png",
            "direct_pooling_cluster_validation.png",
        )
    )
    eye_check_study(cohort, result)
