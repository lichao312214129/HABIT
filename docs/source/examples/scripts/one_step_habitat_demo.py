#!/usr/bin/env python
"""
One-step habitat analysis on demo_data.

Accompanies ``docs/source/examples/one_step_habitat.rst``.
Run from the repository root::

    python docs/source/examples/scripts/one_step_habitat_demo.py
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
    name="habitat_one_step",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": list(MODALITIES)})),
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
print(f"Cohort-level habitat_model: {result.habitat_model}")
print(f"Per-subject models: {len(result.subject_models)}")
for subject_id, model in sorted(result.subject_models.items()):
    print(f"  {subject_id}: {model.n_habitats} habitats")
print(f"Habitat maps: {len(result.habitat_maps)}")
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
    "one_step_overlay.png",
)

labels = habitat_map.label_array
ids = tuple(int(v) for v in habitat_map.habitat_ids)
if ids:
    _save(
        plot_habitat_volume_fractions(habitat_volume_fractions(labels, ids)),
        "one_step_volume_fractions.png",
    )
    n_classes = int(max(ids)) + 1
    msi = spatial_interaction_matrix(labels, n_classes=n_classes)
    _save(
        plot_msi_matrix(msi, habitat_ids=tuple(range(1, n_classes))),
        "one_step_msi_matrix.png",
    )
    _save(
        plot_ith_summary(ith_score(labels), dispersion=habitat_ith_dispersion(labels)),
        "one_step_ith_summary.png",
    )

# One-step has no cohort HabitatModel; use this subject's selection_report.
model = result.subject_models.get(subject.subject_id)
report = None if model is None else (model.preprocessing_state or {}).get(
    "selection_report"
)
if report:
    _save(
        plot_cluster_validation_from_report(report),
        "one_step_cluster_validation.png",
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
            "one_step_overlay.png",
            "one_step_volume_fractions.png",
            "one_step_msi_matrix.png",
            "one_step_ith_summary.png",
            "one_step_cluster_validation.png",
        )
    )
    eye_check_study(cohort, result)
