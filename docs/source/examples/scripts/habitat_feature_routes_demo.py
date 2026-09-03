#!/usr/bin/env python
"""
Raw and concat voxel features on the official demo pack.

Accompanies ``docs/source/examples/habitat_feature_routes.rst``.

Run from the repository root::

    python docs/source/examples/scripts/habitat_feature_routes_demo.py
"""

from __future__ import annotations

# BEGIN example
from habit.spec import HabitatSpec, Spec
from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.pipeline.assembly import build_habitat_components
import habit.recipes as recipes

DATA = fetch_demo()
MODALITIES = ("LAP", "PVP")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
subject = cohort[0]
m0, m1 = MODALITIES
print(f"Cohort: {len(cohort)} subjects -> {list(cohort.subject_ids)}")

raw_spec = HabitatSpec(
    name="route_raw",
    voxel_feature_extractor=Spec("raw", {"modalities": list(MODALITIES)}),
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 8, "n_init": 3}),
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "elbow", "n_init": 3},
    ),
    habitat_assigner=Spec("nearest_centroid"),
    habitat_features=(Spec("volume"), Spec("msi"), Spec("ith_score")),
    random_seed=11,
)
raw_units = build_habitat_components(raw_spec).pipeline(assigner=None).units(subject)
print("=== raw(modalities) ===")
print(f"  atomic n_features: {raw_units.feature_frame().shape[1]}")
raw_result = recipes.Study(spec=raw_spec).fit_predict(cohort)
print(
    f"  batch: {len(raw_result.habitat_maps)} maps, "
    f"{raw_result.habitat_model.n_habitats} habitats"
)

concat_spec = HabitatSpec(
    name="route_concat",
    voxel_feature_extractor=Spec(
        "concat",
        {
            "extractors": [
                {"name": "raw", "params": {"modalities": [m0]}},
                {"name": "raw", "params": {"modalities": [m1]}},
            ],
        },
    ),
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 8, "n_init": 3}),
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "elbow", "n_init": 3},
    ),
    habitat_assigner=Spec("nearest_centroid"),
    habitat_features=(Spec("volume"), Spec("msi"), Spec("ith_score")),
    random_seed=11,
)
concat_units = (
    build_habitat_components(concat_spec).pipeline(assigner=None).units(subject)
)
print("=== concat(raw, raw) per modality ===")
print(f"  atomic n_features: {concat_units.feature_frame().shape[1]}")
concat_result = recipes.Study(spec=concat_spec).fit_predict(cohort)
print(f"  batch: {len(concat_result.habitat_maps)} maps")
# END example

# BEGIN figures
# Paste after the Script block. Uses subject, m0, and raw_result.
from pathlib import Path

from habit.viz import plot_habitat_overlay

Path("out").mkdir(exist_ok=True)
fig = plot_habitat_overlay(
    subject.image(m0),
    raw_result.habitat_maps[0],
    title="habitats",
)
fig.savefig("out/habitat_feature_routes_overlay.png", dpi=150, bbox_inches="tight")
print("Wrote out/habitat_feature_routes_overlay.png")
# END figures

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import copy_out_figures_to_gallery

    copy_out_figures_to_gallery(("habitat_feature_routes_overlay.png",))
