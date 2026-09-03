#!/usr/bin/env python
"""
Train habitats on demo_data, then extract light habitat-map features.

Accompanies ``docs/source/examples/feature_extraction.rst``.
Run from the repository root::

    python docs/source/examples/scripts/feature_extraction_demo.py
"""

from __future__ import annotations

# BEGIN example
from pathlib import Path

from habit.spec import HabitatSpec, Spec, Stage
from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
import habit.recipes as recipes

# Change DATA / MODALITIES / ROI to your preprocessed layout
DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"

cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
spec = HabitatSpec(
    name="extract_demo",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": list(MODALITIES)})),
        Stage("partition", Spec("kmeans", {"n_supervoxels": 6, "n_init": 3})),
        Stage("pool", Spec("pool")),
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 4,
                    "validation": "elbow",
                    "n_init": 3,
                },
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
        Stage("quantify2", Spec("msi")),
        Stage("quantify3", Spec("ith_score")),
        Stage("quantify4", Spec("non_radiomics")),
    ),
    random_seed=11,
)

train_result = recipes.Study(spec=spec).fit_predict(cohort)
maps_dir = Path("out/habitat_maps")
train_result.save(maps_dir, write_maps=True, write_units_table=True)
print(f"Trained: {train_result.habitat_model.n_habitats} habitats -> {maps_dir}")

# Batch extract from the maps you just wrote (paths are plain strings)
feature_types = ["volume", "msi", "ith_score", "non_radiomics", "graph"]
extract_result = recipes.extract_habitat_features(
    {
        "raw_img_folder": str(DATA),
        "habitats_map_folder": str(maps_dir),
        "out_dir": "out/features",
        "n_processes": 1,
        "habitat_pattern": "*_habitats.nrrd",
        "feature_types": feature_types,
        "n_habitats": train_result.habitat_model.n_habitats,
    }
)
print(f"Extracted {feature_types} -> {extract_result.output_dir}")
# END example

# BEGIN figures
# Paste after the Script block. Uses cohort and train_result.
from habit.kernels import habitat_ith_dispersion, habitat_volume_fractions, ith_score, spatial_interaction_matrix
from habit.viz import (
    plot_cluster_validation_from_report,
    plot_habitat_overlay,
    plot_habitat_volume_fractions,
    plot_ith_summary,
    plot_msi_matrix,
    plot_partition_triptych,
)

Path("out").mkdir(exist_ok=True)
subject = cohort[0]
habitat_map = train_result.habitat_maps[0]
fig = plot_habitat_overlay(subject.image(ROI), habitat_map, title="habitats")
fig.savefig("out/feature_extract_overlay.png", dpi=150, bbox_inches="tight")
if train_result.units:
    fig = plot_partition_triptych(
        subject.image(ROI),
        train_result.units[0],
        habitat_map,
    )
    fig.savefig("out/feature_extract_triptych.png", dpi=150, bbox_inches="tight")
labels = habitat_map.label_array
ids = tuple(int(v) for v in habitat_map.habitat_ids)
if ids:
    fig = plot_habitat_volume_fractions(habitat_volume_fractions(labels, ids))
    fig.savefig("out/feature_extract_volume_fractions.png", dpi=150, bbox_inches="tight")
    n_classes = int(max(ids)) + 1
    msi = spatial_interaction_matrix(labels, n_classes=n_classes)
    fig = plot_msi_matrix(msi, habitat_ids=tuple(range(1, n_classes)))
    fig.savefig("out/feature_extract_msi_matrix.png", dpi=150, bbox_inches="tight")
    fig = plot_ith_summary(ith_score(labels), dispersion=habitat_ith_dispersion(labels))
    fig.savefig("out/feature_extract_ith_summary.png", dpi=150, bbox_inches="tight")
model = train_result.habitat_model
report = None if model is None else (model.preprocessing_state or {}).get(
    "selection_report"
)
if report:
    fig = plot_cluster_validation_from_report(report)
    fig.savefig("out/feature_extract_cluster_validation.png", dpi=150, bbox_inches="tight")
print("Wrote figures under out/")
# END figures

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import save_habitat_study_figures
    from _habitat_eye_check import eye_check_study

    save_habitat_study_figures(cohort, train_result, prefix="feature_extract")
    eye_check_study(cohort, train_result)
