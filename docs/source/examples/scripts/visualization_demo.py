#!/usr/bin/env python
"""
Publication figures with ``habit.viz`` (pure matplotlib, English labels).

Demonstrates:

* habitat-clustering PCA scatter from a two-step ``StudyResult``,
* survival Kaplan-Meier plot from a synthetic table,
* regression diagnostic plots (predicted vs observed).

Figures are returned to the caller; persistence is explicit (``savefig``).

This script accompanies ``docs/source/examples/visualization.rst``.

Run from the repository root::

    python docs/source/examples/scripts/visualization_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import habit.recipes as recipes
from habit import HabitatSpec, Spec, Stage, make_synthetic_cohort, make_synthetic_feature_table
from habit.viz import (
    plot_habitat_clustering_pca_2d,
    plot_kaplan_meier,
    plot_predicted_vs_observed,
    use_style,
)

out_dir = Path(tempfile.mkdtemp(prefix="habit_viz_demo_"))

# --- Habitat clustering PCA (population-level units from two-step train) ---
cohort = make_synthetic_cohort(n_subjects=6, shape=(20, 20, 20), rng=3)
spec = HabitatSpec(
    name="viz_two_step",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": ["T1", "T2"]})),
        Stage("partition", Spec("kmeans", {"n_supervoxels": 8, "n_init": 3})),
        Stage("pool", Spec("pool")),
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 3,
                    "validation": "silhouette",
                    "n_init": 3,
                },
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
    ),
    random_seed=3,
)
result = recipes.Study(spec=spec).fit_predict(cohort)
arrays = result._population_clustering_arrays()
if arrays is None:
    raise RuntimeError("Expected pooled clustering arrays from two-step result.")
features, labels, centers = arrays

with use_style("radiology"):
    fig_pca = plot_habitat_clustering_pca_2d(
        features,
        labels,
        centers=centers,
        title="Habitat clustering (PCA)",
    )
pca_path = out_dir / "habitat_pca_2d.png"
fig_pca.savefig(pca_path, dpi=120, bbox_inches="tight")
print(f"Wrote {pca_path.name} ({pca_path.stat().st_size} bytes)")

# --- Survival KM plot ---
survival_table = make_synthetic_feature_table(n_rows=40, n_features=4, task="survival", rng=9)
frame = survival_table.frame
with use_style("nature"):
    fig_km = plot_kaplan_meier(
        frame["time"].to_numpy(),
        frame["event"].to_numpy().astype(bool),
        group=frame["event"].to_numpy(),
        title="Kaplan-Meier (synthetic)",
    )
km_path = out_dir / "kaplan_meier.png"
fig_km.savefig(km_path, dpi=120, bbox_inches="tight")
print(f"Wrote {km_path.name} ({km_path.stat().st_size} bytes)")

# --- Regression scatter ---
predicted = frame["signal"].to_numpy() + 0.1
observed = frame["signal"].to_numpy()
with use_style("radiology"):
    fig_reg = plot_predicted_vs_observed(observed, predicted)
    fig_reg.suptitle("Predicted vs observed")
reg_path = out_dir / "predicted_vs_observed.png"
fig_reg.savefig(reg_path, dpi=120, bbox_inches="tight")
print(f"Wrote {reg_path.name} ({reg_path.stat().st_size} bytes)")

print(f"\nAll figures under {out_dir}")
print(
    "Binary ML ROC/calibration: habit.viz + recipes.ml_reporting "
    "(train_/test_/cv_ under output/visualizations/); "
    "multi-model compare: ml_advanced_demo.py (compare_models)."
)
