#!/usr/bin/env python
"""
Apply a saved .habitatmodel to new subjects.

This script accompanies docs/source/examples/apply_saved_model.rst. It
trains a two-step habitat definition on the first subjects under
``demo_data/preprocessed``, round-trips the model through a
``.habitatmodel`` archive, and projects the reloaded definition onto later
subjects in the same tree -- the publish-and-reuse workflow the v1
HabitatModel contract is designed for.

Primary API: HabitatSpec.stages + recipes.Study(...).fit_predict.

Run from the repository root:

    python docs/source/examples/scripts/apply_saved_model_demo.py
"""

from __future__ import annotations

# BEGIN example
# The spec must match between training and application: the model stores the
# cohort-level preprocessing state, but the upstream stages (voxel features,
# partition) are re-declared here and must be the same.
from pathlib import Path

import numpy as np

from habit.contracts import HabitatModel, cohort_from_directory
from habit.spec import HabitatSpec, Spec, Stage
from habit.datasets import fetch_demo
import habit.recipes as recipes

# Change DATA / MODALITIES / ROI to your preprocessed layout
DATA = fetch_demo()
MODALITIES = ("LAP",)            # series keys under each subject
ROI = "LAP"                      # mask key (often same as a modality)

# Demo pack has subj001..subj005. Train on the first three; apply on the last two.
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
train_cohort = cohort[:3]
new_cohort = cohort[3:5]
print(f"Train: {list(train_cohort.subject_ids)}; apply: {list(new_cohort.subject_ids)}")

spec = HabitatSpec(
    name="habitat_two_step",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": list(MODALITIES)})),
        Stage("partition", Spec("kmeans", {"n_supervoxels": 32, "n_init": 5})),
        Stage("pool", Spec("pool")),
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "n_habitats": 3,
                    "n_init": 5,
                },
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
        Stage("quantify2", Spec("msi")),
        Stage("quantify3", Spec("ith_score")),
        Stage("quantify4", Spec("non_radiomics")),
        # Heavy PyRadiomics families (opt-in; require pyradiomics):
        # Stage("quantify5", Spec("traditional")),
        # Stage("quantify6", Spec("whole_habitat")),
        # Stage("quantify7", Spec("each_habitat")),
    ),
    random_seed=42,
)

# 1. Train the habitat definition on the discovery cohort.
train_result = recipes.Study(spec=spec).fit_predict(train_cohort)
print(f"Trained on {len(train_cohort)} subjects: "
      f"{train_result.habitat_model.n_habitats} habitats")

# 2. Persist the model -- a single self-describing archive.
Path("out").mkdir(exist_ok=True)
archive = Path("out") / "habitat_model.habitatmodel"
train_result.habitat_model.save(archive)
print(f"Saved {archive}")

# 3. Later / elsewhere: reload the published definition. No fitting
#    happens from here on.
model = HabitatModel.load(archive)
print(f"Reloaded model {model.model_id} "
      f"({model.n_habitats} habitats, features {list(model.feature_names)})")

# 4. Project it onto NEW subjects. The model's stored preprocessing state
#    is replayed, so the new units are scored in the training feature space.
print(f"\nNew cohort (batch apply): {list(new_cohort.subject_ids)}")
prediction = recipes.Study.from_model(model, spec).predict(new_cohort)

# Non-batch: apply to one subject via the returned SubjectPipeline.
atomic_subject = new_cohort[0]
atomic_map = prediction.pipeline(atomic_subject)
print(f"Atomic apply pipeline({atomic_subject.subject_id!r}): "
      f"{len(set(atomic_map.label_array[atomic_map.label_array > 0]))} habitat labels")

for habitat_map in prediction.habitat_maps:
    ids, counts = np.unique(
        habitat_map.label_array[habitat_map.label_array > 0],
        return_counts=True,
    )
    print(f"  {habitat_map.subject_id}: voxels per habitat "
          f"{dict(zip(ids.tolist(), counts.tolist()))}")

print("\nPer-subject habitat features:")
print(prediction.features.frame.to_string(index=False))
# END example

# BEGIN figures
# Paste after the Script block. Uses new_cohort, prediction, train_cohort,
# train_result, model, spec, and ROI.
from habit.kernels import habitat_ith_dispersion, habitat_volume_fractions, ith_score, spatial_interaction_matrix
from habit.viz import (
    plot_habitat_label_compare,
    plot_habitat_overlay,
    plot_habitat_volume_fractions,
    plot_ith_summary,
    plot_msi_matrix,
    plot_partition_triptych,
)

Path("out").mkdir(exist_ok=True)
subject = new_cohort[0]
habitat_map = prediction.habitat_maps[0]
# Overlay uses ImageVolume + HabitatMap (3-panel orthogonal default).
fig = plot_habitat_overlay(subject.image(ROI), habitat_map, title="habitats")
fig.savefig("out/apply_overlay.png", dpi=150, bbox_inches="tight")
if prediction.units:
    fig = plot_partition_triptych(
        subject.image(ROI),
        prediction.units[0],
        habitat_map,
    )
    fig.savefig("out/apply_triptych.png", dpi=150, bbox_inches="tight")
labels = habitat_map.label_array
ids = tuple(int(v) for v in habitat_map.habitat_ids)
if ids:
    fig = plot_habitat_volume_fractions(habitat_volume_fractions(labels, ids))
    fig.savefig("out/apply_volume_fractions.png", dpi=150, bbox_inches="tight")
    n_classes = int(max(ids)) + 1
    msi = spatial_interaction_matrix(labels, n_classes=n_classes)
    fig = plot_msi_matrix(msi, habitat_ids=tuple(range(1, n_classes)))
    fig.savefig("out/apply_msi_matrix.png", dpi=150, bbox_inches="tight")
    fig = plot_ith_summary(ith_score(labels), dispersion=habitat_ith_dispersion(labels))
    fig.savefig("out/apply_ith_summary.png", dpi=150, bbox_inches="tight")
replay = recipes.Study.from_model(model, spec).predict(train_cohort[:1])
fig = plot_habitat_label_compare(
    train_cohort[0].image(ROI),
    train_result.habitat_maps[0],
    replay.habitat_maps[0],
    titles=("Fit", "Replay"),
)
fig.savefig("out/apply_train_label_compare.png", dpi=150, bbox_inches="tight")
print("Wrote figures under out/")
# END figures

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import copy_out_figures_to_gallery
    from _habitat_eye_check import eye_check_study

    # Gallery = copy of out/ from the visible block (same composition).
    copy_out_figures_to_gallery(
        (
            "apply_overlay.png",
            "apply_triptych.png",
            "apply_volume_fractions.png",
            "apply_msi_matrix.png",
            "apply_ith_summary.png",
            "apply_train_label_compare.png",
        )
    )
    # Eye-check predicted habitats on anatomy (napari). Set HABIT_NO_VIEW=1 to skip.
    eye_check_study(new_cohort, prediction)
