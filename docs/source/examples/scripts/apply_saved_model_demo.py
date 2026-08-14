#!/usr/bin/env python
"""
Apply a saved .habitatmodel to new subjects.

This script accompanies docs/source/examples/apply_saved_model.rst. It
trains a habitat definition on one synthetic cohort, round-trips it through
a .habitatmodel archive, and projects the reloaded definition onto a SECOND
cohort of previously unseen subjects -- the publish-and-reuse workflow the
v1 HabitatModel contract is designed for.

Primary API: HabitatSpec.stages + recipes.Study(...).fit_predict.

Run from the repository root:

    python docs/source/examples/scripts/apply_saved_model_demo.py
"""

import tempfile
from pathlib import Path

import numpy as np

from habit import HabitatModel, HabitatSpec, Spec, Stage, make_synthetic_cohort
import habit.recipes as recipes

# BEGIN example
# The spec must match between training and application: the model stores the
# cohort-level preprocessing state, but the upstream stages (voxel features,
# partition) are re-declared here and must be the same.
spec = HabitatSpec(
    name="habitat_two_step",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": ["T1", "T2"]})),
        Stage("partition", Spec("kmeans", {"n_supervoxels": 6, "n_init": 5})),
        Stage("pool", Spec("pool")),
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 3,
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
        # Heavy PyRadiomics families (opt-in; require pyradiomics):
        # Stage("quantify5", Spec("traditional")),
        # Stage("quantify6", Spec("whole_habitat")),
        # Stage("quantify7", Spec("each_habitat")),
    ),
    random_seed=42,
)

# 1. Train the habitat definition on the discovery cohort.
train_cohort = make_synthetic_cohort(n_subjects=5, shape=(20, 20, 20), rng=42)
train_result = recipes.Study(spec=spec).fit_predict(train_cohort)
print(f"Trained on {len(train_cohort)} subjects: "
      f"{train_result.habitat_model.n_habitats} habitats")

with tempfile.TemporaryDirectory() as tmp:
    # 2. Persist the model -- a single self-describing archive.
    archive = Path(tmp) / "habitat_model.habitatmodel"
    train_result.habitat_model.save(archive)
    print(f"Saved {archive.name} ({archive.stat().st_size} bytes)")

    # 3. Later / elsewhere: reload the published definition. No fitting
    #    happens from here on.
    model = HabitatModel.load(archive)
    print(f"Reloaded model {model.model_id} "
          f"({model.n_habitats} habitats, features {list(model.feature_names)})")

    # 4. Project it onto NEW subjects (different ids, drawn with a different
    #    seed). The model's stored preprocessing state is replayed, so the
    #    new units are scored in the training feature space.
    new_cohort = make_synthetic_cohort(n_subjects=3, shape=(20, 20, 20), rng=1234)
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
# train_result, model, and spec.
from habit import (
    habitat_region_stats,
    habitat_volume_fractions,
    ith_score,
    spatial_interaction_matrix,
)
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
fig = plot_habitat_overlay(subject.image("T1"), habitat_map, title="habitats")
fig.savefig("out/apply_overlay.png", dpi=150, bbox_inches="tight")
if prediction.units:
    fig = plot_partition_triptych(
        subject.image("T1"),
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
    fig = plot_ith_summary(ith_score(labels), per_habitat=habitat_region_stats(labels))
    fig.savefig("out/apply_ith_summary.png", dpi=150, bbox_inches="tight")
replay = recipes.Study.from_model(model, spec).predict(train_cohort[:1])
fig = plot_habitat_label_compare(
    train_cohort[0].image("T1"),
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
    from _example_roi import save_habitat_study_figures
    from _habitat_eye_check import eye_check_study

    save_habitat_study_figures(new_cohort, prediction, prefix="apply")
    save_habitat_study_figures(
        train_cohort,
        train_result,
        prefix="apply_train",
        compare_labels=replay.habitat_maps[0].label_array,
        compare_titles=("Fit", "Replay"),
    )
    # Eye-check predicted habitats on anatomy (napari). Set HABIT_NO_VIEW=1 to skip.
    eye_check_study(new_cohort, prediction)
