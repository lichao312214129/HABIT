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
                    "validation": "silhouette",
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

    # Eye-check predicted habitats on anatomy (napari). Set HABIT_NO_VIEW=1 to skip.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _habitat_eye_check import eye_check_study
    eye_check_study(new_cohort, prediction)
