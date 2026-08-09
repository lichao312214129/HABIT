#!/usr/bin/env python
"""
Habitat recipe API: three clustering modes + apply + StudyResult / model I/O.

* **Batch** — ``two_step`` / ``one_step`` / ``direct_pooling`` on a Cohort.
* **Atomic** — ``result.pipeline(subject)`` and ``apply_habitat_model`` on
  one subject or a sliced cohort.
* **Persistence** — ``StudyResult.save`` and ``HabitatModel.save/load``
  (``.habitatmodel``).

Accompanies ``docs/source/examples/habitat_recipes_api.rst``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from habit import HabitatModel, HabitatSpec, Spec, make_synthetic_cohort
import habit.recipes as recipes


def _spec(name: str, *, supervoxelizer: bool, pooling: str | None = None) -> HabitatSpec:
    """Build a tiny deterministic habitat spec for demos.

    Keyword order follows the runtime pipeline, not HabitatSpec field order.
    ``pooling`` declares the dataflow explicitly ("cohort" default / "none").
    """
    return HabitatSpec(
        name=name,
        pooling=pooling,
        voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
        voxel_feature_preprocessors=(
            Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
            Spec("minmax", {"across_features": False}),
        ),
        supervoxelizer=(
            Spec("kmeans", {"n_supervoxels": 8, "n_init": 3})
            if supervoxelizer
            else None
        ),
        cohort_feature_preprocessors=(
            Spec(
                "binning",
                {"n_bins": 8, "bin_strategy": "uniform", "across_features": False},
            ),
        )
        if name != "one_step"
        else (),
        habitat_model_fitter=Spec(
            "kmeans",
            {
                "min_habitats": 2,
                "max_habitats": 3,
                "validation": "silhouette",
                "n_init": 3,
            },
        ),
        habitat_assigner=Spec("nearest_centroid"),
        habitat_features=(
            Spec("volume"),
            Spec("msi"),
            Spec("ith_score"),
            Spec("non_radiomics"),
            # Heavy PyRadiomics families (opt-in; require pyradiomics):
            # Spec("traditional"),
            # Spec("whole_habitat"),
            # Spec("each_habitat"),
        ),
        random_seed=11,
    )


cohort = make_synthetic_cohort(n_subjects=3, shape=(16, 16, 16), rng=11)

print("=== two_step (batch train) ===")
two = recipes.two_step(cohort, _spec("two_step", supervoxelizer=True))
assert two.habitat_model is not None
print(f"  habitats={two.habitat_model.n_habitats}, "
      f"features={len(two.features.feature_columns)}")

print("=== apply_habitat_model + .habitatmodel round-trip ===")
with tempfile.TemporaryDirectory(prefix="habit_api_habitat_") as tmp:
    archive = Path(tmp) / "demo.habitatmodel"
    two.habitat_model.save(archive)
    reloaded: HabitatModel = HabitatModel.load(archive)
    predicted = recipes.apply_habitat_model(cohort, _spec("two_step", supervoxelizer=True), reloaded)
    print(f"  reloaded model_id={reloaded.model_id}")
    print(f"  predict maps={len(predicted.habitat_maps)}")

    saved = two.save(Path(tmp) / "study_out", write_cluster_plots=False)
    print(f"  StudyResult.save -> {saved.name}")

print("=== Atomic: SubjectPipeline on one subject ===")
pipeline = two.pipeline
assert pipeline is not None
one_map = pipeline(cohort[0])
print(f"  pipeline({cohort[0].subject_id}) labels="
      f"{sorted(set(int(v) for v in one_map.label_array.ravel() if v > 0))}")

print("=== one_step (per-subject habitats) ===")
one = recipes.one_step(cohort, _spec("one_step", supervoxelizer=False))
print(f"  subject_models={list(one.subject_models)}")

print("=== direct_pooling ===")
pool = recipes.direct_pooling(cohort, _spec("direct_pooling", supervoxelizer=False))
assert pool.habitat_model is not None
print(f"  habitats={pool.habitat_model.n_habitats}")

print("=== fit_habitat (unified entry, spec-driven dispatch) ===")
# pooling="none" declares the subject-level dataflow on the spec itself;
# fit_habitat reads the declaration and runs the same design as one_step.
unified = recipes.fit_habitat(
    cohort, _spec("one_step", supervoxelizer=False, pooling="none")
)
assert list(unified.subject_models) == list(one.subject_models)
print(f"  fit_habitat(pooling='none') subject_models={list(unified.subject_models)}")

# Eye-check: open habitats on anatomy (napari). Set HABIT_NO_VIEW=1 to skip.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _habitat_eye_check import eye_check_study
eye_check_study(cohort, two)
