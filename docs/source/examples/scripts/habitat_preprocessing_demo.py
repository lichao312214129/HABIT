#!/usr/bin/env python
"""
Habitat feature preprocessing before clustering: subject-level + cohort-level.

v1 ``HabitatSpec`` chains (v0.1 names in parentheses):

* ``voxel_feature_preprocessors`` — per subject, before clustering units
  (``preprocessing_for_subject_level``).
* ``supervoxel_feature_preprocessors`` — per subject, after supervoxels
  (two-step only).
* ``cohort_feature_preprocessors`` — fitted once on pooled training rows
  (``preprocessing_for_group_level``).

Design rules exercised here:

* **One-step** — voxel chain only; no cohort chain at train (each subject
  clusters independently; state lives in ``subject_models``).
* **Two-step / direct-pooling** — cohort chain is fitted on pooled units,
  frozen into ``HabitatModel.preprocessing_state``, and replayed at apply.
* **Batch** — ``recipes.two_step(cohort, spec)`` (or ``one_step`` /
  ``direct_pooling``).
* **Non-batch (atomic)** — ``SubjectPipeline.units(subject)`` before fit,
  ``SubjectPipeline(subject)`` after a model is attached (train or apply).

This script accompanies ``docs/source/examples/habitat_preprocessing.rst``.

Run from the repository root::

    python docs/source/examples/scripts/habitat_preprocessing_demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from habit import HabitatModel, HabitatSpec, Spec, cohort_from_directory, make_synthetic_cohort
from habit.domain.assembly import build_habitat_components
import habit.recipes as recipes

REPO_ROOT: Path = Path(__file__).resolve().parents[4]
IMAGING_ROOT: Path = REPO_ROOT / "demo_data" / "preprocessed"

# --- v0.1 -> v1 chain mapping (for readers migrating YAML) -----------------
V0_V1_CHAIN_MAP: Tuple[Tuple[str, str], ...] = (
    ("preprocessing_for_subject_level", "voxel_feature_preprocessors"),
    ("(two-step only, subject)", "supervoxel_feature_preprocessors"),
    ("preprocessing_for_group_level", "cohort_feature_preprocessors"),
)
print("=== v0.1 -> v1 preprocessing chain names ===")
for v0_name, v1_name in V0_V1_CHAIN_MAP:
    print(f"  {v0_name:40s} -> {v1_name}")

# --- Cohort: real demo when available, else synthetic ------------------------
if IMAGING_ROOT.is_dir():
    cohort = cohort_from_directory(
        IMAGING_ROOT,
        modalities=("pre_contrast", "LAP", "PVP", "delay_3min"),
        roi="LAP",
        name="demo_dce",
    )
    modalities = ("pre_contrast", "LAP", "PVP", "delay_3min")
    print(f"\nCohort (batch): {len(cohort)} subjects from demo_data")
else:
    cohort = make_synthetic_cohort(n_subjects=4, shape=(16, 16, 16), rng=7)
    modalities = ("T1", "T2")
    print(f"\nCohort (batch): {len(cohort)} synthetic subjects")

subject = cohort[0]
single = cohort[0:1]
print(f"Atomic subject: {subject.subject_id}")

# --- Shared preprocessing chains (mirrors demo_data/results/api coverage) ----
voxel_chain = (
    Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
    Spec("minmax", {"across_features": False}),
)
supervoxel_chain = (Spec("zscore", {"across_features": False}),)
cohort_chain = (
    Spec("binning", {"n_bins": 6, "bin_strategy": "uniform", "across_features": False}),
)

# Keyword arguments follow runtime order (not HabitatSpec field definition order).
two_step_spec = HabitatSpec(
    name="two_step_with_chains",
    voxel_feature_extractor=Spec("raw", {"modalities": list(modalities)}),
    voxel_feature_preprocessors=voxel_chain,
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 6, "n_init": 3}),
    supervoxel_feature_preprocessors=supervoxel_chain,
    cohort_feature_preprocessors=cohort_chain,
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "silhouette", "n_init": 3},
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
    random_seed=7,
)

direct_pooling_spec = HabitatSpec(
    name="direct_pooling_with_chains",
    voxel_feature_extractor=Spec("raw", {"modalities": list(modalities)}),
    voxel_feature_preprocessors=voxel_chain,
    supervoxelizer=None,
    cohort_feature_preprocessors=cohort_chain,
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "silhouette", "n_init": 3},
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
    random_seed=7,
)

one_step_spec = HabitatSpec(
    name="one_step_subject_only",
    voxel_feature_extractor=Spec("raw", {"modalities": list(modalities)}),
    voxel_feature_preprocessors=voxel_chain,
    supervoxelizer=None,
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "silhouette", "n_init": 3},
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
    random_seed=7,
)


def _units_summary(units: Any) -> Dict[str, Any]:
    """Summarise clustering units for console output."""
    frame = units.feature_frame()
    return {
        "n_units": int(frame.shape[0]),
        "n_features": int(frame.shape[1]),
        "feature_min": float(np.nanmin(frame.to_numpy())),
        "feature_max": float(np.nanmax(frame.to_numpy())),
    }


# --- Non-batch: inspect preprocessing stages on ONE subject --------------------
print("\n=== Non-batch: SubjectPipeline.units (fit-time, no assigner) ===")
fit_components = build_habitat_components(two_step_spec)
fit_pipeline = fit_components.pipeline(assigner=None)
units_one = fit_pipeline.units(subject)
summary = _units_summary(units_one)
print(f"  {subject.subject_id}: {summary['n_units']} units, "
      f"{summary['n_features']} features, "
      f"range [{summary['feature_min']:.3f}, {summary['feature_max']:.3f}]")

# --- Batch: two-step train (all three chains) --------------------------------
print("\n=== Batch: two_step (voxel + supervoxel + cohort chains) ===")
two_step_result = recipes.two_step(cohort, two_step_spec)
model = two_step_result.habitat_model
assert model is not None
print(model.summary())
print("Preprocessing state keys:", sorted(model.preprocessing_state.keys()))
assert "cohort_feature_preprocessor" in model.preprocessing_state

# Non-batch predict on the same subject using the fitted pipeline
predict_pipeline = two_step_result.pipeline
assert predict_pipeline is not None
habitat_map = predict_pipeline(subject)
unique_labels = np.unique(habitat_map.label_array[habitat_map.label_array > 0])
print(f"Atomic predict_pipeline({subject.subject_id!r}): "
      f"{len(unique_labels)} habitat labels in ROI")

# --- Batch: direct-pooling (voxel + cohort; no supervoxel chain) --------------
print("\n=== Batch: direct_pooling (voxel + cohort chains) ===")
dp_result = recipes.direct_pooling(cohort, direct_pooling_spec)
assert dp_result.habitat_model is not None
print(f"  habitats={dp_result.habitat_model.n_habitats}, "
      f"state keys={sorted(dp_result.habitat_model.preprocessing_state.keys())}")

# --- Batch: one-step (voxel chain only; per-subject models) ------------------
print("\n=== Batch: one_step (voxel chain only; no cohort chain) ===")
one_step_result = recipes.one_step(cohort, one_step_spec)
print(f"  subject_models: {len(one_step_result.subject_models)}")
print(f"  cohort habitat_model: {one_step_result.habitat_model}")
first_model = next(iter(one_step_result.subject_models.values()))
print(f"  example subject state keys: {sorted(first_model.preprocessing_state.keys())}")
assert "cohort_feature_preprocessor" not in first_model.preprocessing_state

# --- Train freeze + apply replay (cohort chain travels in HabitatModel) -------
print("\n=== Train freeze + apply replay (cohort preprocessing) ===")
archive = REPO_ROOT / "demo_data" / "results" / "docs_examples_tmp.habitatmodel"
archive.parent.mkdir(parents=True, exist_ok=True)
try:
    model.save(archive)
    reloaded = HabitatModel.load(archive)
    apply_batch = recipes.apply_habitat_model(cohort, two_step_spec, reloaded)
    apply_atomic = recipes.apply_habitat_model(single, two_step_spec, reloaded)
    print(f"  apply batch: {len(apply_batch.habitat_maps)} maps")
    print(f"  apply atomic (1 subject): {apply_atomic.habitat_maps[0].subject_id}")
    replay_map = apply_batch.pipeline(subject)
    print(f"  replay via apply_result.pipeline: {replay_map.subject_id}")
finally:
    if archive.is_file():
        archive.unlink()

print("\nMethods paragraph (two-step chains rendered):")
print(two_step_result.manifest.describe_methods()[:500], "...")

# Eye-check: open habitats on anatomy (napari). Set HABIT_NO_VIEW=1 to skip.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _habitat_eye_check import eye_check_study
eye_check_study(cohort, two_step_result)

