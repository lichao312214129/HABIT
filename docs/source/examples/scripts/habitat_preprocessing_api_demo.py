#!/usr/bin/env python
"""
Clustering feature-preprocessing chains (subject + cohort) via HabitatSpec.

This is NOT image preprocessing. Image intensity resampling lives in
``preprocess_subject`` / ``preprocess_images``. Here the rows are voxels or
supervoxels on the way to a habitat definition.

* **Subject chain** (stateless) — ``voxel_feature_preprocessors`` /
  ``supervoxel_feature_preprocessors``.
* **Cohort chain** (stateful; travels inside HabitatModel) —
  ``cohort_feature_preprocessors``.
* **Step inspection** — optional ``inspect=StepRecorder(...)`` on recipes.

Accompanies ``docs/source/examples/habitat_preprocessing_api.rst``.
"""

from __future__ import annotations

from pathlib import Path
import sys

from habit import HabitatSpec, Spec, StepRecorder, make_synthetic_cohort
from habit.contracts.subject import Cohort, Subject
from habit.domain.assembly import build_habitat_components, build_subject_chain
import habit.recipes as recipes


def main() -> None:
    """Run the preprocessing + step-inspection demo."""
    cohort = make_synthetic_cohort(n_subjects=3, shape=(14, 14, 14), rng=5)
    subject = cohort[0]
    _demo(cohort, subject)


def _demo(cohort: Cohort, subject: Subject) -> None:
    """
    Execute the documented preprocessing and inspection examples.

    Args:
        cohort: Synthetic cohort used for the recipe run.
        subject: One subject used for atomic inspection calls.
    """
    # BEGIN example
    # Sugar form keeps the documented chain field names for this page;
    # batch entry is Study(spec=...).fit_predict (stages expand under the hood).
    spec = HabitatSpec(
        name="feature_prep_demo",
        voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
        voxel_feature_preprocessors=(
            Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
            Spec("minmax", {"across_features": False}),
        ),
        supervoxelizer=Spec("kmeans", {"n_supervoxels": 6, "n_init": 3}),
        supervoxel_feature_extractor=Spec(
            "concat",
            {
                "children": [
                    {"name": "mean", "params": {"modality": "T1"}},
                    {"name": "std", "params": {"modality": "T1", "as_": "t1_spread"}},
                ],
            },
        ),
        cohort_feature_preprocessors=(
            Spec(
                "binning",
                {"n_bins": 6, "bin_strategy": "uniform", "across_features": False},
            ),
        ),
        habitat_model_fitter=Spec(
            "kmeans",
            {
                "min_habitats": 2,
                "max_habitats": 3,
                "validation": "elbow",
                "n_init": 3,
            },
        ),
        habitat_assigner=Spec("nearest_centroid"),
        habitat_features=(
            Spec("volume"),
            Spec("msi"),
            Spec("ith_score"),
            Spec("non_radiomics"),
        ),
        random_seed=5,
    )

    print("=== Spec declares both preprocessor levels ===")
    print(f"  voxel steps:  {[s.name for s in spec.voxel_feature_preprocessors]}")
    print(f"  cohort steps: {[s.name for s in spec.cohort_feature_preprocessors]}")

    components = build_habitat_components(spec)
    assert components.voxel_feature_extractor is not None
    assert components.voxel_feature_preprocessor is not None
    assert components.cohort_feature_preprocessor is not None
    assert components.habitat_model_fitter is not None
    print("=== HabitatComponents names align with Spec ===")
    print(f"  voxel_feature_extractor: {type(components.voxel_feature_extractor).__name__}")
    print(
        f"  voxel_feature_preprocessor steps: "
        f"{len(components.voxel_feature_preprocessor.methods)}"
    )

    recorder = StepRecorder(max_subjects=1)
    result = recipes.Study(spec=spec).fit_predict(cohort, inspect=recorder)
    assert result.habitat_model is not None
    assert result.inspection is recorder
    print(
        f"=== Study.fit_predict with feature chains: habitats={result.habitat_model.n_habitats} ==="
    )
    print("=== Step inspection (result.inspection) ===")
    print(result.inspection.summary().to_string(index=False))
    sid = subject.subject_id
    print(
        "  described supervoxel features:",
        list(
            result.inspection.frame(
                "extract_supervoxel_features.output", sid
            ).columns
        ),
    )

    chain = build_subject_chain(list(spec.voxel_feature_preprocessors))
    assert chain is not None
    raw_voxel = components.voxel_feature_extractor(subject).feature_frame()
    transformed = chain(raw_voxel)
    print("=== Atomic SubjectPreprocessingChain (on raw voxel features) ===")
    print(f"  in={raw_voxel.shape}, out={transformed.shape}")

    bare = HabitatSpec(
        name="bare_units",
        voxel_feature_extractor=spec.voxel_feature_extractor,
        supervoxelizer=spec.supervoxelizer,
        habitat_model_fitter=spec.habitat_model_fitter,
        habitat_assigner=spec.habitat_assigner,
        random_seed=spec.random_seed,
    )
    raw_units = build_habitat_components(bare).pipeline(assigner=None).units(subject)
    print("=== Raw supervoxel units (no feature preprocessing) ===")
    print(f"  features={raw_units.features.shape}")
    # END example

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import save_habitat_study_figures
    from _habitat_eye_check import eye_check_study

    save_habitat_study_figures(cohort, result, prefix="habitat_prep_api")
    # Eye-check: open habitats on anatomy (napari). Set HABIT_NO_VIEW=1 to skip.
    eye_check_study(cohort, result)



if __name__ == "__main__":
    main()
