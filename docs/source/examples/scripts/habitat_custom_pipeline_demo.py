#!/usr/bin/env python
"""
Custom habitat pipelines: Registry.create, operator swaps, and Spec stages.

Shows three equivalent ways to customise the classical two-step design:

1. Construct domain operators by hand (or via Registry.create).
2. Compose a :class:`~habit.domain.SubjectPipeline` and fit a model.
3. Declare the same design as :class:`~habit.spec.HabitatSpec` stages so the
   recipe executor / YAML twin stays isomorphic.

Use this when the built-in recipes are the right *shape* but you need a
different voxel extractor, supervoxelizer, fitter, or quantify family.

Accompanies ``docs/source/examples/habitat_custom_pipeline.rst``.

Run from the repository root::

    python docs/source/examples/scripts/habitat_custom_pipeline_demo.py
"""

from __future__ import annotations

from habit import HabitatSpec, Spec, Stage, make_synthetic_cohort
from habit.domain import (
    HabitatModelFitterRegistry,
    HabitatVolumeFeatures,
    SubjectPipeline,
    SupervoxelizerRegistry,
    VoxelFeatureExtractorRegistry,
)
import habit.recipes as recipes


def main() -> None:
    """Build a custom two-step design three ways and check fingerprints."""
    cohort = make_synthetic_cohort(
        n_subjects=4,
        modalities=("T1", "T2"),
        shape=(18, 18, 18),
        rng=21,
    )
    modalities = ["T1", "T2"]

    # ------------------------------------------------------------------
    # 1) Registry.create — name + params_model fields (never invent synonyms)
    # ------------------------------------------------------------------
    voxel = VoxelFeatureExtractorRegistry.create(
        "raw",
        modalities=modalities,
    )
    # params_model for kmeans supervoxelizer uses n_supervoxels (not n_clusters).
    svx = SupervoxelizerRegistry.create(
        "kmeans",
        n_supervoxels=6,
        n_init=3,
    )
    if hasattr(svx, "set_random_state"):
        svx.set_random_state(21)

    fitter = HabitatModelFitterRegistry.create(
        "kmeans",
        n_habitats=3,
        n_init=5,
        min_habitats=2,
        max_habitats=3,
        validation="silhouette",
    )
    if hasattr(fitter, "set_random_state"):
        fitter.set_random_state(21)

    print("Registry.create components:")
    print(f"  voxel.spec = {voxel.spec}")
    print(f"  svx.spec   = {svx.spec}")
    print(f"  fit.spec   = {fitter.spec}")

    # ------------------------------------------------------------------
    # 2) Hand-assembled SubjectPipeline (custom procedure)
    # ------------------------------------------------------------------
    units = [svx(voxel(subject)) for subject in cohort]
    model = fitter.fit(units, cohort=cohort)
    pipe = SubjectPipeline(voxel, svx, model.assigner())
    one_map = pipe(cohort[0])
    one_table = pipe.extract_features(cohort[0], [HabitatVolumeFeatures()])
    print(
        f"Hand pipeline: map habitats="
        f"{sorted(int(v) for v in set(one_map.label_array.ravel()) if v)} "
        f"volume_features={len(one_table.feature_columns)}"
    )

    # ------------------------------------------------------------------
    # 3) Same design as HabitatSpec.stages (YAML-isomorphic)
    #    Swap any Stage Spec name/params to customise without rewriting
    #    Python callables — then fit_habitat runs the stage executor.
    # ------------------------------------------------------------------
    spec = HabitatSpec(
        name="custom_two_step",
        stages=(
            Stage(
                "extract_voxel_features",
                Spec("raw", {"modalities": modalities}),
            ),
            # Customisation point: change "kmeans" -> "slic", or add
            # Stage("preprocess1", Spec("winsorize", {...})) here.
            Stage(
                "partition",
                Spec("kmeans", {"n_supervoxels": 6, "n_init": 3}),
            ),
            Stage("pool", Spec("pool")),
            Stage(
                "fit",
                Spec(
                    "kmeans",
                    {
                        "n_habitats": 3,
                        "n_init": 5,
                        "min_habitats": 2,
                        "max_habitats": 3,
                        "validation": "silhouette",
                    },
                ),
            ),
            Stage("assign", Spec("nearest_centroid")),
            Stage("quantify", Spec("volume")),
            # Add more quantify* stages (msi, ith_score, ...) as needed.
        ),
        random_seed=21,
    )
    print(f"Spec fingerprint: {spec.fingerprint()}")

    result = recipes.fit_habitat(cohort, spec)
    assert result.habitat_model is not None
    print(
        f"fit_habitat: habitats={result.habitat_model.n_habitats}, "
        f"feature_cols={len(result.features.feature_columns)}"
    )

    # Atomic reuse of the recipe-built procedure on one subject:
    atomic_map = result.pipeline(cohort[0])
    print(
        f"result.pipeline(subject): subject={atomic_map.subject_id}, "
        f"nonzero_labels="
        f"{sorted(int(v) for v in set(atomic_map.label_array.ravel()) if v)}"
    )

    # Guidance: to customise further, change Stage Specs (above) OR swap the
    # Python operators before SubjectPipeline — keep model + procedure paired.
    print("Customisation tips:")
    print("  - New voxel formula: register a plugin, then Spec('your_name', ...)")
    print("  - Different partition: Spec('slic', {'n_supervoxels': 30})")
    print("  - Skip supervoxels: omit partition (one_step) or pool-only voxels")
    print("  - Always ship HabitatModel + the SubjectPipeline that matches it")


if __name__ == "__main__":
    main()
