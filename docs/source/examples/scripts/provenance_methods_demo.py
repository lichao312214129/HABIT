#!/usr/bin/env python
"""
Provenance: Spec fingerprints and RunManifest.describe_methods().

Shows why provenance is part of the data model, not an optional log:

* Changing one Stage params changes ``HabitatSpec.fingerprint()``.
* ``StudyResult.manifest.describe_methods()`` emits a methods paragraph
  from the stages that actually ran.

Accompanies ``docs/source/examples/provenance_methods.rst``.

Run from the repository root::

    python docs/source/examples/scripts/provenance_methods_demo.py
"""

from __future__ import annotations

from habit import HabitatSpec, Spec, Stage, make_synthetic_cohort
import habit.recipes as recipes


def _base_stages(n_supervoxels: int) -> tuple[Stage, ...]:
    """Return a minimal two-step stage tuple."""
    return (
        Stage("extract_voxel_features", Spec("raw", {"modalities": ["T1", "T2"]})),
        Stage(
            "partition",
            Spec("kmeans", {"n_supervoxels": n_supervoxels, "n_init": 3}),
        ),
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
        Stage("quantify", Spec("volume")),
    )


def main() -> None:
    """Compare fingerprints and print a methods paragraph."""
    cohort = make_synthetic_cohort(n_subjects=3, shape=(16, 16, 16), rng=3)

    spec_a = HabitatSpec(
        name="prov_a",
        stages=_base_stages(6),
        random_seed=3,
    )
    spec_b = HabitatSpec(
        name="prov_b",
        stages=_base_stages(8),  # only n_supervoxels differs
        random_seed=3,
    )
    fp_a = spec_a.fingerprint()
    fp_b = spec_b.fingerprint()
    print(f"spec_a fingerprint: {fp_a[:16]}...")
    print(f"spec_b fingerprint: {fp_b[:16]}...")
    print(f"fingerprints equal: {fp_a == fp_b}")

    result = recipes.fit_habitat(cohort, spec_a)
    print("\n--- describe_methods (radiology style) ---")
    print(result.manifest.describe_methods(style="radiology"))
    versions = dict(result.manifest.software_versions())
    print("\nsoftware_versions (sample):", {k: versions[k] for k in list(versions)[:4]})
    print("random_seeds:", dict(result.manifest.random_seeds()))


if __name__ == "__main__":
    main()
