#!/usr/bin/env python
"""
Parallel habitat analysis with RunPolicy and ProcessPoolBackend.

``HabitatSpec`` declares *what* to compute; :class:`~habit.spec.RunPolicy`
declares *how* to schedule it (worker count, failure policy, timeouts).
Pass a :class:`~habit.execution.process_pool.ProcessPoolBackend` to any
habitat recipe to run subjects in parallel — the scientific result is
identical to serial execution when seeds are fixed.

This script accompanies ``docs/source/examples/parallel_execution.rst``.

Run from the repository root::

    python docs/source/examples/scripts/parallel_execution_demo.py
"""

from __future__ import annotations

from habit import HabitatSpec, RunPolicy, Spec, make_synthetic_cohort
from habit.execution.process_pool import ProcessPoolBackend
import habit.recipes as recipes


def main() -> None:
    """Compare serial and parallel two-step runs on a synthetic cohort."""
    cohort = make_synthetic_cohort(n_subjects=6, shape=(14, 14, 14), rng=21)
    spec = HabitatSpec(
        name="parallel_demo",
        voxel_feature_extractor=Spec("raw", {"modalities": ["T1", "T2"]}),
        supervoxelizer=Spec("kmeans", {"n_supervoxels": 6, "n_init": 3}),
        habitat_model_fitter=Spec(
            "kmeans",
            {"min_habitats": 2, "max_habitats": 3, "validation": "silhouette", "n_init": 3},
        ),
        habitat_assigner=Spec("nearest_centroid"),
        habitat_features=(Spec("volume"),),
        random_seed=21,
    )

    serial_result = recipes.two_step(cohort, spec)
    print(f"Serial: {len(serial_result.habitat_maps)} maps, "
          f"{serial_result.habitat_model.n_habitats} habitats")

    # Library default parallel_mode is "persistent" (long-lived workers).
    # Use parallel_mode="isolated" when you need a fresh child process per subject.
    policy = RunPolicy(
        workers=2,
        backend="process",
        on_subject_failure="continue",
        parallel_mode="persistent",
    )
    backend = ProcessPoolBackend.from_policy(policy)
    print(f"RunPolicy: workers={policy.workers}, backend={policy.backend!r}, "
          f"parallel_mode={policy.parallel_mode!r}")

    parallel_result = recipes.two_step(cohort, spec, backend=backend)
    print(f"Parallel: {len(parallel_result.habitat_maps)} maps, "
          f"{parallel_result.habitat_model.n_habitats} habitats")

    mismatches = sum(
        1
        for a, b in zip(serial_result.habitat_maps, parallel_result.habitat_maps)
        if not (a.label_array == b.label_array).all()
    )
    print(f"Label mismatches serial vs parallel: {mismatches} / {len(cohort)}")

    subject = cohort[0]
    pipeline = parallel_result.pipeline
    assert pipeline is not None
    habitat_map = pipeline(subject)
    print(f"Atomic predict on {subject.subject_id}: "
          f"{len(set(habitat_map.label_array[habitat_map.label_array > 0]))} labels")

    print("\nYAML twin: add a top-level ``policy:`` block (see config/habitat/*_wsl.yaml)")


if __name__ == "__main__":
    main()
