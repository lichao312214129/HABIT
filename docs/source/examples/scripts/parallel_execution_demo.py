#!/usr/bin/env python
"""
Parallel habitat analysis with RunPolicy and ProcessPoolBackend.

``HabitatSpec.stages`` declares *what* to compute; :class:`~habit.spec.RunPolicy`
declares *how* to schedule it (worker count, failure policy, timeouts).
Pass a :class:`~habit.execution.process_pool.ProcessPoolBackend` to
:meth:`~habit.recipes.Study.fit_predict` to run subjects in parallel — the
scientific result is identical to serial execution when seeds are fixed.

This script accompanies ``docs/source/examples/parallel_execution.rst``.

Run from the repository root::

    python docs/source/examples/scripts/parallel_execution_demo.py
"""

from __future__ import annotations

from habit.spec import HabitatSpec, RunPolicy, Spec, Stage
from habit.datasets import make_synthetic_cohort
from habit.execution.process_pool import ProcessPoolBackend
import habit.recipes as recipes


# BEGIN example
def main() -> tuple:
    """Compare serial and parallel two-step runs on a synthetic cohort."""
    cohort = make_synthetic_cohort(n_subjects=6, shape=(14, 14, 14), rng=21)
    spec = HabitatSpec(
        name="parallel_demo",
        stages=(
            Stage("extract_voxel_features", Spec("raw", {"modalities": ["T1", "T2"]})),
            Stage("partition", Spec("kmeans", {"n_supervoxels": 6, "n_init": 3})),
            Stage("pool", Spec("pool")),
            Stage(
                "fit",
                Spec(
                    "kmeans",
                    {
                        "min_habitats": 2,
                        "max_habitats": 3,
                        "validation": "elbow",
                        "n_init": 3,
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
        random_seed=21,
    )

    serial_result = recipes.Study(spec=spec).fit_predict(cohort)
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

    parallel_result = recipes.Study(spec=spec).fit_predict(cohort, backend=backend)
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
    return cohort, parallel_result


# Windows spawn re-imports this script; ProcessPoolBackend must stay here.
if __name__ == "__main__":
    cohort, parallel_result = main()
# END example

# BEGIN figures
# Paste after the Script block, still inside ``if __name__ == "__main__":``.
from pathlib import Path

from habit.viz import plot_habitat_overlay

if __name__ == "__main__":
    Path("out").mkdir(exist_ok=True)
    fig = plot_habitat_overlay(
        cohort[0].image("T1"),
        parallel_result.habitat_maps[0],
        title="habitats",
    )
    fig.savefig("out/parallel_execution_overlay.png", dpi=150, bbox_inches="tight")
    print("Wrote out/parallel_execution_overlay.png")
# END figures

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import save_example_figure
    from _habitat_eye_check import eye_check_study

    save_example_figure(fig, "parallel_execution_overlay.png")
    eye_check_study(cohort, parallel_result)
