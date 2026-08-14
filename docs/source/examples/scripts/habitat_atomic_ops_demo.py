#!/usr/bin/env python
"""
Atomic habitat analysis: call each domain operator without YAML / recipes.

Walks the classical two-step dataflow as single-argument callables:

    Subject
      -> voxel features
      -> supervoxels  (per subject)
      -> HabitatModel.fit on pooled units  (cohort-level, once)
      -> SubjectPipeline(assigner) on one subject
      -> habitat feature table for that subject

This is the embedding surface: third-party notebooks can debug one failing
case with ``pipeline(subject)`` and never accept HABIT directory layouts.

Accompanies ``docs/source/examples/habitat_atomic_ops.rst``.

Run from the repository root::

    python docs/source/examples/scripts/habitat_atomic_ops_demo.py
"""

from __future__ import annotations

from typing import List

import numpy as np

from habit import make_synthetic_cohort
from habit.domain import (
    HabitatVolumeFeatures,
    IthHabitatFeatures,
    KMeansHabitatModelFitter,
    KMeansSupervoxelizer,
    MsiHabitatFeatures,
    NonRadiomicsHabitatFeatures,
    RawVoxelFeatures,
    SubjectPipeline,
)
from habit.execution import SerialBackend


# BEGIN example
def main() -> tuple:
    """Run the atomic two-step walkthrough and print intermediate shapes."""
    cohort = make_synthetic_cohort(
        n_subjects=4,
        modalities=("T1", "T2"),
        shape=(20, 20, 20),
        rng=7,
    )
    modalities: List[str] = ["T1", "T2"]
    print(f"Cohort: {len(cohort)} subjects -> {list(cohort.subject_ids)}")

    # --- Subject-level operators (each is op(subject) or op(field)) ---------
    voxel = RawVoxelFeatures(modalities=modalities)
    # K-Means supervoxelizer is Seedable; SLIC is deterministic and has no
    # set_random_state — name the concrete component before documenting seeds.
    svx = KMeansSupervoxelizer(n_supervoxels=8, n_init=3)
    svx.set_random_state(7)

    subject0 = cohort[0]
    field = voxel(subject0)
    print(
        f"VoxelFeatureField[{subject0.subject_id}]: "
        f"voxels={field.values.shape[0]}, features={field.feature_names}"
    )

    units0 = svx(field)
    print(
        f"Supervoxelization[{subject0.subject_id}]: "
        f"n_units={len(units0.features)}, "
        f"label_max={int(units0.label_array.max())}"
    )

    # --- Cohort-level fit (the only shared definition step) ----------------
    units = [svx(voxel(subject)) for subject in cohort]
    fitter = KMeansHabitatModelFitter(
        n_habitats=3,
        n_init=5,
        validation="elbow",
        min_habitats=2,
        max_habitats=3,
    )
    fitter.set_random_state(7)
    model = fitter.fit(units, cohort=cohort)
    print(model.summary())

    # --- Definition + procedure: publishable pair --------------------------
    # FIT-TIME pipeline has assigner=None (units only). APPLY-TIME binds
    # model.assigner() so one callable labels any new Subject.
    fit_pipe = SubjectPipeline(voxel, svx, habitat_assigner=None)
    assert fit_pipe.units(subject0) is not None

    apply_pipe = SubjectPipeline(voxel, svx, model.assigner())
    habitat_map = apply_pipe(subject0)
    present = sorted(
        int(v) for v in np.unique(habitat_map.label_array) if int(v) != 0
    )
    print(
        f"HabitatMap[{habitat_map.subject_id}]: "
        f"habitats_present={present}, model_id={model.model_id}"
    )

    table = apply_pipe.extract_features(
        subject0,
        [
            HabitatVolumeFeatures(),
            MsiHabitatFeatures(),
            IthHabitatFeatures(),
            NonRadiomicsHabitatFeatures(),
        ],
    )
    print(
        f"Feature table (one subject): "
        f"{table.frame.shape[0]} row(s) x {len(table.feature_columns)} features"
    )
    print("First columns:", list(table.feature_columns)[:6])

    # Optional batch map — still no YAML; backend is an optional outer layer.
    maps = cohort.map(apply_pipe, backend=SerialBackend())
    print(f"cohort.map -> {len(maps)} habitat maps")
    return subject0, habitat_map


subject0, habitat_map = main()
# END example

# BEGIN figures
# Paste after the Script block. Uses subject0 and habitat_map.
from pathlib import Path

from habit.viz import plot_habitat_overlay

fig = plot_habitat_overlay(
    subject0.image("T1").data,
    habitat_map.label_array,
    axis=0,
    title="Atomic operators: habitat map",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/habitat_atomic_overlay.png", dpi=150, bbox_inches="tight")
print("Wrote out/habitat_atomic_overlay.png")
# END figures

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import save_example_figure

    save_example_figure(fig, "habitat_atomic_overlay.png")
