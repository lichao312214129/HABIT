#!/usr/bin/env python
"""
Atomic habitat analysis: call each domain operator without YAML / recipes.

Accompanies ``docs/source/examples/habitat_atomic_ops.rst``.

Run from the repository root::

    python docs/source/examples/scripts/habitat_atomic_ops_demo.py
"""

from __future__ import annotations

# BEGIN example
import numpy as np

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.habitat_features import HabitatVolumeFeatures, IthHabitatFeatures, MsiHabitatFeatures, NonRadiomicsHabitatFeatures
from habit.habitat_model import KMeansHabitatModelFitter
from habit.supervoxel import KMeansSupervoxelizer
from habit.voxel_features import RawVoxelFeatures
from habit.pipeline import SubjectPipeline
from habit.execution import SerialBackend

DATA = fetch_demo()
MODALITIES = ["LAP", "PVP"]
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=tuple(MODALITIES), roi=ROI)[:2]
print(f"Cohort: {len(cohort)} subjects -> {list(cohort.subject_ids)}")

voxel = RawVoxelFeatures(modalities=MODALITIES)
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

fit_pipe = SubjectPipeline(voxel, svx, habitat_assigner=None)
assert fit_pipe.units(subject0) is not None

apply_pipe = SubjectPipeline(voxel, svx, model.assigner())
habitat_map = apply_pipe(subject0)
present = sorted(int(v) for v in np.unique(habitat_map.label_array) if int(v) != 0)
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

maps = cohort.map(apply_pipe, backend=SerialBackend())
print(f"cohort.map -> {len(maps)} habitat maps")
# END example

# BEGIN figures
# Paste after the Script block. Uses subject0, habitat_map, and ROI.
from pathlib import Path

from habit.viz import plot_habitat_overlay

fig = plot_habitat_overlay(
    subject0.image(ROI),
    habitat_map,
    title="habitats",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/habitat_atomic_overlay.png", dpi=150, bbox_inches="tight")
print("Wrote out/habitat_atomic_overlay.png")
# END figures

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import copy_out_figures_to_gallery

    copy_out_figures_to_gallery(("habitat_atomic_overlay.png",))
