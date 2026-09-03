"""
Atomic operators
================

Each subject-level step is a single-argument callable. Call
``op(subject)`` (or ``op(field)``) with no :class:`~habit.recipes.Study`.

* Voxel features: ``voxel(subject)`` → :class:`~habit.contracts.VoxelFeatureField`
* Supervoxels: ``svx(field)`` → :class:`~habit.contracts.Supervoxelization`
* Fit (cohort): ``fitter.fit(units, cohort=...)`` → :class:`~habit.contracts.HabitatModel`
* Assign: ``model.assigner()(units)`` → :class:`~habit.contracts.HabitatMap`
* Pipeline: ``pipe(subject)`` → ``HabitatMap``
"""

# sphinx_gallery_thumbnail_number = 2

# %%
# Load two demo subjects and build the voxel extractor + supervoxelizer
# as plain callables.
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution import SerialBackend
from habit.habitat_features import (
    HabitatVolumeFeatures,
    IthHabitatFeatures,
    MsiHabitatFeatures,
)
from habit.habitat_model import KMeansHabitatModelFitter
from habit.pipeline import SubjectPipeline
from habit.supervoxel import KMeansSupervoxelizer
from habit.viz import plot_habitat_clustering_pca_2d, plot_habitat_overlay
from habit.voxel_features import RawVoxelFeatures

DATA = fetch_demo()
MODALITIES = ["LAP", "PVP"]
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=tuple(MODALITIES), roi=ROI)[:2]
print(f"Cohort: {len(cohort)} subjects -> {list(cohort.subject_ids)}")

voxel = RawVoxelFeatures(modalities=MODALITIES)
svx = KMeansSupervoxelizer(n_supervoxels=8, n_init=3)
svx.set_random_state(7)

# %%
# Voxel-feature table for the first subject. Each row is one ROI voxel.
subject0 = cohort[0]
field = voxel(subject0)
print(field.feature_frame().head())
field.feature_frame().head()

# %%
# Supervoxels, then a cohort-level k-means fit. Print unit counts so the
# partition is visible before assignment.
units0 = svx(field)
print(f"n_units={len(units0.features)}, label_max={int(units0.label_array.max())}")
print(units0.feature_frame().head())
units0.feature_frame().head()

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

# %%
# Bind extract + partition + assigner as a
# :class:`~habit.pipeline.SubjectPipeline` and overlay the labels.
apply_pipe = SubjectPipeline(voxel, svx, model.assigner())
habitat_map = apply_pipe(subject0)
present = sorted(int(v) for v in np.unique(habitat_map.label_array) if int(v) != 0)
print(f"habitats_present={present}, model_id={model.model_id}")

table = apply_pipe.extract_features(
    subject0,
    [HabitatVolumeFeatures(), MsiHabitatFeatures(), IthHabitatFeatures()],
)
print(table.frame.head())
table.frame.head()

fig = plot_habitat_overlay(
    subject0.image(ROI),
    habitat_map,
    title="habitats",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/habitat_atomic_overlay.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Population-level PCA of clustering units (supervoxel rows). Colour is
# nearest-centroid habitat id; crosses are the fitted cohort centres.
feature_names = list(model.feature_names)
unit_matrices = [
    np.asarray(unit.features[feature_names].to_numpy(dtype=np.float64))
    for unit in units
]
pca_features = np.vstack(unit_matrices)
pca_labels_parts = []
for matrix in unit_matrices:
    distances = np.linalg.norm(
        matrix[:, None, :] - model.centroids[None, :, :], axis=2
    )
    pca_labels_parts.append(np.argmin(distances, axis=1).astype(np.int64) + 1)
pca_labels = np.concatenate(pca_labels_parts)
fig_pca = plot_habitat_clustering_pca_2d(
    pca_features,
    pca_labels,
    centers=model.centroids,
    title="Habitat clustering (PCA)",
)
fig_pca.savefig("out/habitat_atomic_pca_2d.png", dpi=150, bbox_inches="tight")
plt.show()

maps = cohort.map(apply_pipe, backend=SerialBackend())
print(f"cohort.map -> {len(maps)} habitat maps")
