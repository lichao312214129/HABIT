"""
Quickstart: Python API
======================

Install first (:doc:`/tutorial/installation`). Construct a cohort, a
:class:`~habit.spec.HabitatSpec`, and call :mod:`habit.recipes`. No YAML.

This page uses the official demo pack. Change ``DATA`` / ``MODALITIES`` /
``ROI`` to your preprocessed tree.
"""

# sphinx_gallery_thumbnail_number = 2

# %%
# Load the official imaging pack (downloads once) and take two subjects
# so the run stays short. Drop the slice to use the full pack.
from pathlib import Path
import os

import matplotlib.pyplot as plt

from habit.contracts import HabitatModel, cohort_from_directory
from habit.datasets import fetch_demo
from habit.pipeline.assembly import build_habitat_components
from habit.spec import HabitatSpec, Spec
from habit.viz import plot_habitat_overlay, plot_partition_triptych
import habit.recipes as recipes

DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "pre_contrast"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
print(cohort)
print("subject_ids:", list(cohort.subject_ids))

# %%
# Two-step spec: voxel features, clustering-feature preprocessors,
# supervoxels, cohort binning, then fit / assign.
# ``supervoxelizer`` + default pooling selects the two-step design.
spec = HabitatSpec(
    name="habitat_two_step",
    voxel_feature_extractor=Spec("raw", {"modalities": list(MODALITIES)}),
    voxel_feature_preprocessors=(
        Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
        Spec("minmax", {"across_features": False}),
    ),
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 12, "max_iter": 300, "n_init": 5}),
    cohort_feature_preprocessors=(
        Spec("binning", {"n_bins": 10, "bin_strategy": "uniform", "across_features": False}),
    ),
    habitat_model_fitter=Spec(
        "kmeans",
        {
            "min_habitats": 2,
            "max_habitats": 10,
            "validation": "elbow",
            "max_iter": 300,
            "n_init": 5,
        },
    ),
    habitat_assigner=Spec("nearest_centroid"),
    habitat_features=(Spec("volume"), Spec("msi"), Spec("ith_score")),
    random_seed=42,
)
print(f"Spec fingerprint: {spec.fingerprint()[:16]}...")

# %%
# Clustering-feature tables **before** the voxel preprocessor chain.
# Same rows after winsorize + minmax. This is not image resampling.
pipe = build_habitat_components(spec).pipeline(assigner=None)
subject = cohort[0]
raw = pipe.voxel_feature_extractor(subject).feature_frame()
print("Voxel features before preprocessing:")
print(raw.head())
raw.head()

# %%
# Same voxel rows after the subject-level chain
# (:attr:`~habit.spec.HabitatSpec.voxel_feature_preprocessors`).
processed = pipe.voxel_feature_preprocessor(raw)
print("Voxel features after winsorize + minmax:")
print(processed.head())
processed.head()

# %%
# Fit habitats on the cohort and write maps. Overlay uses
# :func:`~habit.viz.plot_habitat_overlay` with volume objects, not ``.data``.
result = recipes.Study(spec=spec).fit_predict(cohort)
out_dir = Path("out/habitat_two_step")
result.save(out_dir, write_maps=True, write_units_table=True)
print(result.habitat_model.summary())
print("Habitat feature table:")
print(result.features.frame.head())
result.features.frame.head()

fig = plot_habitat_overlay(
    subject.image("LAP"),
    result.habitat_maps[0],
    title="habitats",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/quickstart_overlay.png", dpi=150, bbox_inches="tight")
if os.environ.get("HABIT_NO_VIEW") != "1":
    plt.show()

fig_tri = plot_partition_triptych(
    subject.image(ROI),
    result.units[0],
    result.habitat_maps[0],
    axis=0,
)
fig_tri.savefig("out/quickstart_triptych.png", dpi=150, bbox_inches="tight")
if os.environ.get("HABIT_NO_VIEW") != "1":
    plt.show()

# %%
# Reload the ``.habitatmodel`` and label the same cohort. Optional napari
# view (needs the ``view`` extra)::
#
#    from habit.viz import view_habitat_napari
#    view_habitat_napari(subject.image("LAP"), result.habitat_maps[0])
model = HabitatModel.load(out_dir / "habitat_model.habitatmodel")
prediction = recipes.Study.from_model(model, spec).predict(cohort)
print(len(prediction.habitat_maps), "subjects labelled")
