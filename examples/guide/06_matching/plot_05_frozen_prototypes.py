"""
Naming a new cohort with frozen prototypes
==========================================

Habitat names learned on a training cohort are part of the model
definition. A validation or external cohort must be named **with those
names**, not refitted together with the training subjects: refitting
moves the prototypes, so "habitat 2" would mean something slightly
different in every analysis.

Pass the training :class:`~habit.precision.HabitatPrototypeAlignment` as
``prototypes=``. New subjects get one assignment pass onto the stored
prototypes (no update), inherit the training ``model_id``, and must use
the same features and matching settings.
"""

# %%
# Per-subject habitats for five subjects
# --------------------------------------
# Relative enhancement, each subject chooses its own ``k`` (silhouette
# over 2..5), as in :doc:`/auto_examples/06_matching/plot_07_match_labels`.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import Cohort, cohort_from_directory
from habit.datasets import fetch_demo
from habit.exceptions import HABITAPIError
from habit.habitat_model import KMeansHabitatModelFitter
from habit.pipeline import voxel_units
from habit.precision import align_habitat_maps_to_prototypes
from habit.viz import plot_habitat_overlay, plot_prototype_matching
from habit.voxel_features import ExpressionVoxelFeatures

# Change DATA / MODALITIES / ROI and the expressions to your own layout.
DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
extractor = ExpressionVoxelFeatures(
    features={
        "rel_enh_lap": "(LAP - pre_contrast) / (pre_contrast + eps)",
        "rel_enh_pvp": "(PVP - pre_contrast) / (pre_contrast + eps)",
    },
    roi=ROI,
)

maps, models = [], []
for subject in cohort:
    units = voxel_units(extractor(subject))
    fitter = KMeansHabitatModelFitter(
        min_habitats=2, max_habitats=5, validation="silhouette", n_init=3
    )
    fitter.set_random_state(0)
    model = fitter.fit([units], cohort=Cohort([subject], name=subject.subject_id))
    maps.append(model.assigner()(units))
    models.append(model)
    print(f"{subject.subject_id}: {model.n_habitats} habitats")

# %%
# Train on three subjects, name two new ones
# ------------------------------------------
train_maps, train_models = maps[:3], models[:3]
new_maps, new_models = maps[3:], models[3:]

trained = align_habitat_maps_to_prototypes(train_maps, models=train_models)
named = align_habitat_maps_to_prototypes(new_maps, models=new_models, prototypes=trained)
print(f"training prototypes (K={trained.prototypes.shape[0]}):\n{trained.prototypes.round(3)}")
print(named.assignments.to_string(index=False))
print("model_id shared with training:", {m.model_id for m in named.habitat_maps} == {trained.model_id})

# %%
# Left: how the training subjects defined the prototypes. Right: the new
# subjects' habitats pulled onto those same, unmoved prototypes.
Path("out").mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharex=True, sharey=True)
plot_prototype_matching(
    [np.asarray(m.centroids) for m in train_models], trained,
    title="training cohort (prototypes fitted)", ax=axes[0],
)
plot_prototype_matching(
    [np.asarray(m.centroids) for m in new_models], named,
    title="new cohort (prototypes frozen)", ax=axes[1],
)
fig.tight_layout()
fig.savefig("out/frozen_prototypes.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Frozen naming still names every habitat. Read the ``distance`` column:
# a new habitat far from every prototype resembles no training habitat,
# yet it gets the nearest free name. ``max_distance`` can leave such
# habitats unnamed instead
# (:doc:`/auto_examples/06_matching/plot_06_downstream_tables`).
#
# A new subject's map in training ids. Habitat ``k`` has the colour of
# prototype ``Pk`` above.
subject = cohort[3]
fig = plot_habitat_overlay(
    subject.image(ROI), named.habitat_maps[0], title=f"{subject.subject_id} (training ids)", crop_to="labels"
)
fig.savefig("out/frozen_new_subject.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Why not just refit on everyone?
# -------------------------------
# Refitting with the new subjects moves the prototypes, i.e. changes the
# definition the training analysis was reported with.
refit = align_habitat_maps_to_prototypes(maps, models=models)
if refit.prototypes.shape == trained.prototypes.shape:
    shift = float(np.max(np.abs(refit.prototypes - trained.prototypes)))
    print(f"largest prototype shift after refitting on all subjects: {shift:.3f}")
else:
    print(f"refitting even changes K: {trained.prototypes.shape[0]} -> {refit.prototypes.shape[0]}")
print("refit model_id equals training:", refit.model_id == trained.model_id)

# %%
# Settings are part of the definition
# -----------------------------------
# Naming with another metric (or other features, z-score, reduction)
# would compare against prototypes built under different rules, so it is
# refused instead of silently re-interpreted.
try:
    align_habitat_maps_to_prototypes(new_maps, models=new_models, metric="manhattan", prototypes=trained)
except HABITAPIError as error:
    print(error)
