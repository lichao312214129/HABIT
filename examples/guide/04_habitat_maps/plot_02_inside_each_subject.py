"""
Defining habitats inside each subject
======================================

Input: one subject at a time. Output: a
:class:`~habit.contracts.HabitatMap` whose integer ids belong to that
subject only. The stage list has no ``pool``, so ``fit`` runs inside each
subject instead of on the cohort. ``one_step_habitat(...)`` is a
shortcut that builds the same stage list.

Habitat 1 in the first subject is not habitat 1 in the second. Match
labels before comparing people:
:doc:`/auto_examples/04_habitat_maps/plot_05_match_labels`.
"""

# %%
# Load the cohort
# ---------------
# Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed layout.
# sphinx_gallery_thumbnail_number = 1
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.viz import plot_habitat_overlay

DATA = fetch_demo()
# Three DCE phases: unenhanced, arterial, and portal-venous.
MODALITIES = ("pre_contrast", "LAP", "PVP")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:3]
print(f"Cohort: {list(cohort.subject_ids)}")

# %%
# Fit a private model inside each subject
# ---------------------------------------
# No ``partition`` and no ``pool``: each subject's ROI voxels are
# clustered on their own. Habitat 1 in the first subject is not habitat 1
# in the second. There is no shared centroid.
spec = HabitatSpec(
    name="one_step",
    stages=(
        Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
        Stage("fit", Spec("kmeans", {"n_habitats": 3, "n_init": 10})),
        Stage("assign", Spec("nearest_centroid")),
        Stage("volume", Spec("volume")),
    ),
    random_seed=0,
)
result = Study(spec).fit_predict(cohort)
print(f"Per-subject models: {list(result.subject_models)}")
for habitat_map in result.habitat_maps:
    present = sorted(
        int(v) for v in np.unique(habitat_map.label_array) if int(v) != 0
    )
    print(habitat_map.subject_id, present)
print(result.features.frame)
result.features.frame

# %%
# Show each subject's habitat map on its own anatomy. No comparison --
# just the three maps side by side.
Path("out").mkdir(exist_ok=True)
for subject, habitat_map in zip(cohort, result.habitat_maps):
    fig = plot_habitat_overlay(
        subject.image(ROI),
        habitat_map,
        title=f"habitats ({habitat_map.subject_id})",
        crop_to="labels",
    )
    fig.savefig(
        f"out/one_step_{habitat_map.subject_id}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()
