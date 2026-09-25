"""
Applying a saved habitat model
==============================

**Background.** Habitat ids only mean the same thing in two cohorts when both
are labelled with the same centroids. Saving the fitted model lets a later
or external cohort be labelled with the training definition instead of a
new, incomparable fit.

**Purpose.** You get a ``.habitatmodel`` file, the habitat map of a training
subject, and the habitat map and feature table of two held-out subjects
labelled from the reloaded file.

**Key terms.**

* **.habitatmodel** -- the saved habitat definition (centroids plus the
  preprocessing state they depend on); loading it labels new subjects
  without refitting.
* **fit vs. predict** -- ``fit_predict`` learns the centroids and labels the
  training cohort; ``predict`` only labels subjects with an existing model.

Train a two-step habitat definition, round-trip the
:class:`~habit.contracts.HabitatModel` through a ``.habitatmodel``
archive, and project it onto later subjects.

The spec must match between training and application: the model stores
cohort-level preprocessing state, but upstream stages are re-declared.
"""

# %%
# Demo pack has ``subj001`` … ``subj005``. Train on the first three;
# apply on the last two.
# sphinx_gallery_thumbnail_number = 2
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import HabitatModel, cohort_from_directory
from habit.datasets import fetch_demo
from habit.spec import HabitatSpec, Spec, Stage
from habit.viz import plot_habitat_label_compare, plot_habitat_overlay
import habit.recipes as recipes

DATA = fetch_demo()
# Three DCE phases: unenhanced, arterial, and portal-venous.
MODALITIES = ("pre_contrast", "LAP", "PVP")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
train_cohort = cohort[:3]
new_cohort = cohort[3:5]
print(f"Train: {list(train_cohort.subject_ids)}; apply: {list(new_cohort.subject_ids)}")

spec = HabitatSpec(
    name="habitat_two_step",
    stages=(
        # extract: raw intensities. The stage label can be any unique string.
        Stage("extract_voxel_features", Spec("raw", {"modalities": list(MODALITIES)})),
        # preprocess: per-subject winsorize, then min-max. Statistics are
        # refit on each subject; they are not part of the saved centroids.
        Stage(
            "preprocess1",
            Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
        ),
        Stage("preprocess2", Spec("minmax", {"across_features": False})),
        # partition + pool + fit: the shared definition that gets saved.
        Stage("partition", Spec("kmeans", {"n_supervoxels": 8, "n_init": 5})),
        Stage("pool", Spec("pool")),
        Stage(
            "fit",
            Spec("kmeans", {"n_habitats": 3, "n_init": 5}),
        ),
        # assign and quantify run again at predict time, using the loaded model.
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
        Stage("quantify2", Spec("msi")),
        Stage("quantify3", Spec("ith_score")),
    ),
    random_seed=42,
)

# %%
# Fit on the discovery cohort and persist one self-describing archive.
train_result = recipes.Study(spec=spec).fit_predict(train_cohort)
print(train_result.habitat_model.summary())
print(train_result.features.frame.head())
train_result.features.frame.head()

Path("out").mkdir(exist_ok=True)
archive = Path("out/habitat_model.habitatmodel")
# Versioned, self-describing archive: this file is what you share or keep
# next to a paper, not the Python objects above.
train_result.habitat_model.save(archive)
print(f"Saved {archive}")

fig = plot_habitat_overlay(
    train_cohort[0].image(ROI),
    train_result.habitat_maps[0],
    title="train habitats",
    crop_to="labels",
)
fig.savefig("out/apply_saved_train_overlay.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Reload and label held-out subjects. No fitting after the reload.
model = HabitatModel.load(archive)
# from_model pairs the loaded centroids with the same spec, so the new
# subjects' features are extracted and preprocessed as in training;
# predict still builds each new subject's supervoxels, but assigns them to
# the saved centroids; the habitat model is never refitted.
prediction = recipes.Study.from_model(model, spec).predict(new_cohort)
print(f"Applied to {list(s.subject_id for s in prediction.habitat_maps)}")
print(prediction.features.frame.head())
prediction.features.frame.head()

if prediction.habitat_maps:
    fig_new = plot_habitat_overlay(
        new_cohort[0].image(ROI),
        prediction.habitat_maps[0],
        title="applied habitats",
    )
    fig_new.savefig("out/apply_saved_new_overlay.png", dpi=150, bbox_inches="tight")
    plt.show()
