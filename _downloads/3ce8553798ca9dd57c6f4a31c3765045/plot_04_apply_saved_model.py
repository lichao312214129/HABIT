"""
Apply a saved model
===================

Train a two-step habitat definition, round-trip the
:class:`~habit.contracts.HabitatModel` through a ``.habitatmodel``
archive, and project it onto later subjects.

The spec must match between training and application: the model stores
cohort-level preprocessing state, but upstream stages are re-declared.
"""

# %%
# Demo pack has ``subj001`` … ``subj005``. Train on the first three;
# apply on the last two.
from pathlib import Path

import matplotlib.pyplot as plt

from habit.contracts import HabitatModel, cohort_from_directory
from habit.datasets import fetch_demo
from habit.spec import HabitatSpec, Spec, Stage
from habit.viz import plot_habitat_overlay
import habit.recipes as recipes

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
train_cohort = cohort[:3]
new_cohort = cohort[3:5]
print(f"Train: {list(train_cohort.subject_ids)}; apply: {list(new_cohort.subject_ids)}")

spec = HabitatSpec(
    name="habitat_two_step",
    stages=(
        Stage("extract_voxel_features", Spec("raw", {"modalities": list(MODALITIES)})),
        Stage(
            "preprocess1",
            Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
        ),
        Stage("preprocess2", Spec("minmax", {"across_features": False})),
        Stage("partition", Spec("kmeans", {"n_supervoxels": 8, "n_init": 5})),
        Stage("pool", Spec("pool")),
        Stage(
            "fit",
            Spec("kmeans", {"n_habitats": 3, "n_init": 5}),
        ),
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
train_result.habitat_model.save(archive)
print(f"Saved {archive}")

fig = plot_habitat_overlay(
    train_cohort[0].image(ROI),
    train_result.habitat_maps[0],
    title="train habitats",
)
fig.savefig("out/apply_saved_train_overlay.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Reload and label held-out subjects. No fitting after the reload.
model = HabitatModel.load(archive)
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
