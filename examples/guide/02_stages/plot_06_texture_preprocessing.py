"""
Preprocessing voxel texture before clustering
=============================================

Voxel texture columns live on very different scales: GLCM contrast runs
into the hundreds, correlation stays within [-1, 1]. Clustering then sees
mostly the large columns, so a preprocessing step comes first.

This page extracts texture **once** and compares five preprocessing
choices on the same voxels:

* **inside each subject** (no state, one lesion at a time): z-score,
  robust scaling, winsorizing;
* **fitted on the cohort** (state learned on the pooled training rows,
  then replayed on every subject): z-score, binning.

Each choice is then clustered and the habitat maps are compared with the
subject z-score map by voxel overlap. The last cell shows that a
:class:`~habit.spec.HabitatSpec` produces the same labels as the atomic
calls.
"""

# %%
# Load the cohort
# ---------------
# Four DCE series, arterial-phase ROI, two subjects.
# sphinx_gallery_thumbnail_number = 2
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution import SerialBackend
from habit.feature_preprocessing import (
    Binning,
    CohortPreprocessingChain,
    RobustScaling,
    SubjectPreprocessingChain,
    Winsorizing,
    ZScoreScaling,
)
from habit.habitat_model import KMeansHabitatModelFitter
from habit.pipeline import voxel_units
from habit.precision import habitat_stability
from habit.viz import plot_habitat_label_compare
from habit.voxel_features import VoxelRadiomicsFeatures

# Change DATA / MODALITIES / ROI to your preprocessed layout.
DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
print(cohort)
Path("out").mkdir(exist_ok=True)
# A cache hit skips PyRadiomics. Device and voxel_batch are not part of the key.
CACHE = str((Path("out") / "voxel_texture_cache").resolve())

# %%
# Extract voxel texture once
# --------------------------
# Radius 3 is a 7×7×7 neighbourhood; ``binWidth`` is the PyRadiomics
# grey-level width. Each feature is repeated on every series, so a column
# looks like ``original_glcm_Contrast-LAP``. The device is left at its
# default (``"auto"``): CUDA when available, otherwise CPU. How the two
# paths compare numerically: :doc:`/how_to/voxel_texture`.
RADIOMICS_PARAMS = {
    "imageType": {"Original": {}},
    "featureClass": {
        "firstorder": ["Mean", "Entropy"],
        "glcm": ["Contrast", "Correlation", "Idm", "JointEntropy"],
        "glrlm": ["ShortRunEmphasis", "LongRunEmphasis"],
    },
    "setting": {"binWidth": 12},
}
texture = VoxelRadiomicsFeatures(
    modalities=list(MODALITIES),
    roi=ROI,
    kernel_radius=3,
    params=RADIOMICS_PARAMS,
    cache_dir=CACHE,
)
backend = SerialBackend()
fields = [slot.result() for slot in backend.map(texture, cohort)]
for field in fields:
    print(f"{field.subject_id}: {field.values.shape[0]} voxels, {len(field.feature_names)} columns")
print(fields[0].feature_frame().describe().T[["mean", "std", "min", "max"]].round(2).head(6))

# %%
# Five preprocessing chains
# -------------------------
# A subject chain is called on one frame and keeps no state. A cohort chain
# is ``fit`` on the pooled rows of every training subject, then
# ``transform`` replays that state; this state belongs to the habitat
# definition and has to travel with the model. Impute is inserted in front
# of each method automatically.
subject_chains = {
    "subject z-score": SubjectPreprocessingChain([ZScoreScaling(across_features=False)]),
    "subject robust": SubjectPreprocessingChain([RobustScaling(across_features=False)]),
    "subject winsorize": SubjectPreprocessingChain(
        [Winsorizing(winsor_limits=(0.05, 0.05), across_features=False)]
    ),
}
cohort_chains = {
    "cohort z-score": CohortPreprocessingChain([ZScoreScaling(across_features=False)]),
    "cohort binning": CohortPreprocessingChain(
        [Binning(n_bins=6, bin_strategy="uniform", across_features=False)]
    ),
}
pooled = pd.concat([field.feature_frame() for field in fields], axis=0, ignore_index=True)
for chain in cohort_chains.values():
    chain.fit(pooled)

processed = {}
for name, chain in subject_chains.items():
    processed[name] = [
        field.with_feature_frame(
            chain(field.feature_frame()),
            produced_by="subject_feature_preprocessor",
            spec_fingerprint=chain.spec.fingerprint(),
        )
        for field in fields
    ]
for name, chain in cohort_chains.items():
    processed[name] = [
        field.with_feature_frame(
            chain.transform(field.feature_frame()),
            produced_by="cohort_feature_preprocessor",
            spec_fingerprint=chain.spec.fingerprint(),
        )
        for field in fields
    ]

# %%
# What each chain does to one column
# ----------------------------------
# GLCM contrast on the arterial series, per subject. Subject z-score puts
# every subject at mean 0 / sd 1; cohort z-score puts only the *pool*
# there, so the two subjects keep their difference. Robust scaling centres
# the median, winsorizing only clips the tails, binning stores bin indices.
# Uniform bins span min..max of the pool, so one extreme voxel (412 in
# subj002) stretches them and nearly all of subj001 falls into bins 0-1;
# ``bin_strategy="quantile"`` avoids that.
column = next(n for n in fields[0].feature_names if "Contrast" in n and n.endswith("-LAP"))
rows = []
for name, values in [("raw", fields)] + list(processed.items()):
    for field in values:
        x = field.feature_frame()[column]
        rows.append(
            {"chain": name, "subject": field.subject_id, "mean": x.mean(), "sd": x.std(),
             "median": x.median(), "min": x.min(), "max": x.max()}
        )
print(column)
print(pd.DataFrame(rows).round(2).to_string(index=False))

fig_hist, axes = plt.subplots(2, 3, figsize=(9, 5), constrained_layout=True)
for ax, (name, values) in zip(axes.ravel(), [("raw", fields)] + list(processed.items())):
    for field in values:
        ax.hist(field.feature_frame()[column].to_numpy(), bins=30, alpha=0.6, label=field.subject_id)
    ax.set_title(name)
axes[0, 0].legend(frameon=False)
fig_hist.suptitle(column)
fig_hist.savefig("out/texture_preprocessing_hist.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Cluster each preprocessed field
# -------------------------------
# Same fitter and seed for every chain: 3 habitats on the pooled voxels.
# Assignment reuses the fitted units; texture is not extracted again.
habitat_maps = {}
for name, values in processed.items():
    fitter = KMeansHabitatModelFitter(n_habitats=3, n_init=3)
    fitter.set_random_state(0)
    units = [voxel_units(field) for field in values]
    model = fitter.fit(units, cohort=cohort)
    habitat_maps[name] = [slot.result() for slot in backend.map(model.assigner(), units)]

# %%
# How far each map is from the subject z-score map
# ------------------------------------------------
# The ids of two independent fits are arbitrary, so habitats are paired by
# voxel overlap first (:func:`~habit.precision.habitat_stability`), then
# Dice is computed per pair. The table is the mean Dice over the three
# habitats of each subject. Winsorizing keeps each column in its own units,
# so the large-valued columns still dominate the distance and the map moves
# furthest from the scaled chains.
reference_name = "subject z-score"
dice_rows = []
for name, maps in habitat_maps.items():
    for reference, other in zip(habitat_maps[reference_name], maps):
        scores = habitat_stability(reference, [other])
        dice_rows.append({"chain": name, "subject": reference.subject_id, "mean Dice": scores["dice"].mean()})
dice_table = pd.DataFrame(dice_rows).pivot(index="chain", columns="subject", values="mean Dice")
print(dice_table.round(3))

fig_compare = plot_habitat_label_compare(
    cohort[0].image(ROI),
    habitat_maps[reference_name][0],
    habitat_maps["cohort binning"][0],
    titles=(reference_name, "cohort binning"),
    crop_to="labels",
)
fig_compare.savefig("out/texture_preprocessing_compare.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# The same analysis as a HabitatSpec
# ----------------------------------
# ``stages`` names the same extractor, the same subject z-score and the
# same fitter; ``pool`` with no partition is the pooled-voxel fit and
# ``random_seed=0`` is ``set_random_state(0)``. The labels match the
# atomic calls voxel for voxel.
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage

spec = HabitatSpec(
    name="texture_subject_zscore",
    stages=(
        Stage(
            "extract_voxel_features",
            Spec(
                "voxel_radiomics",
                {
                    "modalities": list(MODALITIES),
                    "roi": ROI,
                    "kernel_radius": 3,
                    "params": RADIOMICS_PARAMS,
                    "cache_dir": CACHE,
                },
            ),
            role="extract_voxel_features",
        ),
        Stage("preprocess_voxels", Spec("zscore", {"across_features": False}), role="preprocess"),
        Stage("pool", Spec("pool"), role="pool"),
        Stage("fit", Spec("kmeans", {"n_habitats": 3, "n_init": 3}), role="fit"),
        Stage("assign", Spec("nearest_centroid"), role="assign"),
    ),
    random_seed=0,
)
result = Study(spec).fit_predict(cohort, backend=backend)
for atomic, from_spec in zip(habitat_maps[reference_name], result.habitat_maps):
    same = np.array_equal(atomic.label_array, from_spec.label_array)
    print(f"{atomic.subject_id}: spec labels == atomic labels: {same}")
