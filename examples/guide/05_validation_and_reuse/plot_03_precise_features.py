"""
Precise features and habitat stability
======================================

**Background.** Voxel-wise texture features are noisy. Clustering on
features that flip under a simulated re-acquisition produces habitats
nobody can reproduce. Prior et al. (Radiol Artif Intell
2024;6(2):e230118) keep only features whose lower confidence limit
(LCL) of ICC clears a threshold under three experiments: repeatability
(original vs simulated retest), kernel-radius reproducibility, and
bin-width reproducibility. The same page then asks whether the habitat
**map** stays stable when the image is lightly perturbed.

**Purpose.** You will screen a small first-order + GLCM set on one demo
patient, fit one-step habitats on that patient with all features versus
only the precise subset, and score per-habitat Dice under a simulated
retest of the image (noise then small translation then small rotation).

**When to use.** Before committing a voxel-texture habitat study, and
when a reviewer asks which features and how stable the maps are under
rescan-like noise.

**Key terms.** Full definitions are on :doc:`/tutorial/concepts`.

* **simulated retest** -- a stronger rescan-like chain than the mild
  Appendix S2 defaults used elsewhere: Gaussian noise with an explicit
  ``sigma``, then a multi-voxel translation, then several degrees of
  rotation, on the intensity volume (mask unchanged). The point of this
  page is to make disagreement without precise screening visible.
* **precision panel** -- per-feature ICC with confidence limits on the
  voxels of one subject (or aggregated across subjects).
* **precise feature** -- a feature whose LCL clears the threshold in
  every experiment that was run (here LCL >= 0.5).
* **feature whitelist** -- preprocessor that drops every column except
  the precise names, so clustering uses exactly that subset.
* **habitat stability** -- per-habitat Dice after Hungarian matching of
  independently fitted label maps of the same subject.

This page is **not** the approved intensity two-step definition: it is
a one-step texture screen on a single sequence. Subject-level winsorize 1 % + z-score
is kept; cohort binning is omitted (one-step, one patient).

.. note::

   One patient and a short feature list are a demo of the mechanics.
   A published precise-feature set needs the paper's full feature list,
   many patients, and the reported LCL threshold of your study.

Change ``DATA`` / ``MODALITIES`` / ``ROI`` to your preprocessed tree.
"""

# %%
# Load one patient and one sequence
# ---------------------------------
# LAP alone keeps voxel texture extraction interactive. The screen and
# the habitat fits all use this single subject.
# sphinx_gallery_thumbnail_number = 3
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from habit.contracts import Cohort, cohort_from_directory
from habit.datasets import fetch_demo
from habit.precision import (
    aggregate_panels,
    align_habitat_map,
    habitat_stability,
    identify_precise_features,
    perturb_image,
    precision_panel,
)
from habit.execution import backend_from_policy
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.spec.policy import RunPolicy
from habit.viz import (
    plot_habitat_label_compare,
    plot_habitat_overlay,
    plot_precision_icc,
)
from habit.voxel_features import extract_voxel_texture

# Change DATA / MODALITIES / ROI to your preprocessed layout.
DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
subject = cohort[0]
image = subject.image(MODALITIES[0])
mask = subject.mask(ROI)
print(subject)
Path("out").mkdir(exist_ok=True)

# %%
# Simulated retest of the image
# -----------------------------
# Stronger than the mild Appendix S2 defaults so habitats without the
# precise whitelist disagree in a way you can see in the Dice table.
# One RNG advances through the chain.
retest_rng = np.random.default_rng(7)
noisy = perturb_image(
    image,
    method="gaussian_noise",
    sigma=float(np.nanstd(image.data)) * 0.15,
    rng=retest_rng,
)
shifted = perturb_image(
    noisy,
    method="translation",
    shift_voxels=(2.0, 1.5, 1.0),
    rng=retest_rng,
)
perturbed = perturb_image(
    shifted,
    method="rotation",
    angle_degrees=5.0,
    rng=retest_rng,
)
print(
    "retest chain: gaussian_noise(sigma=0.15*std) -> "
    "translation(2,1.5,1 voxels) -> rotation(5 deg)"
)
print("grid unchanged:", perturbed.data.shape == image.data.shape)

# %%
# Extract a small texture set at three settings
# ---------------------------------------------
# Base setting R1B25 (kernel radius 1, bin width 25) plus the two
# reproducibility siblings (R3 and B12). Radius 1 keeps this page fast;
# use the paper's R3B12 base when you reproduce Prior et al. fully.
FEATURE_CLASSES: Dict[str, Tuple[str, ...]] = {
    "firstorder": ("Entropy", "Mean", "Variance", "Skewness", "Kurtosis"),
    "glcm": (
        "Contrast",
        "Correlation",
        "JointEntropy",
        "Idm",
        "DifferenceEntropy",
    ),
    "glrlm": ("ShortRunEmphasis", "LongRunEmphasis", "RunEntropy"),
    "glszm": ("SmallAreaEmphasis", "LargeAreaEmphasis", "ZoneEntropy"),
    "gldm": ("SmallDependenceEmphasis", "LargeDependenceEmphasis"),
    "ngtdm": ("Coarseness", "Contrast", "Busyness"),
}
feat_r1 = extract_voxel_texture(
    image, mask, kernel_radius=1, bin_width=25, feature_classes=FEATURE_CLASSES
)
feat_r3 = extract_voxel_texture(
    image, mask, kernel_radius=3, bin_width=25, feature_classes=FEATURE_CLASSES
)
feat_b12 = extract_voxel_texture(
    image, mask, kernel_radius=1, bin_width=12, feature_classes=FEATURE_CLASSES
)
feat_pert = extract_voxel_texture(
    perturbed, mask, kernel_radius=1, bin_width=25, feature_classes=FEATURE_CLASSES
)
print("features:", list(feat_r1.feature_names))
print("voxels:", feat_r1.values.shape[0])

# %%
# Screen: precise = intersection of LCL flags
# -------------------------------------------
# repeatability: original vs retest at the base setting.
# reproducibility: R1 vs R3, and B25 vs B12, on the original image.
# ``identify_precise_features`` keeps names that clear LCL in every
# experiment that was run.
LCL_THRESHOLD = 0.5
repeat_panel = precision_panel(
    {"original": feat_r1, "perturbed": feat_pert},
    agreement="absolute",
)
kernel_panel = precision_panel(
    {"R1": feat_r1, "R3": feat_r3},
    agreement="consistency",
)
bin_panel = precision_panel(
    {"B25": feat_r1, "B12": feat_b12},
    agreement="consistency",
)
precise = identify_precise_features(
    {
        "repeatability": aggregate_panels([repeat_panel]),
        "reproducibility_kernel_radius": aggregate_panels([kernel_panel]),
        "reproducibility_bin_width": aggregate_panels([bin_panel]),
    },
    lcl_threshold=LCL_THRESHOLD,
)
evidence = precise.to_frame().round(3)
screened: Tuple[str, ...] = tuple(precise.feature_names)
all_names = list(dict.fromkeys(evidence["feature"].astype(str).tolist()))
unstable = [name for name in all_names if name not in set(screened)]
print("precise:", list(screened))
print("unstable:", unstable)
print(evidence.to_string(index=False))

fig = plot_precision_icc(evidence, title="Precision screen (one subject)")
fig.savefig("out/precise_features_icc.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Habitats: all texture features vs precise subset
# ------------------------------------------------
# One-step design on this single patient. Subject-level min-max only
# (no cohort binning). The precise run inserts the whitelist Spec before
# min-max so clustering sees only the screened columns.
texture_params: Dict[str, Any] = {
    "imageType": {"Original": {}},
    "featureClass": {key: list(values) for key, values in FEATURE_CLASSES.items()},
    "setting": {"binWidth": 25.0, "normalize": False},
}
extract_stage = Stage(
    "extract",
    Spec(
        "voxel_radiomics",
        {
            "modalities": list(MODALITIES),
            "roi": ROI,
            "kernel_radius": 1,
            "params": texture_params,
        },
    ),
)
fit_stage = Stage(
    "fit",
    Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 4, "validation": "elbow", "n_init": 5},
    ),
)
assign_stage = Stage("assign", Spec("nearest_centroid"))
volume_stage = Stage("volume", Spec("volume"))

spec_all = HabitatSpec(
    name="precise_all_texture",
    stages=(
        extract_stage,
        Stage("preprocess", Spec("minmax")),
        fit_stage,
        assign_stage,
        volume_stage,
    ),
    random_seed=11,
    pooling="none",
)
result_all = Study(spec_all).fit_predict(
    cohort,
    backend=backend_from_policy(
        RunPolicy(backend="serial", on_subject_failure="continue")
    ),
)

# Elbow / Kneedle caveat: HABIT elbow is Kneedle (Satopaa et al., 2011),
# not Thorndike (1953) informal elbow, not pre-v1.0 discrete-curvature.
# One-step maps keep each subject's Kneedle K (range 2..4 here).
_sid0 = cohort[0].subject_id
_model0 = result_all.subject_models[_sid0]
_report = _model0.preprocessing_state["selection_report"]
_cand = [int(k) for k in _report["candidates"]]
_iner = np.asarray(_report["scores"][_report["methods"][0]], dtype=float)
_k_kn = int(_model0.n_habitats)
_k_dc = int(_cand[int(np.argmax(np.diff(_iner, n=2))) + 1]) if len(_iner) >= 3 else _k_kn
from habit.habitat_model import KMeansHabitatModelFitter as _KF

_k_table = {"elbow_kneedle": _k_kn, "discrete_curvature_elbow": _k_dc}
for _name in ("silhouette", "calinski_harabasz", "davies_bouldin"):
    _f = _KF(min_habitats=2, max_habitats=4, validation=_name, n_init=5)
    _f.set_random_state(11)
    _k_table[_name] = _f.fit(result_all.units[0:1], cohort=cohort[:1]).n_habitats
print(f"{_sid0}: K by criterion (skip gap)")
for _n, _k in _k_table.items():
    print(f"  {_n}: {_k}")

def n_habitats_one_step(study_result: object) -> int:
    """Return K for a one-step run (per-subject model; no cohort model).

    Args:
        study_result: Output of ``Study(...).fit_predict`` with ``pooling='none'``.

    Returns:
        Number of habitats on the first subject.
    """
    # One-step stores a private model per subject; there is no cohort
    # ``habitat_model``. Fall back to unique positive labels if needed.
    models = getattr(study_result, "subject_models", None) or {}
    if models:
        first = next(iter(models.values()))
        return int(first.n_habitats)
    maps = getattr(study_result, "habitat_maps", ())
    labels = np.asarray(maps[0].label_array)
    return int(len(np.unique(labels[labels > 0])))


result_precise = None
if screened:
    whitelist = precise.preprocessor()
    spec_precise = HabitatSpec(
        name="precise_subset",
        stages=(
            extract_stage,
            Stage("whitelist", whitelist.spec),
            Stage("preprocess", Spec("minmax")),
            fit_stage,
            assign_stage,
            volume_stage,
        ),
        random_seed=11,
        pooling="none",
    )
    result_precise = Study(spec_precise).fit_predict(
    cohort,
    backend=backend_from_policy(
        RunPolicy(backend="serial", on_subject_failure="continue")
    ),
)
    print(
        "habitats all=",
        n_habitats_one_step(result_all),
        " precise=",
        n_habitats_one_step(result_precise),
    )
else:
    print("no feature passed every experiment; skip precise habitats")

fig = plot_habitat_overlay(
    image,
    result_all.habitat_maps[0],
    title=f"{subject.subject_id}: habitats (all texture features)",
    crop_to="labels",
)
fig.savefig("out/precise_features_all_habitats.png", dpi=150, bbox_inches="tight")
plt.show()

if result_precise is not None:
    fig = plot_habitat_overlay(
        image,
        result_precise.habitat_maps[0],
        title=f"{subject.subject_id}: habitats (precise subset)",
        crop_to="labels",
    )
    fig.savefig(
        "out/precise_features_precise_habitats.png", dpi=150, bbox_inches="tight"
    )
    plt.show()

# %%
# Habitat stability under the same retest
# ---------------------------------------
# Refit one-step habitats on the perturbed image with the same Spec,
# then match ids to the original map and report per-habitat Dice.
# ``force=True`` allows Hungarian matching across independently fitted
# model ids of the same subject.
from habit.contracts import ImageVolume, Subject


def with_image(base: Subject, new_image: ImageVolume) -> Subject:
    """Return a copy of ``base`` whose LAP image is ``new_image``.

    Args:
        base: Original subject (masks and other modalities unchanged).
        new_image: Replacement intensity volume on the same grid.

    Returns:
        New ``Subject`` with the same id and masks.
    """
    images = dict(base.images)
    images[MODALITIES[0]] = new_image
    return Subject(subject_id=base.subject_id, images=images, masks=dict(base.masks))


perturbed_subject = with_image(subject, perturbed)
perturbed_cohort = Cohort(subjects=(perturbed_subject,))
# Prefer the precise Spec when the screen kept at least one feature.
stability_spec = (
    HabitatSpec(
        name="precise_stability",
        stages=(
            extract_stage,
            Stage("whitelist", precise.preprocessor().spec),
            Stage("preprocess", Spec("minmax")),
            fit_stage,
            assign_stage,
            volume_stage,
        ),
        random_seed=11,
        pooling="none",
    )
    if screened
    else spec_all
)
ref_maps = (
    result_precise.habitat_maps
    if result_precise is not None
    else result_all.habitat_maps
)
mov_result = Study(stability_spec).fit_predict(
    perturbed_cohort,
    backend=backend_from_policy(
        RunPolicy(backend="serial", on_subject_failure="continue")
    ),
)
ref_map = ref_maps[0]
mov_map = mov_result.habitat_maps[0]
aligned = align_habitat_map(ref_map, mov_map, force=True)
dice = habitat_stability(ref_map, [mov_map])
print("Per-habitat Dice after simulated retest (Hungarian match):")
print(dice[["habitat_id", "matched_id", "dice"]].round(3).to_string(index=False))

fig = plot_habitat_label_compare(
    image,
    ref_map,
    aligned,
    titles=("Original image habitats", "Retest image habitats (matched)"),
    crop_to="labels",
)
fig.savefig("out/precise_features_stability.png", dpi=150, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(5.4, 3.2))
ax.bar(
    [f"H{int(h)}" for h in dice["habitat_id"].to_numpy()],
    dice["dice"].to_numpy(dtype=float),
    color="#0072B2",
)
ax.set_ylim(0.0, 1.05)
ax.set_ylabel("Dice")
ax.set_title("Per-habitat Dice under simulated retest")
fig.tight_layout()
fig.savefig("out/precise_features_dice.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# How to read the result
# ----------------------
# Measured on this run (subj001, LAP, 10 texture features, LCL >= 0.5):
#
# * Precise (7): Entropy, Mean, Variance, GLCM Contrast, JointEntropy,
#   Idm, DifferenceEntropy.
# * Unstable (3): Skewness, Kurtosis, GLCM Correlation (failed
#   repeatability and/or kernel-radius LCL; bin-width alone was not
#   enough to keep them).
# * One-step elbow kept K = 3 habitats for both the full set and the
#   precise subset.
# * Under the simulated retest, per-habitat Dice after Hungarian
#   matching was about 0.68--0.69 for all three habitats: the partition
#   moved, but not randomly.
#
# These numbers are one patient and a short feature list. Report your
# own LCL threshold, feature list and retest chain with the results.

# %%
# Where to go next
# ----------------
# * Texture habitats without the precision screen:
#   :doc:`/auto_examples/02_features/plot_02_texture_feature_maps`.
# * Robustness when the ROI outline moves, not the image:
#   :doc:`/auto_examples/05_validation_and_reuse/plot_03_precise_features`.
# * Matching habitat ids between two independent fits:
#   :doc:`/auto_examples/05_validation_and_reuse/plot_02_matching_labels`.
# * The intensity two-step reference analysis:
#   :doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec`.
