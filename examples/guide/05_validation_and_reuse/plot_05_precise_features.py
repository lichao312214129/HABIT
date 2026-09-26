"""
Precise features vs all textures under perturbation
===================================================

Voxel texture features that flip under a simulated re-acquisition make
unstable habitats. Prior et al. (Radiol Artif Intell 2024;6(2):e230118)
keep features whose ICC lower confidence limit (LCL) clears a threshold
**across a large cohort**, then drop Spearman-redundant columns
(``select_precise_correlation_columns`` / ``precise_correlation_filter``).

This page has two separate parts:

1. **Self-screen demo only** (5 demo subjects): run HABIT's ICC LCL screen
   plus Prior-style Spearman redundancy filtering. ``n=5`` is too small,
   so that whitelist is unstable and is **not** used for habitat maps.
2. **Habitat comparison** (all 5 subjects): original vs Appendix S2
   perturbation, once with **all** extracted textures and once with
   Prior's **published** precise feature list
   (``robust_names`` in radiomicsgroup/precise-habitats
   ``habitat_computation.py``, 26 ICC-precise names, mapped to HABIT's
   ``original_*`` columns with the modality suffix). Subject-level
   winsorize 1 % then z-score; fixed ``n_habitats=3``. Dice is per
   subject after Hungarian matching; the five-subject summary is the
   **median** (not the mean).

Intensity overlays use one subject only so the page stays readable.
"""

# %%
# Load the full demo cohort
# -------------------------
# sphinx_gallery_thumbnail_number = 4
# Habitats and the self-screen demo both use every subject. Only the raw
# intensity pair below is limited to the first subject for readability.
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from habit.contracts import (
    Cohort,
    HabitatMap,
    ImageVolume,
    Subject,
    cohort_from_directory,
)
from habit.datasets import fetch_demo
from habit.execution import backend_from_policy
from habit.feature_preprocessing import FeatureWhitelist, PreciseCorrelationFilter
from habit.precision import (
    aggregate_panels,
    align_habitat_map,
    habitat_stability,
    identify_precise_features,
    prior2024_retest_perturbation,
    precision_panel,
)
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.spec.policy import RunPolicy
from habit.viz import plot_habitat_overlay, plot_intensity_slice
from habit.voxel_features import extract_voxel_texture

# Change DATA / MODALITIES / ROI to your preprocessed layout.
DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)
print(f"demo cohort size: {len(cohort)}")
if len(cohort) < 5:
    raise RuntimeError(
        f"expected at least 5 demo subjects; got {len(cohort)}"
    )
# Intensity pair only: first subject keeps the page readable.
example = cohort[0]
image = example.image(MODALITIES[0])
mask = example.mask(ROI)
Path("out").mkdir(exist_ok=True)

# %%
# Appendix S2 retest on every subject
# -----------------------------------
# Paper defaults: Chang-estimated Gaussian noise, then sub-voxel
# translation (shift_fraction=0.5), then 0.5 deg in-plane rotation.
# Masks stay on the original grid (warp_masks=False), matching the
# Prior 2024 GitHub pairing (perturbed CT + original ROI).
chain = prior2024_retest_perturbation(
    shift_fraction=0.5,
    angle_degrees=0.5,
    warp_masks=False,
)
rng = np.random.default_rng(7)
perturbed_subjects = []
for one in cohort:
    # Independent chain draws per subject; same seed stream order.
    perturbed_subjects.append(chain(one, rng=rng))
pert_cohort = Cohort(subjects=tuple(perturbed_subjects))
example_pert = pert_cohort[0]
perturbed = example_pert.image(MODALITIES[0])
print(
    "perturbation (Prior 2024 Appendix S2 / MIRP 1.2.0): "
    "Chang gaussian_noise -> translation shift_fraction=0.5 -> "
    "rotation angle_degrees=0.5"
)

# %%
# Original vs perturbed image (example subject only)
# --------------------------------------------------
# Display one subject so the page stays readable. Habitats below use all
# five subjects.
for vol, name, stem in (
    (image, "original", "precise_image_original"),
    (perturbed, "perturbed (Appendix S2)", "precise_image_perturbed"),
):
    fig = plot_intensity_slice(
        vol,
        roi_mask=mask,
        roi_contour=True,
        image_label=MODALITIES[0],
        title=f"{example.subject_id}: {name}",
    )
    fig.savefig(f"out/{stem}.png", dpi=150, bbox_inches="tight")
    plt.show()

# %%
# Part 1 — Self-screen demo (ICC LCL + Spearman; not for habitats)
# ----------------------------------------------------------------
# Same three experiments as the paper (repeatability, kernel radius,
# bin width). Then Prior-style Spearman redundancy filter on one
# subject's LCL survivors: signed r > 0.7 and p < 0.05 drops the
# *earlier* column and keeps the later one
# (``PreciseCorrelationFilter`` / ``select_precise_correlation_columns``).
# Caveat: n=5 is too small; do NOT use this whitelist for the habitat
# comparison below.
LCL_THRESHOLD = 0.5
repeat_panels = []
kernel_panels = []
bin_panels = []
feat_r1_example = None
for orig_subj, pert_subj in zip(cohort, pert_cohort):
    vol = orig_subj.image(MODALITIES[0])
    vol_p = pert_subj.image(MODALITIES[0])
    m = orig_subj.mask(ROI)
    # Base setting: kernel_radius=1, bin_width=12 (same as habitat extract).
    feat_r1 = extract_voxel_texture(vol, m, kernel_radius=1, bin_width=12)
    feat_r3 = extract_voxel_texture(vol, m, kernel_radius=3, bin_width=12)
    feat_b25 = extract_voxel_texture(vol, m, kernel_radius=1, bin_width=25)
    feat_pert = extract_voxel_texture(vol_p, m, kernel_radius=1, bin_width=12)
    if feat_r1_example is None:
        feat_r1_example = feat_r1
    repeat_panels.append(
        precision_panel(
            {"original": feat_r1, "perturbed": feat_pert},
            agreement="absolute",
        )
    )
    kernel_panels.append(
        precision_panel({"R1": feat_r1, "R3": feat_r3}, agreement="consistency")
    )
    bin_panels.append(
        precision_panel({"B12": feat_r1, "B25": feat_b25}, agreement="consistency")
    )

assert feat_r1_example is not None
print("n features (all families):", len(feat_r1_example.feature_names))
demo_precise = identify_precise_features(
    {
        "repeatability": aggregate_panels(repeat_panels),
        "reproducibility_kernel_radius": aggregate_panels(kernel_panels),
        "reproducibility_bin_width": aggregate_panels(bin_panels),
    },
    lcl_threshold=LCL_THRESHOLD,
)
lcl_names: Tuple[str, ...] = tuple(demo_precise.feature_names)
print(f"self-screen LCL survivors (n={len(cohort)}): {len(lcl_names)}")

# Spearman on the first subject's LCL columns (Prior habitat filtering rule).
lcl_present = [n for n in lcl_names if n in feat_r1_example.feature_names]
spearman_demo = PreciseCorrelationFilter(corr_threshold=0.7, p_threshold=0.05)
demo_frame = feat_r1_example.feature_frame()[lcl_present]
spearman_kept: List[str] = list(spearman_demo.fit(demo_frame)["columns"])
print(
    f"self-screen after Spearman (r>0.7, p<0.05, keep later column): "
    f"{len(spearman_kept)} of {len(lcl_present)}"
)
print(
    "CAVEAT: n=5 demo self-screen is unstable; "
    "it is NOT used for the habitat comparison below."
)

# %%
# Part 2 — Prior published precise names (habitat whitelist)
# ----------------------------------------------------------
# Fixed list from Prior et al. 2024 GitHub
# ``habitat_computation.py`` ``robust_names`` (26 ICC-precise features).
# HABIT columns use the ``original_`` prefix. Missing names are reported;
# they are not replaced with other features.
_PRIOR_ROBUST_STEMS: Tuple[str, ...] = (
    "firstorder_10Percentile",
    "firstorder_90Percentile",
    "firstorder_Energy",
    "firstorder_Mean",
    "firstorder_Median",
    "firstorder_Minimum",
    "firstorder_RootMeanSquared",
    "glcm_Autocorrelation",
    "glcm_JointAverage",
    "glcm_SumAverage",
    "gldm_DependenceEntropy",
    "gldm_GrayLevelNonUniformity",
    "gldm_HighGrayLevelEmphasis",
    "gldm_LargeDependenceLowGrayLevelEmphasis",
    "gldm_LowGrayLevelEmphasis",
    "gldm_SmallDependenceHighGrayLevelEmphasis",
    "glrlm_GrayLevelNonUniformity",
    "glrlm_HighGrayLevelRunEmphasis",
    "glrlm_LongRunHighGrayLevelEmphasis",
    "glrlm_LongRunLowGrayLevelEmphasis",
    "glrlm_LowGrayLevelRunEmphasis",
    "glrlm_RunLengthNonUniformity",
    "glrlm_RunPercentage",
    "glrlm_RunVariance",
    "glrlm_ShortRunHighGrayLevelEmphasis",
    "ngtdm_Coarseness",
)
# HABIT voxel columns are ``original_<class>_<name>-<modality>``.
PRIOR_PRECISE_FEATURES: Tuple[str, ...] = tuple(
    f"original_{stem}-{MODALITIES[0]}" for stem in _PRIOR_ROBUST_STEMS
)
available = set(feat_r1_example.feature_names)
prior_present = [n for n in PRIOR_PRECISE_FEATURES if n in available]
prior_missing = [n for n in PRIOR_PRECISE_FEATURES if n not in available]
print(f"Prior published precise features: {len(PRIOR_PRECISE_FEATURES)}")
print(f"present in HABIT extract: {len(prior_present)}")
if prior_missing:
    print("Prior features absent from extract (not substituted):", prior_missing)
if not prior_present:
    raise RuntimeError(
        "no Prior precise feature is present in the voxel extract; "
        "cannot build Prior-precise habitats"
    )
prior_whitelist = FeatureWhitelist(prior_present)

# %%
# Habitat Spec helpers (winsorize 1 % then z-score)
# -------------------------------------------------
# Clustering is subject-level only (no cohort z-score). The Prior arm
# whitelists the published precise names above — not the demo self-screen.
extract_stage = Stage(
    "extract",
    Spec(
        "voxel_radiomics",
        {"modalities": list(MODALITIES), "roi": ROI, "kernel_radius": 1},
    ),
)
scale_stages = (
    Stage("preprocess", Spec("winsorize", {"winsor_limits": [0.01, 0.01]})),
    Stage("preprocess2", Spec("zscore")),
)
fit_assign = (
    Stage("fit", Spec("kmeans", {"n_habitats": 3, "n_init": 5})),
    Stage("assign", Spec("nearest_centroid")),
    Stage("volume", Spec("volume")),
)


def make_spec(name: str, *, use_prior_precise: bool) -> HabitatSpec:
    """Build a one-step Spec with optional Prior precise whitelist.

    Args:
        name: Spec name.
        use_prior_precise: If True, keep only Prior published precise columns.

    Returns:
        HabitatSpec for ``Study.fit_predict``.
    """
    stages: List[Stage] = [extract_stage]
    if use_prior_precise:
        stages.append(Stage("whitelist", prior_whitelist.spec))
    stages.extend(scale_stages)
    stages.extend(fit_assign)
    return HabitatSpec(
        name=name, stages=tuple(stages), random_seed=11, pooling="none"
    )


backend = backend_from_policy(RunPolicy(backend="serial", on_subject_failure="continue"))


def mean_dice_and_overlays(
    *,
    tag: str,
    title_prefix: str,
    maps_orig: Sequence[HabitatMap],
    maps_pert: Sequence[HabitatMap],
    subjects_orig: Sequence[Subject],
) -> List[float]:
    """Align perturbed maps, report per-subject mean Dice, save overlays.

    Args:
        tag: Filename stem prefix (``all`` or ``prior``).
        title_prefix: Short English label in figure titles.
        maps_orig: Habitat maps on original images (one per subject).
        maps_pert: Habitat maps on perturbed images (same order).
        subjects_orig: Original subjects (anatomy for overlays).

    Returns:
        Per-subject mean Dice across habitats (Hungarian-matched).
    """
    per_subject: List[float] = []
    for map_o, map_p, subj in zip(maps_orig, maps_pert, subjects_orig):
        sid = subj.subject_id
        vol: ImageVolume = subj.image(MODALITIES[0])
        map_p_aligned = align_habitat_map(map_o, map_p, force=True)
        dice_df = habitat_stability(map_o, [map_p])
        mean_dice = float(dice_df["dice"].mean())
        per_subject.append(mean_dice)
        print(f"Dice {tag}-features {sid} (per habitat):")
        print(dice_df[["habitat_id", "matched_id", "dice"]].round(3).to_string(index=False))
        print(f"mean Dice {tag}-features {sid}={mean_dice:.3f}")

        # Same anatomy (original ImageVolume) for both overlays.
        for one_map, kind, stem_suffix in (
            (map_o, "original", "orig"),
            (map_p_aligned, "perturbed (matched)", "pert"),
        ):
            fig = plot_habitat_overlay(
                vol,
                one_map,
                title=f"{sid}: {title_prefix}, {kind}",
                crop_to="labels",
            )
            fig.savefig(
                f"out/precise_habitats_{tag}_{sid}_{stem_suffix}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.show()
    return per_subject


# %%
# Full-feature habitats: original vs perturbed (all 5 subjects)
# -------------------------------------------------------------
# Fit/assign independently per subject (pooling="none").
result_all_orig = Study(make_spec("all_original", use_prior_precise=False)).fit_predict(
    cohort, backend=backend
)
result_all_pert = Study(make_spec("all_perturbed", use_prior_precise=False)).fit_predict(
    pert_cohort, backend=backend
)
dice_all_per = mean_dice_and_overlays(
    tag="all",
    title_prefix="all features",
    maps_orig=result_all_orig.habitat_maps,
    maps_pert=result_all_pert.habitat_maps,
    subjects_orig=tuple(cohort),
)
median_all = float(np.median(dice_all_per))
ids = [s.subject_id for s in cohort]
print(
    "per-subject mean Dice all-features:",
    {sid: f"{d:.3f}" for sid, d in zip(ids, dice_all_per)},
)
print(f"median Dice all-features={median_all:.3f} (n={len(cohort)})")

# %%
# Prior-precise habitats: original vs perturbed (all 5 subjects)
# --------------------------------------------------------------
# Same K and preprocessing; columns restricted to Prior's published
# precise list (not the 5-subject self-screen).
result_pr_orig = Study(make_spec("prior_original", use_prior_precise=True)).fit_predict(
    cohort, backend=backend
)
result_pr_pert = Study(make_spec("prior_perturbed", use_prior_precise=True)).fit_predict(
    pert_cohort, backend=backend
)
dice_pr_per = mean_dice_and_overlays(
    tag="prior",
    title_prefix="Prior precise features",
    maps_orig=result_pr_orig.habitat_maps,
    maps_pert=result_pr_pert.habitat_maps,
    subjects_orig=tuple(cohort),
)
median_pr = float(np.median(dice_pr_per))
print(
    "per-subject mean Dice Prior-precise:",
    {sid: f"{d:.3f}" for sid, d in zip(ids, dice_pr_per)},
)
print(f"median Dice Prior-precise={median_pr:.3f} (n={len(cohort)})")

# Per-subject table (mean Dice per subject; cohort summary = median).
table = pd.DataFrame(
    {
        "subject_id": ids,
        "dice_all_features": dice_all_per,
        "dice_prior_precise": dice_pr_per,
    }
)
print(table.round(3).to_string(index=False))
cmp = (
    f"median Dice all={median_all:.3f}; median Dice Prior-precise={median_pr:.3f}"
)
if median_pr > median_all:
    cmp += " (Prior-precise median is higher on this demo)"
elif median_pr < median_all:
    cmp += " (Prior-precise median is not higher on this demo)"
else:
    cmp += " (medians equal on this demo)"
print(cmp)
print(
    "Habitats used Prior published precise names "
    f"({len(prior_present)}/{len(PRIOR_PRECISE_FEATURES)} present); "
    "not the n=5 self-screen whitelist."
)
