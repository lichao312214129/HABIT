"""
Precise features vs all textures under perturbation
===================================================

Voxel texture features that flip under a simulated re-acquisition make
unstable habitats. Prior et al. (Radiol Artif Intell 2024;6(2):e230118)
screen features in two stages:

1. **ICC LCL** across a large cohort (Table 3 liver: ICC LCL >= 0.50,
   plus Coarseness) — a published list of **26 names before Spearman**.
2. **Per-lesion Spearman** immediately before habitat computation:
   drop highly correlated features at signed r > 0.7 and P < .001
   (``select_precise_correlation_columns`` /
   ``precise_correlation_filter``; keep the later column). Surviving
   columns differ by lesion; there is no published global post-Spearman
   list of 26.

This page has two separate parts:

1. **Self-screen demo only** (5 demo subjects): ICC LCL then Spearman
   with the library default ``p_threshold=0.001``. ``n=5`` is too small,
   so that whitelist is unstable and is **not** used for habitat maps.
2. **Habitat comparison** (all 5 subjects): original vs Appendix S2
   perturbation, two arms that both apply **per-subject Spearman** after
   winsorize 1 % + z-score (no cohort z-score), then cluster:

   - all extracted textures → per-subject Spearman → habitats
   - Table 3 precise 26 → per-subject Spearman → habitats

   Fixed ``n_habitats=3``. Dice is per subject after Hungarian matching;
   the five-subject summary is the **median** (not the mean).

Intensity overlays use one subject only so the page stays readable.
"""

# %%
# Load the full demo cohort
# -------------------------
# sphinx_gallery_thumbnail_number = 4
# Habitats and the self-screen demo both use every subject. Only the raw
# intensity pair below is limited to the first subject for readability.
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from habit.recipes.result import StudyResult
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
# bin width). Then Prior-style Spearman on one subject's LCL survivors:
# signed r > 0.7 and p < 0.001 (HABIT default) drops the *earlier*
# column and keeps the later one
# (``PreciseCorrelationFilter`` / ``select_precise_correlation_columns``).
# Caveat: n=5 is too small; do NOT use this whitelist for habitats below.
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

# %%
# Self-screen Spearman (demo only; default p=0.001)
# -------------------------------------------------
# Library default matches Prior paper P < .001. This whitelist is NOT
# used to draw habitats below.
lcl_present = [n for n in lcl_names if n in feat_r1_example.feature_names]
# Default p_threshold=0.001 (paper); do not pass 0.05 here.
spearman_demo = PreciseCorrelationFilter(corr_threshold=0.7)
demo_frame = feat_r1_example.feature_frame()[lcl_present]
spearman_kept: List[str] = list(spearman_demo.fit(demo_frame)["columns"])
print(
    f"self-screen after Spearman (r>0.7, p<0.001, keep later column): "
    f"{len(spearman_kept)} of {len(lcl_present)}"
)
print(
    "CAVEAT: n=5 demo self-screen is unstable; "
    "it is NOT used for the habitat comparison below."
)

# %%
# Part 2 — Table 3 precise parent set (ICC LCL list, before Spearman)
# -------------------------------------------------------------------
# Fixed list from Prior et al. 2024 Table 3 liver (ICC LCL >= 0.50)
# plus Coarseness — 26 names **before** per-lesion Spearman. Same stems
# appear as ``robust_names`` in radiomicsgroup/precise-habitats
# ``habitat_computation.py``. HABIT columns use the ``original_`` prefix.
# Missing names are reported; they are not replaced with other features.
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
print(
    f"Table 3 / robust_names (ICC LCL parent set, before Spearman): "
    f"{len(PRIOR_PRECISE_FEATURES)}"
)
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
# Habitat Spec helpers (winsorize → z-score → per-lesion Spearman)
# ----------------------------------------------------------------
# Clustering is subject-level only (no cohort z-score). Both arms run
# Spearman on the **scaled** matrix that enters clustering (after
# winsorize 1 % and per-feature z-score). Do not cluster the raw 26
# without this per-lesion step; do not use the n=5 self-screen whitelist.
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
# Default p_threshold=0.001 (paper); corr_threshold=0.7 (signed r > 0.7).
spearman_stage = Stage("spearman", PreciseCorrelationFilter(corr_threshold=0.7).spec)
fit_assign = (
    Stage("fit", Spec("kmeans", {"n_habitats": 3, "n_init": 5})),
    Stage("assign", Spec("nearest_centroid")),
    Stage("volume", Spec("volume")),
)


def make_spec(name: str, *, use_table3_precise: bool) -> HabitatSpec:
    """Build a one-step Spec with optional Table-3 whitelist + Spearman.

    Args:
        name: Spec name.
        use_table3_precise: If True, start from Table 3 ICC-LCL columns
            before the per-lesion Spearman stage.

    Returns:
        HabitatSpec for ``Study.fit_predict`` (pooling=none).
    """
    stages: List[Stage] = [extract_stage]
    if use_table3_precise:
        stages.append(Stage("whitelist", prior_whitelist.spec))
    stages.extend(scale_stages)
    # Spearman after scaling: filter the same matrix k-means will see.
    stages.append(spearman_stage)
    stages.extend(fit_assign)
    return HabitatSpec(
        name=name, stages=tuple(stages), random_seed=11, pooling="none"
    )


backend = backend_from_policy(RunPolicy(backend="serial", on_subject_failure="continue"))


def kept_feature_counts(study_result: StudyResult) -> Dict[str, int]:
    """Read post-Spearman feature counts from per-subject habitat models.

    Args:
        study_result: ``Study.fit_predict`` result (``pooling="none"``).

    Returns:
        ``subject_id -> n_features`` that entered k-means after Spearman.
    """
    counts: Dict[str, int] = {}
    for sid, model in study_result.subject_models.items():
        counts[str(sid)] = len(model.feature_names)
    return counts


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
        tag: Filename stem prefix (``all`` or ``table3``).
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
# All-features arm: extract → scale → per-subject Spearman → habitats
# -------------------------------------------------------------------
# Parent set = every extracted texture. Spearman is per subject on that
# subject's own voxel rows (pooling="none"). Counts below are after
# Spearman — not a universal feature count.
result_all_orig = Study(make_spec("all_original", use_table3_precise=False)).fit_predict(
    cohort, backend=backend
)
result_all_pert = Study(make_spec("all_perturbed", use_table3_precise=False)).fit_predict(
    pert_cohort, backend=backend
)

# %%
# Per-subject Spearman kept-column counts (all-features arm)
# ----------------------------------------------------------
# Different subjects may keep different columns (paper: per-lesion step).
counts_all = kept_feature_counts(result_all_orig)
print("all-features arm: columns after per-subject Spearman (original images):")
for sid in [s.subject_id for s in cohort]:
    print(f"  {sid}: {counts_all.get(sid, 'missing')}")
print(
    "Note: these are post-Spearman counts, not a fixed global list. "
    "Do not read them as '26 after Spearman'."
)

# %%
# All-features habitats: original vs perturbed (all 5 subjects)
# -------------------------------------------------------------
dice_all_per = mean_dice_and_overlays(
    tag="all",
    title_prefix="all features + Spearman",
    maps_orig=result_all_orig.habitat_maps,
    maps_pert=result_all_pert.habitat_maps,
    subjects_orig=tuple(cohort),
)
median_all = float(np.median(dice_all_per))
ids = [s.subject_id for s in cohort]
print(
    "per-subject mean Dice all-features+Spearman:",
    {sid: f"{d:.3f}" for sid, d in zip(ids, dice_all_per)},
)
print(f"median Dice all-features+Spearman={median_all:.3f} (n={len(cohort)})")

# %%
# Table-3 arm: whitelist 26 → scale → per-subject Spearman → habitats
# -------------------------------------------------------------------
# Parent set = Table 3 ICC-LCL list (before Spearman). Still run
# per-lesion Spearman before clustering — do not cluster the raw 26.
result_pr_orig = Study(
    make_spec("table3_original", use_table3_precise=True)
).fit_predict(cohort, backend=backend)
result_pr_pert = Study(
    make_spec("table3_perturbed", use_table3_precise=True)
).fit_predict(pert_cohort, backend=backend)

# %%
# Per-subject Spearman kept-column counts (Table-3 arm)
# -----------------------------------------------------
counts_pr = kept_feature_counts(result_pr_orig)
print(
    "Table-3 arm: columns after per-subject Spearman "
    f"(parent set {len(prior_present)} ICC-LCL names):"
)
for sid in ids:
    print(f"  {sid}: {counts_pr.get(sid, 'missing')}")
print(
    "Post-Spearman counts are per lesion; they are not a universal 26."
)

# %%
# Table-3 + Spearman habitats: original vs perturbed (all 5 subjects)
# -------------------------------------------------------------------
dice_pr_per = mean_dice_and_overlays(
    tag="table3",
    title_prefix="Table3 precise + Spearman",
    maps_orig=result_pr_orig.habitat_maps,
    maps_pert=result_pr_pert.habitat_maps,
    subjects_orig=tuple(cohort),
)
median_pr = float(np.median(dice_pr_per))
print(
    "per-subject mean Dice Table3+Spearman:",
    {sid: f"{d:.3f}" for sid, d in zip(ids, dice_pr_per)},
)
print(f"median Dice Table3+Spearman={median_pr:.3f} (n={len(cohort)})")

# %%
# Dice summary (median across subjects; no overclaim)
# ---------------------------------------------------
table = pd.DataFrame(
    {
        "subject_id": ids,
        "n_kept_all_spearman": [counts_all.get(sid) for sid in ids],
        "n_kept_table3_spearman": [counts_pr.get(sid) for sid in ids],
        "dice_all_spearman": dice_all_per,
        "dice_table3_spearman": dice_pr_per,
    }
)
print(table.round(3).to_string(index=False))
cmp = (
    f"median Dice all+Spearman={median_all:.3f}; "
    f"median Dice Table3+Spearman={median_pr:.3f}"
)
if median_pr > median_all:
    cmp += " (Table3+Spearman median is higher on this demo)"
elif median_pr < median_all:
    cmp += " (Table3+Spearman median is not higher on this demo)"
else:
    cmp += " (medians equal on this demo)"
print(cmp)
print(
    "Habitats used Table 3 ICC-LCL names as the parent set "
    f"({len(prior_present)}/{len(PRIOR_PRECISE_FEATURES)} present), "
    "then per-subject Spearman (p<0.001); not the n=5 self-screen whitelist."
)
