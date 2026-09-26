"""
Precise features vs all textures under perturbation
===================================================

Voxel texture features that flip under a simulated re-acquisition make
unstable habitats. Prior et al. (Radiol Artif Intell 2024;6(2):e230118)
keep features whose ICC lower confidence limit (LCL) clears a threshold
**across subjects**, not inside one patient.

This page:

1. Perturbs every demo subject with the paper Appendix S2 chain
   (Chang Gaussian noise, 0.5-voxel translation fraction, 0.5° z-rotation).
2. Screens **all** bundled voxel radiomics families on the **full demo
   cohort** (5 subjects) with ICC LCL >= 0.5 in every experiment.
3. Builds habitats on **all five** subjects: all features vs precise
   features, original vs perturbed. Subject-level winsorize 1 % then
   z-score; fixed ``n_habitats=3``. Dice is reported per subject after
   Hungarian label matching; the five-subject summary is the **median**
   of those scores (all-features 0.743, precise 0.708). The precise
   median is not higher. A short original/perturbed **image** pair uses
   one subject only so the page stays readable.

On subj003, precise-feature mean Dice falls to 0.515 versus 0.749 with
all features, while the other four subjects stay near their all-feature
scores. The LCL whitelist is estimated on only five demo subjects
(60 of 90 features pass), so the selected columns are a noisy cohort
screen. A cohort-stable subset need not maximize Dice on every case;
subj003 is one demo outlier, not a general precise-is-worse result.
"""

# %%
# Load the full demo cohort
# -------------------------
# sphinx_gallery_thumbnail_number = 4
# LCL screen and habitat maps both use every subject. Only the raw
# intensity pair below is limited to the first subject for readability.
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import (
    Cohort,
    HabitatMap,
    ImageVolume,
    Subject,
    cohort_from_directory,
)
from habit.datasets import fetch_demo
from habit.execution import backend_from_policy
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
print(f"demo cohort size for LCL screen: {len(cohort)}")
if len(cohort) < 5:
    raise RuntimeError(
        f"expected at least 5 demo subjects for cohort LCL; got {len(cohort)}"
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
# Display one subject so the page stays readable. The LCL screen and
# habitat panels below use all five subjects.
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
# Cohort LCL screen on all texture families
# -----------------------------------------
# Default ``extract_voxel_texture`` uses the bundled voxel preset (all
# supported families: firstorder, GLCM, GLRLM, GLSZM, GLDM, NGTDM).
# One panel per subject; aggregate_panels pools them before the LCL cut.
LCL_THRESHOLD = 0.5
repeat_panels = []
kernel_panels = []
bin_panels = []
for orig_subj, pert_subj in zip(cohort, pert_cohort):
    vol = orig_subj.image(MODALITIES[0])
    vol_p = pert_subj.image(MODALITIES[0])
    m = orig_subj.mask(ROI)
    # Base setting: kernel_radius=1, bin_width=12 (same as habitat extract).
    feat_r1 = extract_voxel_texture(vol, m, kernel_radius=1, bin_width=12)
    feat_r3 = extract_voxel_texture(vol, m, kernel_radius=3, bin_width=12)
    feat_b25 = extract_voxel_texture(vol, m, kernel_radius=1, bin_width=25)
    feat_pert = extract_voxel_texture(vol_p, m, kernel_radius=1, bin_width=12)
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

print("n features (all families):", len(feat_r1.feature_names))
precise = identify_precise_features(
    {
        "repeatability": aggregate_panels(repeat_panels),
        "reproducibility_kernel_radius": aggregate_panels(kernel_panels),
        "reproducibility_bin_width": aggregate_panels(bin_panels),
    },
    lcl_threshold=LCL_THRESHOLD,
)
screened = tuple(precise.feature_names)
print(f"n subjects in LCL screen: {len(cohort)}")
print("n precise:", len(screened))
print("precise names:", list(screened)[:12], "..." if len(screened) > 12 else "")
if not screened:
    raise RuntimeError("no feature passed every ICC experiment; cannot build precise Spec")

# %%
# Habitat Spec helpers (winsorize 1 % then z-score)
# -------------------------------------------------
# Whitelist uses the cohort-level PreciseFeatureSet from above.
# Clustering is subject-level only (no cohort z-score).
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


def make_spec(name: str, *, whitelist: bool) -> HabitatSpec:
    """Build a one-step Spec with optional precise whitelist.

    Args:
        name: Spec name.
        whitelist: If True, keep only precise feature columns.

    Returns:
        HabitatSpec for ``Study.fit_predict``.
    """
    stages = [extract_stage]
    if whitelist:
        stages.append(Stage("whitelist", precise.preprocessor().spec))
    stages.extend(scale_stages)
    stages.extend(fit_assign)
    return HabitatSpec(name=name, stages=tuple(stages), random_seed=11, pooling="none")


backend = backend_from_policy(RunPolicy(backend="serial", on_subject_failure="continue"))


def mean_dice_and_overlays(
    *,
    tag: str,
    title_prefix: str,
    maps_orig: Tuple[HabitatMap, ...],
    maps_pert: Tuple[HabitatMap, ...],
    subjects_orig: Tuple[Subject, ...],
) -> List[float]:
    """Align perturbed maps, report mean Dice, and save overlay pairs.

    Args:
        tag: Filename stem prefix (``all`` or ``precise``).
        title_prefix: Short English label in figure titles.
        maps_orig: Habitat maps on original images (one per subject).
        maps_pert: Habitat maps on perturbed images (same order).
        subjects_orig: Original subjects (anatomy for overlays).

    Returns:
        Per-subject mean Dice (Hungarian-matched habitats).
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
# Fit/assign independently per subject (pooling="none"). Overlay every
# subject so the page matches the cohort used for the LCL screen.
result_all_orig = Study(make_spec("all_original", whitelist=False)).fit_predict(
    cohort, backend=backend
)
result_all_pert = Study(make_spec("all_perturbed", whitelist=False)).fit_predict(
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
print(
    "per-subject mean Dice all-features:",
    {sid: f"{d:.3f}" for sid, d in zip([s.subject_id for s in cohort], dice_all_per)},
)
print(f"median Dice all-features={median_all:.3f} (n={len(cohort)})")

# %%
# Precise-feature habitats: original vs perturbed (all 5 subjects)
# ----------------------------------------------------------------
# Same K and preprocessing; columns restricted to the cohort whitelist.
result_prec_orig = Study(make_spec("precise_original", whitelist=True)).fit_predict(
    cohort, backend=backend
)
result_prec_pert = Study(make_spec("precise_perturbed", whitelist=True)).fit_predict(
    pert_cohort, backend=backend
)
dice_pr_per = mean_dice_and_overlays(
    tag="precise",
    title_prefix="precise features",
    maps_orig=result_prec_orig.habitat_maps,
    maps_pert=result_prec_pert.habitat_maps,
    subjects_orig=tuple(cohort),
)
median_pr = float(np.median(dice_pr_per))
print(
    "per-subject mean Dice precise-features:",
    {sid: f"{d:.3f}" for sid, d in zip([s.subject_id for s in cohort], dice_pr_per)},
)
print(f"median Dice precise-features={median_pr:.3f} (n={len(cohort)})")
print(
    f"median Dice all={median_all:.3f}; median Dice precise={median_pr:.3f} "
    f"(screen and maps on n={len(cohort)} subjects; precise median is not higher)"
)
