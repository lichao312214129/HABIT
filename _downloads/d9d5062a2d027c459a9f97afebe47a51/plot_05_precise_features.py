"""
Precise features vs all textures under perturbation
===================================================

Voxel texture features that flip under a simulated re-acquisition make
unstable habitats. Prior et al. (Radiol Artif Intell 2024;6(2):e230118)
keep features whose ICC lower confidence limit clears a threshold.

This page extracts the **full** bundled voxel radiomics families
(firstorder, GLCM, GLRLM, GLSZM, GLDM, NGTDM), screens them, and
compares four habitat maps:

1. all features on the original image
2. all features on the perturbed image
3. precise-screened features on the original image
4. precise-screened features on the perturbed image

Subject-level winsorize 1 % then z-score before clustering. Fixed
``n_habitats=3``. Dice is reported after Hungarian label matching; do
not treat a small gain as a large one.
"""

# %%
# Load one subject
# ----------------
# sphinx_gallery_thumbnail_number = 4
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import Cohort, Subject, cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution import backend_from_policy
from habit.precision import (
    aggregate_panels,
    align_habitat_map,
    habitat_stability,
    identify_precise_features,
    perturb_image,
    precision_panel,
)
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.spec.policy import RunPolicy
from habit.viz import plot_habitat_label_compare, plot_habitat_overlay, plot_intensity_slice
from habit.voxel_features import extract_voxel_texture

# Change DATA / MODALITIES / ROI to your preprocessed layout.
DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
subject = cohort[0]
image = subject.image(MODALITIES[0])
mask = subject.mask(ROI)
Path("out").mkdir(exist_ok=True)

# %%
# Original and perturbed images
# -----------------------------
# Stronger than mild Appendix S2 defaults so full-feature maps can
# disagree visibly. Mask unchanged; intensity volume is perturbed.
rng = np.random.default_rng(7)
sigma = float(np.nanstd(image.data)) * 0.20
perturbed = perturb_image(image, method="gaussian_noise", sigma=sigma, rng=rng)
perturbed = perturb_image(
    perturbed, method="translation", shift_voxels=(3.0, 2.0, 1.5), rng=rng
)
perturbed = perturb_image(perturbed, method="rotation", angle_degrees=8.0, rng=rng)
print(f"perturbation: noise sigma={sigma:.4g}, translation (3,2,1.5), rotation 8 deg")

for vol, name, stem in (
    (image, "original", "precise_image_original"),
    (perturbed, "perturbed", "precise_image_perturbed"),
):
    fig = plot_intensity_slice(
        vol,
        roi_mask=mask,
        roi_contour=True,
        image_label=MODALITIES[0],
        title=f"{subject.subject_id}: {name} image",
    )
    fig.savefig(f"out/{stem}.png", dpi=150, bbox_inches="tight")
    plt.show()

# %%
# Screen all bundled voxel features
# ---------------------------------
# Default ``extract_voxel_texture`` uses the bundled voxel preset (all
# supported texture families). Whitelist = ICC LCL >= 0.5 in every
# experiment below.
# Bundled voxel preset: all supported texture families (binWidth overridden).
feat_r1 = extract_voxel_texture(image, mask, kernel_radius=1, bin_width=12)
feat_r3 = extract_voxel_texture(image, mask, kernel_radius=3, bin_width=12)
feat_b25 = extract_voxel_texture(image, mask, kernel_radius=1, bin_width=25)
feat_pert = extract_voxel_texture(perturbed, mask, kernel_radius=1, bin_width=12)
print("n features (all families):", len(feat_r1.feature_names))

LCL_THRESHOLD = 0.5
precise = identify_precise_features(
    {
        "repeatability": aggregate_panels(
            [precision_panel({"original": feat_r1, "perturbed": feat_pert}, agreement="absolute")]
        ),
        "reproducibility_kernel_radius": aggregate_panels(
            [precision_panel({"R1": feat_r1, "R3": feat_r3}, agreement="consistency")]
        ),
        "reproducibility_bin_width": aggregate_panels(
            [precision_panel({"B12": feat_r1, "B25": feat_b25}, agreement="consistency")]
        ),
    },
    lcl_threshold=LCL_THRESHOLD,
)
screened = tuple(precise.feature_names)
print("n precise:", len(screened))
print("precise names:", list(screened)[:12], "..." if len(screened) > 12 else "")

# %%
# Habitats: all features vs precise, original vs perturbed
# --------------------------------------------------------
# Same extract Spec for every run. Precise runs insert the whitelist
# before subject-level scale. Fixed K keeps the page about features.
# Omit params -> bundled voxel preset (same families as extract_voxel_texture).
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


def with_image(base: Subject, new_image) -> Subject:
    """Copy ``base`` with a replacement intensity volume.

    Args:
        base: Original subject.
        new_image: ImageVolume on the same grid.

    Returns:
        New Subject with unchanged masks.
    """
    images = dict(base.images)
    images[MODALITIES[0]] = new_image
    return Subject(subject_id=base.subject_id, images=images, masks=dict(base.masks))


backend = backend_from_policy(RunPolicy(backend="serial", on_subject_failure="continue"))
pert_cohort = Cohort(subjects=(with_image(subject, perturbed),))

if not screened:
    raise RuntimeError("no feature passed every ICC experiment; cannot build precise Spec")

result_all_orig = Study(make_spec("all_original", whitelist=False)).fit_predict(
    cohort, backend=backend
)
result_all_pert = Study(make_spec("all_perturbed", whitelist=False)).fit_predict(
    pert_cohort, backend=backend
)
result_prec_orig = Study(make_spec("precise_original", whitelist=True)).fit_predict(
    cohort, backend=backend
)
result_prec_pert = Study(make_spec("precise_perturbed", whitelist=True)).fit_predict(
    pert_cohort, backend=backend
)

map_all_o = result_all_orig.habitat_maps[0]
map_all_p = result_all_pert.habitat_maps[0]
map_pr_o = result_prec_orig.habitat_maps[0]
map_pr_p = result_prec_pert.habitat_maps[0]

# Match perturbed labels onto the matching original map before Dice.
map_all_p_aligned = align_habitat_map(map_all_o, map_all_p, force=True)
map_pr_p_aligned = align_habitat_map(map_pr_o, map_pr_p, force=True)
dice_all = habitat_stability(map_all_o, [map_all_p])
dice_pr = habitat_stability(map_pr_o, [map_pr_p])
mean_all = float(dice_all["dice"].mean())
mean_pr = float(dice_pr["dice"].mean())
print("Dice all-features (per habitat):")
print(dice_all[["habitat_id", "matched_id", "dice"]].round(3).to_string(index=False))
print("Dice precise-features (per habitat):")
print(dice_pr[["habitat_id", "matched_id", "dice"]].round(3).to_string(index=False))
print(f"mean Dice all={mean_all:.3f}; mean Dice precise={mean_pr:.3f}")

fig = plot_habitat_label_compare(
    image,
    map_all_o,
    map_all_p_aligned,
    titles=("all features, original", "all features, perturbed (matched)"),
    crop_to="labels",
)
fig.savefig("out/precise_habitats_all_orig_vs_pert.png", dpi=150, bbox_inches="tight")
plt.show()

fig = plot_habitat_label_compare(
    image,
    map_pr_o,
    map_pr_p_aligned,
    titles=("precise features, original", "precise features, perturbed (matched)"),
    crop_to="labels",
)
fig.savefig("out/precise_habitats_precise_orig_vs_pert.png", dpi=150, bbox_inches="tight")
plt.show()

fig = plot_habitat_overlay(
    image,
    map_pr_o,
    title=f"{subject.subject_id}: precise habitats (original)",
    crop_to="labels",
)
fig.savefig("out/precise_habitats_precise_original.png", dpi=150, bbox_inches="tight")
plt.show()
