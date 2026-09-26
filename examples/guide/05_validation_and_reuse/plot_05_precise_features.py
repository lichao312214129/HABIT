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
3. Builds habitats on one example subject: all features vs precise
   features, original vs perturbed. Subject-level winsorize 1 % then
   z-score; fixed ``n_habitats=3``. Dice is reported after Hungarian
   label matching; do not treat a small gain as a large one.
"""

# %%
# Load the full demo cohort
# -------------------------
# sphinx_gallery_thumbnail_number = 4
# The LCL screen needs every subject. Habitat figures below use only
# the first subject after the whitelist is fixed on all five.
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
    prior2024_retest_perturbation,
    precision_panel,
)
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.spec.policy import RunPolicy
from habit.viz import plot_habitat_label_compare, plot_intensity_slice
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
# Display one subject so the page stays readable. The LCL screen still
# uses all five subjects in the next cell.
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
example_cohort = Cohort(subjects=(example,))
example_pert_cohort = Cohort(subjects=(example_pert,))

# %%
# Full-feature habitats: original vs perturbed
# --------------------------------------------
# One example subject only; the whitelist above was fit on all five.
result_all_orig = Study(make_spec("all_original", whitelist=False)).fit_predict(
    example_cohort, backend=backend
)
result_all_pert = Study(make_spec("all_perturbed", whitelist=False)).fit_predict(
    example_pert_cohort, backend=backend
)
map_all_o = result_all_orig.habitat_maps[0]
map_all_p = result_all_pert.habitat_maps[0]
map_all_p_aligned = align_habitat_map(map_all_o, map_all_p, force=True)
dice_all = habitat_stability(map_all_o, [map_all_p])
mean_all = float(dice_all["dice"].mean())
print("Dice all-features (per habitat):")
print(dice_all[["habitat_id", "matched_id", "dice"]].round(3).to_string(index=False))
print(f"mean Dice all-features={mean_all:.3f}")

fig = plot_habitat_label_compare(
    image,
    map_all_o,
    map_all_p_aligned,
    titles=("all features, original", "all features, perturbed (matched)"),
    crop_to="labels",
)
fig.savefig("out/precise_habitats_all_orig_vs_pert.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Precise-feature habitats: original vs perturbed
# -----------------------------------------------
# Same example subject and K; columns restricted to the cohort whitelist.
result_prec_orig = Study(make_spec("precise_original", whitelist=True)).fit_predict(
    example_cohort, backend=backend
)
result_prec_pert = Study(make_spec("precise_perturbed", whitelist=True)).fit_predict(
    example_pert_cohort, backend=backend
)
map_pr_o = result_prec_orig.habitat_maps[0]
map_pr_p = result_prec_pert.habitat_maps[0]
map_pr_p_aligned = align_habitat_map(map_pr_o, map_pr_p, force=True)
dice_pr = habitat_stability(map_pr_o, [map_pr_p])
mean_pr = float(dice_pr["dice"].mean())
print("Dice precise-features (per habitat):")
print(dice_pr[["habitat_id", "matched_id", "dice"]].round(3).to_string(index=False))
print(f"mean Dice precise-features={mean_pr:.3f}")
print(
    f"mean Dice all={mean_all:.3f}; mean Dice precise={mean_pr:.3f} "
    f"(screen on n={len(cohort)} subjects; maps for {example.subject_id})"
)

fig = plot_habitat_label_compare(
    image,
    map_pr_o,
    map_pr_p_aligned,
    titles=("precise features, original", "precise features, perturbed (matched)"),
    crop_to="labels",
)
fig.savefig("out/precise_habitats_precise_orig_vs_pert.png", dpi=150, bbox_inches="tight")
plt.show()
