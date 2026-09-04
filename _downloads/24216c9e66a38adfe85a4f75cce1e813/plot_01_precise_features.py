"""
Precise voxel features
======================

Decide **which voxel features may define habitats**, then cluster only
those robust features. This is the Prior et al. precision screen (*Radiol Artif Intell*
2024;6(2):e230118; `DOI <https://doi.org/10.1148/ryai.230118>`__).

Evaluating stability under perturbation
---------------------------------------
The core scientific value of Precise features is **stability under perturbation**:
when image acquisition has minor variations (simulated retest: noise, shift, rotation),
habitats defined on all texture features may undergo unpredictable partition shifts.
Filtering features through repeatability and reproducibility ICC panels yields
a robust whitelist that produces significantly more stable habitat maps (higher Dice
and ARI between original and perturbed scans).

In addition, an elastic ROI edge perturbation (MONAI ``bspline_deform``) is demonstrated
to inspect contour and anatomy deformations.
"""

# sphinx_gallery_thumbnail_number = 2

# %%
# Load one demo subject. ``extract_voxel_texture`` crops to the ROI box
# internally (``crop_to_roi=True``).
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from habit.contracts import Cohort, Subject, cohort_from_directory
from habit.datasets import fetch_demo
from habit.kernels.habitat_label_match import (
    adjusted_rand_index,
    habitat_dice_from_mapping,
    match_labels_by_overlap,
    present_habitat_ids,
    remap_label_array,
)
from habit.kernels.image_perturbation import binary_mask_dice
from habit.precision import (
    ImagePerturbationRegistry,
    aggregate_panels,
    identify_precise_features,
    perturb_image,
    precision_panel,
)
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec
from habit.voxel_features import extract_voxel_texture
from habit.viz import plot_habitat_label_compare, plot_intensity_slice, plot_precision_icc
from habit.viz import use_style
from habit.viz.labels import sanitize_label

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
subject = cohort[0]
image = subject.image(MODALITIES[0])
mask = subject.mask(ROI)
Path("out").mkdir(exist_ok=True)
print(f"Grid shape: {image.data.shape}")

# %%
# Appendix S2 retest chain on one shared RNG.
# Sequentially applies Gaussian noise -> translation -> rotation.
retest_rng = np.random.default_rng(7)
noisy = perturb_image(image, method="gaussian_noise", rng=retest_rng)
shifted = perturb_image(
    noisy, method="translation", shift_fraction=0.5, rng=retest_rng
)
perturbed = perturb_image(
    shifted, method="rotation", angle_degrees=0.5, rng=retest_rng
)
print("Appendix S2: gaussian_noise -> translation -> rotation")
fig_s2 = plot_intensity_slice(
    perturbed,
    before=image,
    roi_mask=mask,
    roi_contour=True,
    title="Appendix S2 chain (original vs perturbed)",
    before_label="Original",
    image_label="+ noise / shift / rotation",
)
fig_s2.savefig("out/precise_features_perturb_methods.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Extract texture features at base R3/B12 and the two reproducibility contrasts.
FEATURE_CLASSES: Dict[str, Tuple[str, ...]] = {
    "firstorder": ("Entropy", "Mean", "Variance", "Skewness", "Kurtosis"),
    "glcm": (
        "Contrast",
        "Correlation",
        "JointEntropy",
        "Idm",
        "DifferenceEntropy",
    ),
}
feat_r1 = extract_voxel_texture(
    image, mask, kernel_radius=1, bin_width=12, feature_classes=FEATURE_CLASSES
)
feat_r3 = extract_voxel_texture(
    image, mask, kernel_radius=3, bin_width=12, feature_classes=FEATURE_CLASSES
)
feat_b25 = extract_voxel_texture(
    image, mask, kernel_radius=3, bin_width=25, feature_classes=FEATURE_CLASSES
)
feat_pert = extract_voxel_texture(
    perturbed, mask, kernel_radius=3, bin_width=12, feature_classes=FEATURE_CLASSES
)
print(f"Texture features ({len(feat_r3.feature_names)}): {list(feat_r3.feature_names)}")
feat_r3.feature_frame().head()

# %%
# Precise screening = Lower Confidence Limit (LCL) >= 0.5 across all 3 ICC experiments.
precise = identify_precise_features(
    {
        "repeatability": aggregate_panels(
            [precision_panel({"original": feat_r3, "perturbed": feat_pert}, agreement="absolute")]
        ),
        "reproducibility_kernel_radius": aggregate_panels(
            [precision_panel({"R1": feat_r1, "R3": feat_r3}, agreement="consistency")]
        ),
        "reproducibility_bin_width": aggregate_panels(
            [precision_panel({"B12": feat_r3, "B25": feat_b25}, agreement="consistency")]
        ),
    },
    lcl_threshold=0.5,
)
evidence = precise.to_frame().round(3)
kept: List[str] = list(precise.feature_names)
dropped = [n for n in feat_r3.feature_names if n not in set(kept)]
print(f"Kept features ({len(kept)}): {kept}")
print(f"Dropped features ({len(dropped)}): {dropped}")
evidence

# %%
# Plot one ICC forest per experiment to inspect lower confidence limits.
for experiment, fname, title in (
    ("repeatability", "precise_features_icc_lcl.png", "Repeatability ICC"),
    (
        "reproducibility_kernel_radius",
        "precise_features_icc_kernel.png",
        "Kernel-radius reproducibility ICC",
    ),
    (
        "reproducibility_bin_width",
        "precise_features_icc_bin.png",
        "Bin-width reproducibility ICC",
    ),
):
    panel = evidence.loc[evidence["experiment"] == experiment].drop(
        columns=["precise"], errors="ignore"
    )
    fig_icc = plot_precision_icc(
        panel.dropna(subset=["value", "lcl", "ucl"]),
        lcl_threshold=precise.lcl_threshold,
        title=title,
        orientation="row",
    )
    fig_icc.savefig(f"out/{fname}", dpi=150, bbox_inches="tight")
    plt.show()

# %%
# Cluster habitats using all texture features vs precise whitelist.
# Compare the stability of habitats between the original image and perturbed image.
texture_params = {
    "imageType": {"Original": {}},
    "featureClass": {k: list(v) for k, v in FEATURE_CLASSES.items()},
    "setting": {"binWidth": 12.0, "normalize": False},
}
extractor_spec = Spec(
    "voxel_radiomics",
    {"modalities": list(MODALITIES), "kernel_radius": 3, "params": texture_params},
)
fitter_spec = Spec(
    "kmeans",
    {"n_habitats": 3, "n_init": 3},
)
minmax_spec = Spec("minmax", {"across_features": False})
subject_pert = Subject(
    subject_id=subject.subject_id,
    images={MODALITIES[0]: perturbed},
    masks=subject.masks,
)
demo = Cohort(subjects=(subject,))
demo_pert = Cohort(subjects=(subject_pert,))

# --- Experiment A: All texture features under perturbation ---
spec_all = HabitatSpec(
    name="all_texture_one_step",
    voxel_feature_extractor=extractor_spec,
    voxel_feature_preprocessors=(minmax_spec,),
    habitat_model_fitter=fitter_spec,
    habitat_assigner=Spec("nearest_centroid"),
    random_seed=11,
    pooling="none",
)
result_all_orig = Study(spec_all).fit_predict(demo)
result_all_pert = Study(spec_all).fit_predict(demo_pert)

ref_all = np.asarray(result_all_orig.habitat_maps[0].label_array)
mov_all = np.asarray(result_all_pert.habitat_maps[0].label_array)
map_all = match_labels_by_overlap(ref_all, mov_all)
aligned_all = remap_label_array(
    mov_all, map_all, reserved_ids=[int(v) for v in present_habitat_ids(ref_all)]
)
dice_all = [float(d) for _, _, d, _, _ in habitat_dice_from_mapping(ref_all, mov_all, map_all)]
mean_dice_all = float(np.mean(dice_all)) if dice_all else float("nan")
ari_all = float(adjusted_rand_index(ref_all, mov_all))
print(f"All texture features under perturbation: mean Dice={mean_dice_all:.3f}, ARI={ari_all:.3f}")

# Label comparison for all texture features: Original scan vs Perturbed scan
fig_cmp_all = plot_habitat_label_compare(
    image,
    result_all_orig.habitat_maps[0],
    aligned_all,
    titles=(
        "All features: original image",
        f"All features: perturbed image (Dice={mean_dice_all:.3f})",
    ),
    align_labels=False,
)
fig_cmp_all.savefig("out/precise_features_all_orig_vs_pert.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# --- Experiment B: Precise whitelist under perturbation ---
if not kept:
    print("No feature passed every experiment; skip precise habitats")
else:
    whitelist = precise.preprocessor()
    spec_precise = HabitatSpec(
        name="precise_one_step",
        voxel_feature_extractor=extractor_spec,
        voxel_feature_preprocessors=(whitelist.spec, minmax_spec),
        habitat_model_fitter=fitter_spec,
        habitat_assigner=Spec("nearest_centroid"),
        random_seed=11,
        pooling="none",
    )
    result_precise_orig = Study(spec_precise).fit_predict(demo)
    result_precise_pert = Study(spec_precise).fit_predict(demo_pert)

    ref_p = np.asarray(result_precise_orig.habitat_maps[0].label_array)
    mov_p = np.asarray(result_precise_pert.habitat_maps[0].label_array)
    map_p = match_labels_by_overlap(ref_p, mov_p)
    aligned_p = remap_label_array(
        mov_p, map_p, reserved_ids=[int(v) for v in present_habitat_ids(ref_p)]
    )
    dice_p = [float(d) for _, _, d, _, _ in habitat_dice_from_mapping(ref_p, mov_p, map_p)]
    mean_dice_p = float(np.mean(dice_p)) if dice_p else float("nan")
    ari_p = float(adjusted_rand_index(ref_p, mov_p))
    print(f"Precise whitelist under perturbation: mean Dice={mean_dice_p:.3f}, ARI={ari_p:.3f}")

    # Label comparison for precise features: Original scan vs Perturbed scan
    fig_cmp_p = plot_habitat_label_compare(
        image,
        result_precise_orig.habitat_maps[0],
        aligned_p,
        titles=(
            "Precise whitelist: original image",
            f"Precise whitelist: perturbed image (Dice={mean_dice_p:.3f})",
        ),
        align_labels=False,
    )
    fig_cmp_p.savefig(
        "out/precise_features_precise_orig_vs_pert.png", dpi=150, bbox_inches="tight"
    )
    plt.show()

    # Quantitative stability summary comparison table and bar chart
    stability = pd.DataFrame(
        [
            {"feature_set": "All texture features", "mean_dice": mean_dice_all, "ari": ari_all},
            {"feature_set": "Precise whitelist only", "mean_dice": mean_dice_p, "ari": ari_p},
        ]
    )
    print("Stability under perturbation (Original vs Perturbed):")
    print(stability.to_string(index=False))

    with use_style("radiology"):
        fig_stab, ax_s = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)
        x_indices = np.arange(2)
        bar_width = 0.35
        dices = [mean_dice_all, mean_dice_p]
        aris = [ari_all, ari_p]
        ax_s.bar(x_indices - bar_width / 2, dices, bar_width, label="Mean Dice", color="#0072B2")
        ax_s.bar(x_indices + bar_width / 2, aris, bar_width, label="Adjusted Rand Index", color="#E69F00")
        ax_s.set_xticks(x_indices)
        ax_s.set_xticklabels(["All features", "Precise only"])
        ax_s.set_ylim(0.0, 1.05)
        ax_s.set_ylabel("Stability score")
        ax_s.set_title(sanitize_label("Habitat stability under image perturbation"))
        ax_s.legend(loc="lower right", frameon=True)
    fig_stab.savefig("out/precise_features_stability_bar.png", dpi=150, bbox_inches="tight")
    plt.show()
    stability

# %%
# MONAI elastic / B-spline deformation of image and ROI mask.
# A realistic displacement field (magnitude_range=(35.0, 50.0) voxels) models
# anatomical and contour variation across repeat acquisitions or observer differences.
deform = ImagePerturbationRegistry.create(
    "bspline_deform",
    sigma_range=(2.0, 4.0),
    magnitude_range=(35.0, 50.0),
)
warped = deform(subject, rng=np.random.default_rng(0))
image_w = warped.image(MODALITIES[0])
mask_w = warped.mask(ROI)
ref_bin = np.asarray(mask.data) > 0
mov_bin = np.asarray(mask_w.data) > 0
n_inter = int(np.count_nonzero(ref_bin & mov_bin))
n_union = int(np.count_nonzero(ref_bin | mov_bin))
overlap = pd.DataFrame(
    [
        {
            "metric": "dice",
            "value": binary_mask_dice(ref_bin, mov_bin),
        },
        {
            "metric": "jaccard",
            "value": float(n_inter / n_union) if n_union else float("nan"),
        },
        {
            "metric": "intersection_voxels",
            "value": float(n_inter),
        },
        {
            "metric": "union_voxels",
            "value": float(n_union),
        },
    ]
)
print("MONAI bspline_deform ROI overlap metrics:")
print(overlap.round(4).to_string(index=False))
overlap

# %%
# Anatomy slice before and after the elastic deformation.
fig_warp = plot_intensity_slice(
    image_w,
    before=image,
    roi_mask=mask,
    roi_contour=True,
    title="MONAI Rand3DElastic (image + ROI share one field)",
    before_label="Original",
    image_label="bspline_deform",
)
fig_warp.savefig("out/precise_features_bspline_anatomy.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Zoomed edge perturbation figure: Original vs Deformed ROI with XOR contour difference.
counts = np.sum(ref_bin, axis=(1, 2))
z = int(np.argmax(counts)) if int(np.max(counts)) > 0 else int(ref_bin.shape[0] // 2)
grey = np.take(np.asarray(image.data), z, axis=0)
m0 = np.take(ref_bin, z, axis=0)
m1 = np.take(mov_bin, z, axis=0)

# Crop closely around the ROI on the slice so contour differences are clearly visible
union_slice = m0 | m1
rows = np.any(union_slice, axis=1)
cols = np.any(union_slice, axis=0)
ymin, ymax = np.where(rows)[0][[0, -1]]
xmin, xmax = np.where(cols)[0][[0, -1]]
pad = 20
ymin = max(0, ymin - pad)
ymax = min(grey.shape[0], ymax + pad)
xmin = max(0, xmin - pad)
xmax = min(grey.shape[1], xmax + pad)

grey_c = grey[ymin:ymax, xmin:xmax]
m0_c = m0[ymin:ymax, xmin:xmax]
m1_c = m1[ymin:ymax, xmin:xmax]
xor_c = (m0_c != m1_c)

finite = grey_c[np.isfinite(grey_c)]
vmin, vmax = np.percentile(finite, (2.0, 98.0))

with use_style("radiology"):
    fig_c, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    ax.imshow(grey_c, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
    ax.contourf(xor_c.astype(float), levels=[0.5, 1.5], colors=["#E69F00"], alpha=0.45, origin="upper")
    ax.contour(m0_c.astype(float), levels=[0.5], colors=["#00E5FF"], linewidths=2.0, origin="upper")
    ax.contour(m1_c.astype(float), levels=[0.5], colors=["#D55E00"], linewidths=2.0, linestyles="--", origin="upper")
    ax.set_title(sanitize_label("MONAI Elastic Edge Perturbation (ROI Zoom)"))
    ax.axis("off")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#00E5FF", lw=2.0, label="Original ROI"),
            Line2D([0], [0], color="#D55E00", lw=2.0, ls="--", label="Deformed ROI"),
            Patch(facecolor="#E69F00", edgecolor="none", alpha=0.45, label="Contour shift (XOR)"),
        ],
        loc="lower right",
        frameon=True,
    )
fig_c.savefig("out/precise_features_bspline_contours.png", dpi=150, bbox_inches="tight")
plt.show()
