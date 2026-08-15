"""Precise-feature atoms: perturb, extract, then combine like the paper.

Teaches the Prior et al. (Radiol Artif Intell 2024;6(2):e230118) screen
as volume-level atoms, then the small composition that builds ICC / LCL
and a PreciseFeatureSet:

* perturb_image(image, method, **params) -- one registered method
* extract_voxel_texture(image, mask, kernel_radius, bin_width, ...)
* repeatability = extract(original) vs extract(perturbed)
* reproducibility = extract(orig, params_A) vs extract(orig, params_B)
* precise = intersection of those flags (identify_precise_features)

Habitats from the full texture set vs the precise subset come after.
An optional follow-up warps the ROI contour (Subject-level
bspline_deform) and scores habitat-table feature ICC + 95% CI.
The one-call recipe identify_precise_voxel_features is optional (see
the rst page), not this script.

Run from the repository root::

    python docs/source/examples/scripts/precise_features_demo.py
"""

from __future__ import annotations

# BEGIN example
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from habit import (
    Cohort,
    HabitatSpec,
    Spec,
    Subject,
    aggregate_panels,
    cohort_from_directory,
    extract_voxel_texture,
    identify_precise_features,
    perturb_image,
    precision_panel,
)
from habit.contracts import ArrayImageRef, Geometry
from habit.recipes import Study

# Change DATA / MODALITIES / ROI to your preprocessed layout
DATA = "demo_data/preprocessed"
MODALITIES = ("LAP",)
ROI = "LAP"

cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
subject = cohort[0]


def _crop_to_roi(item: Subject, modality: str, roi: str, pad: int = 8) -> Subject:
    """Crop one subject to the ROI bounding box plus pad (demo speed)."""
    mask_arr = np.asarray(item.mask(roi).data)
    image_arr = np.asarray(item.image(modality).data)
    nz = np.argwhere(mask_arr > 0)
    lo = np.maximum(nz.min(axis=0) - pad, 0)
    hi = np.minimum(nz.max(axis=0) + pad + 1, mask_arr.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    src = item.image(modality).geometry
    geom = Geometry.from_array(
        image_arr[sl].shape, spacing=src.spacing, direction=src.direction
    )
    return Subject(
        subject_id=item.subject_id,
        images={modality: ArrayImageRef(array=image_arr[sl], geometry=geom)},
        masks={roi: ArrayImageRef(array=mask_arr[sl], geometry=geom)},
    )


subject = _crop_to_roi(subject, MODALITIES[0], ROI)
image = subject.image(MODALITIES[0])
mask = subject.mask(ROI)

# --- Atom 1: perturb (image + method + params -> same-grid image) ---
# gaussian_noise with sigma omitted uses Chang's wavelet estimator.
perturbed = perturb_image(image, method="gaussian_noise", seed=7)
shifted = perturb_image(
    image,
    method="translation",
    shift_voxels=(0.5, 0.5, 0.0),
    random_signs=False,
    seed=7,
)
rotated = perturb_image(image, method="rotation", angle_degrees=0.5, seed=7)
print("perturbed methods:", "gaussian_noise, translation, rotation")
print(f"  grid unchanged: {perturbed.data.shape == image.data.shape}")

# --- Atom 2: extract texture (same image, different paper knobs) ---
# Small first-order + GLCM set so R1 vs R3 stays interactive on a crop.
# Pass feature_classes=None to use the bundled voxel (paper CT) preset.
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
print(f"  features: {list(feat_r3.feature_names)}")
print(f"  voxels: {feat_r3.values.shape[0]}")

# --- Combine like the paper (small composition, not a new extractor) ---
# repeatability: extract(original) vs extract(perturbed) at the base setting
# reproducibility: extract(orig, params_A) vs extract(orig, params_B)
# precise = intersection of LCL >= 0.5 across experiments that were run
repeat_panel = precision_panel(
    {"original": feat_r3, "perturbed": feat_pert},
    agreement="absolute",
)
kernel_panel = precision_panel(
    {"R1": feat_r1, "R3": feat_r3},
    agreement="consistency",
)
bin_panel = precision_panel(
    {"B12": feat_r3, "B25": feat_b25},
    agreement="consistency",
)
precise = identify_precise_features(
    {
        "repeatability": aggregate_panels([repeat_panel]),
        "reproducibility_kernel_radius": aggregate_panels([kernel_panel]),
        "reproducibility_bin_width": aggregate_panels([bin_panel]),
    },
    lcl_threshold=0.5,
)
evidence = precise.to_frame().round(3)
screened: Tuple[str, ...] = tuple(precise.feature_names)
all_names = list(dict.fromkeys(evidence["feature"].astype(str).tolist()))
unstable = [name for name in all_names if name not in set(screened)]
print(f"  experiments: {list(precise.experiments)}")
print(f"  precise: {list(screened)}")
print(f"  unstable: {unstable}")
print(evidence.to_string(index=False))

# --- Habitat use: all-texture vs precise-texture (same k search) ---
cropped = Cohort(subjects=(subject,))
texture_params: Dict[str, Any] = {
    "imageType": {"Original": {}},
    "featureClass": {key: list(values) for key, values in FEATURE_CLASSES.items()},
    "setting": {"binWidth": 12.0, "normalize": False},
}
extractor_spec = Spec(
    "voxel_radiomics",
    {
        "modalities": list(MODALITIES),
        "kernel_radius": 3,
        "params": texture_params,
    },
)
fitter_spec = Spec(
    "kmeans",
    {"min_habitats": 2, "max_habitats": 3, "validation": "elbow", "n_init": 3},
)
minmax_spec = Spec("minmax", {"across_features": False})
assigner_spec = Spec("nearest_centroid")
spec_all = HabitatSpec(
    name="all_texture_one_step",
    voxel_feature_extractor=extractor_spec,
    voxel_feature_preprocessors=(minmax_spec,),
    habitat_model_fitter=fitter_spec,
    habitat_assigner=assigner_spec,
    random_seed=11,
    pooling="none",
)
result_all = Study(spec_all).fit_predict(cropped)
result_precise = None
if screened:
    whitelist = precise.preprocessor()
    spec_precise = HabitatSpec(
        name="precise_one_step",
        voxel_feature_extractor=extractor_spec,
        voxel_feature_preprocessors=(whitelist.spec, minmax_spec),
        habitat_model_fitter=fitter_spec,
        habitat_assigner=assigner_spec,
        random_seed=11,
        pooling="none",
    )
    result_precise = Study(spec_precise).fit_predict(cropped)
    print(f"  habitat maps: all={len(result_all.habitat_maps)} precise={len(result_precise.habitat_maps)}")
else:
    print("  no feature passed every experiment; skip precise habitats")
# END example

# BEGIN figures
# Paste after the Script block. Uses image, mask, perturbed, shifted,
# rotated, precise, evidence, result_all, result_precise.
from habit.viz import (
    plot_habitat_label_compare,
    plot_intensity_slice,
    plot_precision_icc,
)

Path("out").mkdir(exist_ok=True)
written: list[str] = []


def _save_fig(fig: object, name: str) -> None:
    """Write one taught figure under out/."""
    fig.savefig(f"out/{name}", dpi=150, bbox_inches="tight")
    written.append(name)


_save_fig(
    plot_intensity_slice(
        perturbed,
        before=image,
        roi_mask=mask,
        roi_contour=True,
        title="Original vs Gaussian noise",
        before_label="Original",
        image_label="Gaussian noise",
    ),
    "precise_features_original_vs_perturbed.png",
)

# Small multiples: each panel is one perturb_image call (same slice as above).
# plot_intensity_slice is the pair figure; this mosaic only arranges those
# already-computed volumes.
import matplotlib.pyplot as plt

from habit.viz import use_style
from habit.viz.labels import sanitize_label

mask_arr = np.asarray(mask.data)
counts = np.sum(mask_arr > 0, axis=(1, 2))
index = int(np.argmax(counts)) if int(np.max(counts)) > 0 else int(mask_arr.shape[0] // 2)
panels = (
    ("Original", image),
    ("Gaussian noise", perturbed),
    ("Translation 0.5 vx", shifted),
    ("Rotation 0.5 deg", rotated),
)
orig_sl = np.take(np.asarray(image.data), index, axis=0)
finite = orig_sl[np.isfinite(orig_sl)]
lo, hi = np.percentile(finite, (1.0, 99.0))
with use_style("radiology"):
    fig_m, axes_m = plt.subplots(1, 4, figsize=(12.8, 3.4), constrained_layout=True)
    for ax, (label, volume) in zip(axes_m, panels):
        sl = np.take(np.asarray(volume.data), index, axis=0)
        ax.imshow(
            sl,
            cmap="gray",
            interpolation="nearest",
            origin="upper",
            vmin=lo,
            vmax=hi,
        )
        ax.set_title(sanitize_label(label))
        ax.axis("off")
    fig_m.suptitle(sanitize_label("perturb_image methods (same slice)"))
_save_fig(fig_m, "precise_features_perturb_methods.png")


def _experiment_frame(table: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """Rows of one experiment, coloured by that experiment's own LCL."""
    panel = table.loc[table["experiment"] == experiment].copy()
    return panel.drop(columns=["precise"], errors="ignore")


icc_jobs = (
    ("repeatability", "precise_features_icc_lcl.png", "Repeatability ICC and 95% CI"),
    (
        "reproducibility_kernel_radius",
        "precise_features_icc_kernel.png",
        "Kernel-radius reproducibility ICC and 95% CI",
    ),
    (
        "reproducibility_bin_width",
        "precise_features_icc_bin.png",
        "Bin-width reproducibility ICC and 95% CI",
    ),
)
for experiment, filename, title in icc_jobs:
    if experiment not in set(precise.experiments):
        continue
    panel = _experiment_frame(evidence, experiment).dropna(subset=["value", "lcl", "ucl"])
    if panel.empty:
        continue
    _save_fig(
        plot_precision_icc(panel, lcl_threshold=precise.lcl_threshold, title=title),
        filename,
    )

combined = evidence.dropna(subset=["value", "lcl", "ucl"])
if not combined.empty:
    _save_fig(
        plot_precision_icc(
            combined,
            lcl_threshold=precise.lcl_threshold,
            title="All experiments: ICC and 95% CI",
        ),
        "precise_features_icc_all.png",
    )

if result_precise is not None:
    _save_fig(
        plot_habitat_label_compare(
            image,
            result_all.habitat_maps[0],
            result_precise.habitat_maps[0],
            titles=("All texture features", "Precise features only"),
            align_labels=True,
        ),
        "precise_features_all_vs_precise.png",
    )
print("Wrote " + ", ".join(f"out/{name}" for name in written))
# END figures

# BEGIN roi_followup
# Optional follow-up (not Prior Appendix S2, not MIRP ROI grow/shrink):
# warp image+mask together, then score habitat-map Dice and habitat-table
# feature ICC(3A,1) with a 95% CI. Paste after the Script block. Uses
# _crop_to_roi, DATA, MODALITIES, ROI.
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from habit import (
    HabitatMap,
    ImagePerturbationRegistry,
    align_habitat_map,
    habitat_stability,
    habitat_volume_fractions,
    icc3a_1,
    ith_score,
    one_step_habitat,
)
from habit.viz import plot_habitat_label_compare, plot_precision_icc, use_style
from habit.viz.labels import sanitize_label
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

Path("out").mkdir(exist_ok=True)

deform = ImagePerturbationRegistry.create(
    "bspline_deform",
    target_dice=0.85,
    dice_tolerance=0.03,
    sigma_range=(1.0, 2.0),
    magnitude_range=(10.0, 15.0),
    device="cpu",
)


def _habitat_feature_row(habitat_map: HabitatMap) -> Dict[str, float]:
    """Volume fractions and ITH from one HabitatMap (aligned ids)."""
    labels = np.asarray(habitat_map.label_array)
    habitat_ids = tuple(int(hid) for hid in habitat_map.habitat_ids)
    fractions = habitat_volume_fractions(labels, habitat_ids)
    row: Dict[str, float] = {
        f"habitat_{hid}_volume_fraction": float(fractions[hid])
        for hid in habitat_ids
    }
    row["ith_score"] = float(ith_score(labels))
    return row


FEATURE_ORDER = (
    "habitat_1_volume_fraction",
    "habitat_2_volume_fraction",
    "habitat_3_volume_fraction",
    "ith_score",
)
orig_rows: List[Dict[str, float]] = []
warp_rows: List[Dict[str, float]] = []
first_bundle: Optional[tuple] = None
icc_source = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:3]
for item in icc_source:
    cropped = _crop_to_roi(item, MODALITIES[0], ROI)
    print(f"FFD + one-step habitats: {cropped.subject_id}", flush=True)
    warped_item = deform(cropped, rng=np.random.default_rng(7))
    orig_fit = one_step_habitat(
        modalities=MODALITIES, n_habitats=3, random_seed=0, roi=ROI
    ).fit_predict(Cohort(subjects=(cropped,)))
    warp_fit = one_step_habitat(
        modalities=MODALITIES, n_habitats=3, random_seed=0, roi=ROI
    ).fit_predict(Cohort(subjects=(warped_item,)))
    ref_map = orig_fit.habitat_maps[0]
    mov_map = warp_fit.habitat_maps[0]
    # Independent one_step fits share a model_id digest; force overlap align
    # so table columns (habitat_1_...) name the same spatial region.
    aligned_map = align_habitat_map(ref_map, mov_map, method="overlap", force=True)
    orig_rows.append(_habitat_feature_row(ref_map))
    warp_rows.append(_habitat_feature_row(aligned_map))
    if first_bundle is None:
        first_bundle = (
            cropped,
            warped_item,
            ref_map,
            aligned_map,
            habitat_stability(ref_map, [mov_map]),
        )

cropped, warped_item, ref_map, aligned_map, dice_frame = first_bundle
print("Habitat Dice after bspline_deform (Hungarian match)")
print(dice_frame.to_string(index=False))

# Shared axial index: densest original ROI (same crop, same slice).
orig_mask = np.asarray(cropped.mask(ROI).data)
warped_mask = np.asarray(warped_item.mask(ROI).data)
counts = np.sum(orig_mask > 0, axis=(1, 2))
index = int(np.argmax(counts)) if int(np.max(counts)) > 0 else int(orig_mask.shape[0] // 2)
grey = np.take(np.asarray(cropped.image(MODALITIES[0]).data), index, axis=0)
mask_orig = np.take(orig_mask > 0, index, axis=0).astype(np.uint8)
mask_warp = np.take(warped_mask > 0, index, axis=0).astype(np.uint8)
xor_map = np.abs(mask_warp.astype(np.float64) - mask_orig.astype(np.float64))
finite = grey[np.isfinite(grey)]
lo, hi = np.percentile(finite, (1.0, 99.0))
original_color = "#00E5FF"
warped_color = "#D55E00"
xor_color = "#F0E442"
with use_style("radiology"):
    fig_edge, axes_edge = plt.subplots(
        1, 2, figsize=(8.8, 4.4), constrained_layout=True
    )
    for ax, show_xor, title in (
        (axes_edge[0], False, "Original vs warped ROI"),
        (axes_edge[1], True, "Membership change (XOR)"),
    ):
        ax.imshow(
            grey, cmap="gray", interpolation="nearest", origin="upper", vmin=lo, vmax=hi
        )
        if show_xor and np.any(xor_map > 0):
            ax.contourf(
                xor_map, levels=[0.5, 1.5], colors=[xor_color], alpha=0.55, origin="upper"
            )
        ax.contour(mask_orig, levels=[0.5], colors=[original_color], linewidths=1.6, origin="upper")
        ax.contour(
            mask_warp,
            levels=[0.5],
            colors=[warped_color],
            linewidths=1.6,
            linestyles="--",
            origin="upper",
        )
        ax.set_title(sanitize_label(title))
        ax.axis("off")
    fig_edge.legend(
        handles=[
            Line2D([0], [0], color=original_color, lw=1.6, label="Original ROI"),
            Line2D(
                [0], [0], color=warped_color, lw=1.6, ls="--", label="Warped ROI"
            ),
            Patch(facecolor=xor_color, edgecolor="none", alpha=0.55, label="XOR"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
fig_edge.savefig("out/precise_perturb_mask_edge.png", dpi=150, bbox_inches="tight")

fig_cmp = plot_habitat_label_compare(
    cropped.image(MODALITIES[0]),
    ref_map,
    aligned_map,
    titles=("Original ROI habitats", "Warped ROI habitats"),
    index=index,
    align_labels=False,
    display_convention="native",
)
fig_cmp.savefig(
    "out/precise_habitat_stability_compare.png", dpi=150, bbox_inches="tight"
)

with use_style("radiology"):
    fig_dice, ax_dice = plt.subplots(figsize=(5.4, 3.2), constrained_layout=True)
    ax_dice.bar(
        [f"H{int(hid)}" for hid in dice_frame["habitat_id"].to_numpy()],
        dice_frame["dice"].to_numpy(dtype=float),
        color="#0072B2",
    )
    ax_dice.set_ylim(0.0, 1.05)
    ax_dice.set_ylabel("Dice")
    ax_dice.set_title(sanitize_label("Per-habitat Dice (matched)"))
fig_dice.savefig("out/precise_habitat_dice.png", dpi=150, bbox_inches="tight")

# ICC(3A,1) on the habitat-table features: one row per subject, two columns
# (original vs overlap-aligned warped map). n=3 => wide 95% CIs (honest).
feature_names = [
    name
    for name in FEATURE_ORDER
    if name in orig_rows[0] and name in warp_rows[0]
]
icc_records = []
for name in feature_names:
    matrix = np.column_stack(
        [
            [float(row[name]) for row in orig_rows],
            [float(row[name]) for row in warp_rows],
        ]
    )
    if not np.isfinite(matrix).all():
        continue
    estimate = icc3a_1(matrix)
    icc_records.append(
        {
            "feature": name,
            "value": estimate.value,
            "lcl": estimate.lcl,
            "ucl": estimate.ucl,
            "precise": estimate.lcl >= 0.5,
        }
    )
icc_frame = pd.DataFrame(icc_records)
print("Habitat-table feature ICC(3A,1) with 95% CI (n=3 subjects)")
print(icc_frame.to_string(index=False))
fig_icc = plot_precision_icc(
    icc_frame,
    lcl_threshold=0.5,
    title="Habitat-table features: ICC and 95% CI (FFD retest)",
)
fig_icc.savefig("out/precise_habitat_feature_icc.png", dpi=150, bbox_inches="tight")
print(
    "Wrote out/precise_perturb_mask_edge.png, "
    "out/precise_habitat_stability_compare.png, "
    "out/precise_habitat_dice.png, "
    "out/precise_habitat_feature_icc.png"
)
# END roi_followup

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import copy_out_figures_to_gallery

    copy_out_figures_to_gallery(
        (
            "precise_features_original_vs_perturbed.png",
            "precise_features_perturb_methods.png",
            "precise_features_icc_lcl.png",
            "precise_features_icc_kernel.png",
            "precise_features_icc_bin.png",
            "precise_features_icc_all.png",
            "precise_features_all_vs_precise.png",
            "precise_perturb_mask_edge.png",
            "precise_habitat_stability_compare.png",
            "precise_habitat_dice.png",
            "precise_habitat_feature_icc.png",
        )
    )
