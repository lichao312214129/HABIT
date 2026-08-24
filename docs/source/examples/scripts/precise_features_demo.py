"""Precise-feature atoms: perturb, extract, then combine like the paper.

Teaches the Prior et al. (Radiol Artif Intell 2024;6(2):e230118) screen
as volume-level atoms, then the small composition that builds ICC / LCL
and a PreciseFeatureSet:

* perturb_image(image, method, **params) -- one registered method
* Appendix S2 retest = noise then 0.5-vx translation then 0.5 deg rotation
* extract_voxel_texture(image, mask, kernel_radius, bin_width, ...)
* repeatability = extract(original) vs extract(chained retest)
* reproducibility = extract(orig, params_A) vs extract(orig, params_B)
* precise = intersection of those flags (identify_precise_features)

Habitats from the full texture set vs the precise subset come after.
An optional follow-up warps image and mask together with MONAI
B-spline / elastic FFD (bspline_deform), keeps only the intersection
of the two ROIs, extracts every light habitat-map family on that
core, and scores ICC + 95% CI plus a paired difference heatmap. The
one-call recipe identify_precise_voxel_features is optional (see the
rst page), not this script.

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
# Prior Appendix S2 / MIRP 1.2.0 simulated retest as three sequential
# atoms (not the prior2024_retest_perturbation recipe): Chang-estimated
# Gaussian noise, then 0.5-voxel translation (random axis signs), then
# +0.5 deg in-plane (z) rotation. Intensity interpolator defaults to
# B-spline. One rng is advanced through the chain so the draws match
# a single PerturbationChain call.
retest_rng = np.random.default_rng(7)
noisy = perturb_image(image, method="gaussian_noise", rng=retest_rng)
shifted = perturb_image(
    noisy,
    method="translation",
    shift_fraction=0.5,
    rng=retest_rng,
)
perturbed = perturb_image(
    shifted,
    method="rotation",
    angle_degrees=0.5,
    rng=retest_rng,
)
print("perturbed methods:", "gaussian_noise -> translation -> rotation")
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
# Paste after the Script block. Uses image, mask, noisy, shifted,
# perturbed, precise, evidence, result_all, result_precise.
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
        title="Original vs Appendix S2 retest",
        before_label="Original",
        image_label="Noise + 0.5 vx + 0.5 deg",
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
    ("Gaussian noise", noisy),
    ("+ translation 0.5 vx", shifted),
    ("+ rotation 0.5 deg", perturbed),
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
    fig_m.suptitle(sanitize_label("Appendix S2 chain (sequential perturb_image atoms)"))
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
        plot_precision_icc(
            panel,
            lcl_threshold=precise.lcl_threshold,
            title=title,
            orientation="row",
        ),
        filename,
    )

combined = evidence.dropna(subset=["value", "lcl", "ucl"])
if not combined.empty:
    _save_fig(
        plot_precision_icc(
            combined,
            lcl_threshold=precise.lcl_threshold,
            title="All experiments: ICC and 95% CI",
            orientation="row",
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
# Optional follow-up (not Prior Appendix S2): MONAI B-spline /
# elastic FFD (bspline_deform) warps image and mask together, then
# score habitats and every light habitat-map family on the
# intersection of the two ROIs. Paste after the Script block.
# Uses _crop_to_roi, DATA, MODALITIES, ROI.
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from habit import (
    HabitatGraphFeatureOptions,
    HabitatMap,
    ImagePerturbationRegistry,
    align_habitat_map,
    extract_graph_features,
    habitat_region_stats,
    habitat_stability,
    habitat_volume_fractions,
    icc3a_1,
    ith_score,
    msi_features_from_matrix,
    one_step_habitat,
    spatial_interaction_matrix,
)
from habit.viz import (
    plot_graph_feature_heatmap,
    plot_habitat_label_compare,
    plot_precision_icc,
    use_style,
)
from habit.viz.labels import sanitize_label
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

Path("out").mkdir(exist_ok=True)

# Coarse-lattice cubic B-spline FFD (control_spacing=16 vx). Image
# and mask share one displacement field so the contour stays paired
# with anatomy. Knots every 16 voxels make a slow bulge; the default
# MONAI Rand3DElastic path (full-res noise + Gaussian) looks like
# 1-voxel teeth. bilinear mask + rint is a 0.5 iso-contour.
# target_dice scales that field so ROI overlap is about 0.95.
# Intersection of the two masks is the core both contours still cover.
edge = ImagePerturbationRegistry.create(
    "bspline_deform",
    target_dice=0.95,
    dice_tolerance=0.02,
    control_spacing=16.0,
    magnitude_range=(4.0, 10.0),
    mask_mode="bilinear",
    device="cpu",
)


def _restrict_to_intersection(
    habitat_map: HabitatMap, keep: np.ndarray
) -> HabitatMap:
    """Zero habitat labels outside the intersection of the two ROIs."""
    labels = np.asarray(habitat_map.label_array).copy()
    labels[np.asarray(keep) <= 0] = 0
    return replace(habitat_map, label_array=labels)


def _all_habitat_features(habitat_map: HabitatMap) -> Dict[str, float]:
    """
    Every light habitat-map family on one (already restricted) map.

    volume, non_radiomics, ith_score, msi, and graph. IBSI radiomics
    families are omitted: they need a params file and dominate runtime.
    """
    labels = np.asarray(habitat_map.label_array)
    habitat_ids: Tuple[int, ...] = tuple(int(hid) for hid in habitat_map.habitat_ids)
    fractions = habitat_volume_fractions(labels, habitat_ids)
    stats = habitat_region_stats(labels)
    row: Dict[str, float] = {"num_habitats": float(len(stats))}
    for hid in habitat_ids:
        count = int(np.count_nonzero(labels == hid))
        row[f"habitat_{hid}_voxel_count"] = float(count)
        row[f"habitat_{hid}_volume_fraction"] = float(fractions[hid])
        n_regions, _largest = stats.get(int(hid), (0, 0))
        row[f"{hid}_num_regions"] = float(n_regions)
        row[f"{hid}_volume_ratio"] = float(fractions[hid])
    row["ith_score"] = float(ith_score(labels))
    n_classes = (max(habitat_ids) + 1) if habitat_ids else 1
    row.update(msi_features_from_matrix(spatial_interaction_matrix(labels, n_classes)))
    # Gallery pins extended graph metrics off (library default is on).
    graph = extract_graph_features(
        labels,
        expected_labels=habitat_ids,
        options=HabitatGraphFeatureOptions(include_extended_metrics=False),
    )
    for key, value in graph.items():
        row[str(key)] = float(value)
    return row


LIGHT_FIRST = (
    "habitat_1_volume_fraction",
    "habitat_2_volume_fraction",
    "habitat_3_volume_fraction",
    "ith_score",
    "contrast",
    "homogeneity",
    "correlation",
    "energy",
    "1_num_regions",
    "2_num_regions",
    "3_num_regions",
)
orig_rows: List[Dict[str, float]] = []
edge_rows: List[Dict[str, float]] = []
subject_ids: List[str] = []
first_bundle: Optional[tuple] = None
icc_source = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:3]
for item in icc_source:
    cropped = _crop_to_roi(item, MODALITIES[0], ROI)
    print(f"B-spline warp + intersection habitats: {cropped.subject_id}", flush=True)
    edge_item = edge(cropped, rng=np.random.default_rng(7))
    orig_mask = np.asarray(cropped.mask(ROI).data) > 0
    edge_mask = np.asarray(edge_item.mask(ROI).data) > 0
    intersection = orig_mask & edge_mask
    n_orig = int(orig_mask.sum())
    n_edge = int(edge_mask.sum())
    n_inter = int(intersection.sum())
    print(
        f"  ROI voxels original={n_orig} warped={n_edge} intersection={n_inter}",
        flush=True,
    )
    orig_fit = one_step_habitat(
        modalities=MODALITIES, n_habitats=3, random_seed=0, roi=ROI
    ).fit_predict(Cohort(subjects=(cropped,)))
    edge_fit = one_step_habitat(
        modalities=MODALITIES, n_habitats=3, random_seed=0, roi=ROI
    ).fit_predict(Cohort(subjects=(edge_item,)))
    orig_image = cropped.image(MODALITIES[0])
    edge_image = edge_item.image(MODALITIES[0])
    # Restrict both maps to the agreed core before pairing / features.
    ref_core = _restrict_to_intersection(orig_fit.habitat_maps[0], intersection)
    mov_core = _restrict_to_intersection(edge_fit.habitat_maps[0], intersection)
    # Pair by Hungarian assignment on per-habitat mean intensity (same
    # quantity k-means uses as a cluster centre). force=True: independent
    # one_step fits share a model_id digest even though ids are permuted.
    aligned_core = align_habitat_map(
        ref_core,
        mov_core,
        method="centroid",
        image=orig_image,
        moving_image=edge_image,
        force=True,
    )
    print(f"  light habitat-map features on intersection: {cropped.subject_id}", flush=True)
    orig_rows.append(_all_habitat_features(ref_core))
    edge_rows.append(_all_habitat_features(aligned_core))
    subject_ids.append(cropped.subject_id)
    if first_bundle is None:
        first_bundle = (
            cropped,
            edge_item,
            intersection,
            ref_core,
            aligned_core,
            habitat_stability(
                ref_core,
                [mov_core],
                method="centroid",
                image=orig_image,
                moving_images=(edge_image,),
            ),
        )

cropped, edge_item, intersection, ref_core, aligned_core, dice_frame = first_bundle
print("Habitat Dice on ROI intersection (mean-intensity match)")
print(dice_frame.to_string(index=False))

# Shared axial index: densest original ROI (same crop, same slice).
orig_mask = np.asarray(cropped.mask(ROI).data)
edge_mask = np.asarray(edge_item.mask(ROI).data)
counts = np.sum(orig_mask > 0, axis=(1, 2))
index = int(np.argmax(counts)) if int(np.max(counts)) > 0 else int(orig_mask.shape[0] // 2)
grey = np.take(np.asarray(cropped.image(MODALITIES[0]).data), index, axis=0)
mask_orig = np.take(orig_mask > 0, index, axis=0).astype(np.uint8)
mask_edge = np.take(edge_mask > 0, index, axis=0).astype(np.uint8)
mask_inter = np.take(intersection, index, axis=0).astype(np.uint8)
xor_map = np.abs(mask_edge.astype(np.float64) - mask_orig.astype(np.float64))
finite = grey[np.isfinite(grey)]
lo, hi = np.percentile(finite, (1.0, 99.0))
original_color = "#00E5FF"
edge_color = "#D55E00"
xor_color = "#F0E442"
inter_color = "#56B4E9"
with use_style("radiology"):
    fig_edge, axes_edge = plt.subplots(
        1, 2, figsize=(8.8, 4.4), constrained_layout=True
    )
    for ax, show_core, title in (
        (axes_edge[0], False, "Original vs B-spline warped ROI"),
        (axes_edge[1], True, "Intersection and XOR"),
    ):
        ax.imshow(
            grey, cmap="gray", interpolation="nearest", origin="upper", vmin=lo, vmax=hi
        )
        if show_core and np.any(mask_inter > 0):
            ax.contourf(
                mask_inter,
                levels=[0.5, 1.5],
                colors=[inter_color],
                alpha=0.35,
                origin="upper",
            )
        if show_core and np.any(xor_map > 0):
            ax.contourf(
                xor_map, levels=[0.5, 1.5], colors=[xor_color], alpha=0.55, origin="upper"
            )
        ax.contour(mask_orig, levels=[0.5], colors=[original_color], linewidths=1.6, origin="upper")
        ax.contour(
            mask_edge,
            levels=[0.5],
            colors=[edge_color],
            linewidths=1.6,
            linestyles="--",
            origin="upper",
        )
        ax.set_title(sanitize_label(title))
        ax.axis("off")
    fig_edge.legend(
        handles=[
            Line2D([0], [0], color=original_color, lw=1.6, label="Original ROI"),
            Line2D([0], [0], color=edge_color, lw=1.6, ls="--", label="Warped ROI"),
            Patch(facecolor=inter_color, edgecolor="none", alpha=0.35, label="Intersection"),
            Patch(facecolor=xor_color, edgecolor="none", alpha=0.55, label="XOR"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
fig_edge.savefig("out/precise_perturb_mask_edge.png", dpi=150, bbox_inches="tight")

fig_cmp = plot_habitat_label_compare(
    cropped.image(MODALITIES[0]),
    ref_core,
    aligned_core,
    titles=("Original habitats in intersection", "Warped habitats in intersection"),
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
    ax_dice.set_title(sanitize_label("Per-habitat Dice on intersection"))
fig_dice.savefig("out/precise_habitat_dice.png", dpi=150, bbox_inches="tight")

# ICC(3A,1) on every shared light-family column. One row per subject,
# two columns (original-core vs aligned warped-core). n=3 => wide CIs.
shared_names = set(orig_rows[0]) & set(edge_rows[0])
feature_names = [name for name in LIGHT_FIRST if name in shared_names]
feature_names.extend(
    sorted(name for name in shared_names if name not in set(LIGHT_FIRST))
)
icc_records = []
for name in feature_names:
    matrix = np.column_stack(
        [
            [float(row[name]) for row in orig_rows],
            [float(row[name]) for row in edge_rows],
        ]
    )
    if not np.isfinite(matrix).all():
        continue
    # Skip columns that do not vary across the paired conditions
    # (ICC is then a 0/0 sentinel, not a real agreement score).
    if float(np.std(matrix[:, 0] - matrix[:, 1])) == 0.0:
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
print(
    f"Intersection habitat-feature ICC(3A,1) with 95% CI "
    f"(n=3 subjects, {len(icc_frame)} shared columns)"
)
# Full light+graph table is too tall. Keep the teaching columns, then
# fill with a reproducible random subset. Printed count is the full set.
priority = icc_frame[icc_frame["feature"].isin(LIGHT_FIRST)]
remainder = icc_frame[~icc_frame["feature"].isin(LIGHT_FIRST)]
n_extra = max(0, min(16, len(remainder)))
extra = remainder.sample(n=n_extra, random_state=0) if n_extra else remainder.iloc[0:0]
plot_frame = pd.concat([priority, extra], ignore_index=True).sort_values(
    "feature", kind="stable"
)
print(f"Plotting {len(plot_frame)} columns (light families + random graph subset)")
print(plot_frame.to_string(index=False))
fig_icc = plot_precision_icc(
    plot_frame,
    lcl_threshold=0.5,
    title="Intersection habitat features: ICC and 95% CI",
    orientation="row",
)
fig_icc.savefig("out/precise_habitat_feature_icc.png", dpi=150, bbox_inches="tight")

orig_table = pd.DataFrame(orig_rows)
orig_table.insert(0, "subject_id", subject_ids)
edge_table = pd.DataFrame(edge_rows)
edge_table.insert(0, "subject_id", subject_ids)
# Difference heatmap: highest-variance raw (warped - original) columns.
delta = edge_table.set_index("subject_id") - orig_table.set_index("subject_id")
variances = delta.var(axis=0, skipna=True).sort_values(ascending=False)
delta_features = tuple(variances.head(40).index)
fig_delta = plot_graph_feature_heatmap(
    edge_table,
    reference=orig_table,
    subjects=tuple(subject_ids),
    features=delta_features,
    zscore=True,
    star_significant=True,
    title="Intersection features: warped minus original",
    cbar_label="Z-scored difference (warped - original)",
)
fig_delta.savefig("out/precise_habitat_feature_delta.png", dpi=150, bbox_inches="tight")
print(
    "Wrote out/precise_perturb_mask_edge.png, "
    "out/precise_habitat_stability_compare.png, "
    "out/precise_habitat_dice.png, "
    "out/precise_habitat_feature_icc.png, "
    "out/precise_habitat_feature_delta.png"
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
            "precise_habitat_feature_delta.png",
        )
    )
