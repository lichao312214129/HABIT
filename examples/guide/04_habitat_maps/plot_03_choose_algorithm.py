"""
Choose a habitat algorithm
==========================

The habitat **definition** is the cohort fitter: ``kmeans`` vs ``gmm``.
Those two names are different generative assumptions on the same units
(spherical hard clusters vs Gaussian mixtures). The assigner in this
build is ``nearest_centroid`` only.

``slic`` is a **partition** name only — spatial oversegmentation that
makes supervoxels. It is not a habitat fitter and is not compared here.
``kmeans`` / ``gmm`` also exist as partitioners; this page holds the
partitioner fixed (``kmeans`` parcels) so the overlay is the fitter.

Compare maps only **after** habitat-id matching. Independent fitters
permute integers: Dice is undefined until Hungarian overlap names the
pairs. ARI is permutation-invariant and is reported on the same voxels.

``clustering_mode`` (``two_step`` / ``one_step``) is dataflow — where
units are formed and whether a cohort model is shared — not an extra
algorithm next to k-means or GMM.
"""

# sphinx_gallery_thumbnail_number = 1

# %%
# Load two demo subjects. The voxel extractor, preprocessor chain, and
# k-means partition stay fixed so the table and overlay compare only the
# registered fitter.
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from habit.contracts import HabitatMap, cohort_from_directory
from habit.datasets import fetch_demo
from habit.habitat_model import HabitatAssignerRegistry, HabitatModelFitterRegistry
from habit.kernels.habitat_label_match import (
    adjusted_rand_index,
    habitat_dice_from_mapping,
    match_labels_by_overlap,
    present_habitat_ids,
    remap_label_array,
)
from habit.recipes import Study, StudyResult
from habit.spec import HabitatSpec, Spec, Stage
from habit.supervoxel import SupervoxelizerRegistry
from habit.viz import plot_cluster_validation_from_report, plot_habitat_label_compare

DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
RANDOM_SEED = 42
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
print(f"Cohort: {len(cohort)} subjects -> {list(cohort.subject_ids)}")
print("supervoxelizer:", SupervoxelizerRegistry.available())
print("fitter:", HabitatModelFitterRegistry.available())
print("assigner:", HabitatAssignerRegistry.available())

# %%
# Shared voxel field: raw intensities, winsorize, minmax, then k-means
# parcels. k-means validation may use elbow / kneedle / inertia /
# silhouette / calinski_harabasz / davies_bouldin / gap. GMM uses bic /
# aic / silhouette / calinski_harabasz / davies_bouldin / gap — not elbow.


def two_step_spec(
    name: str,
    *,
    fitter: str,
    validation: object,
    n_habitats: Optional[int] = None,
    n_supervoxels: int = 6,
) -> HabitatSpec:
    """Build a two-step spec that differs only at the habitat fitter.

    Partition is always ``kmeans`` so SLIC compactness does not enter the
    comparison. ``slic`` remains a registered supervoxelizer; use it when
    you want spatially compact parcels, not as a stand-in for GMM.

    Args:
        name: Spec name stored on the study (printed in comparison tables).
        fitter: Registered cohort fitter (``kmeans`` or ``gmm``).
        validation: One criterion or a list of criteria for auto-K.
        n_habitats: Fixed K, or ``None`` to search ``2..4``.
        n_supervoxels: Requested parcel count (gallery uses 6).

    Returns:
        A :class:`~habit.spec.HabitatSpec` with a fixed random seed.
    """
    partition_params: Dict[str, Any] = {
        "n_supervoxels": int(n_supervoxels),
        "n_init": 3,
    }
    if n_habitats is None:
        fit_params: Dict[str, Any] = {
            "min_habitats": 2,
            "max_habitats": 4,
            "validation": validation,
            "n_init": 3,
        }
    else:
        fit_params = {"n_habitats": int(n_habitats), "n_init": 3}
    return HabitatSpec(
        name=name,
        stages=(
            Stage(
                "extract_voxel_features",
                Spec("raw", {"modalities": list(MODALITIES)}),
            ),
            Stage(
                "preprocess1",
                Spec(
                    "winsorize",
                    {"winsor_limits": (0.05, 0.05), "across_features": False},
                ),
            ),
            Stage("preprocess2", Spec("minmax", {"across_features": False})),
            Stage("partition", Spec("kmeans", partition_params)),
            Stage("pool", Spec("pool")),
            Stage("fit", Spec(fitter, fit_params)),
            Stage("assign", Spec("nearest_centroid")),
            Stage("quantify", Spec("volume")),
        ),
        random_seed=RANDOM_SEED,
    )


def selection_table_row(
    spec: HabitatSpec,
    result: StudyResult,
) -> Dict[str, Any]:
    """One comparison row: Spec names plus selected K and scores.

    Args:
        spec: Spec that produced ``result``.
        result: Fitted two-step study (cohort ``habitat_model`` required).

    Returns:
        A flat dict for :class:`pandas.DataFrame`.
    """
    model = result.habitat_model
    report: Mapping[str, Any] = {}
    if model is not None:
        report = (model.preprocessing_state or {}).get("selection_report") or {}
    n_units = int(len(result.units[0].features)) if result.units else 0
    selected = int(report.get("selected", model.n_habitats if model else 0))
    scores = report.get("scores") or {}
    candidates = [int(v) for v in report.get("candidates", [])]
    fitter_name = ""
    for stage in spec.stages or ():
        if stage.name == "fit":
            fitter_name = stage.component.name
    row: Dict[str, Any] = {
        "spec_name": spec.name,
        "fitter": fitter_name,
        "n_units": n_units,
        "n_habitats": selected,
    }
    if candidates and selected in candidates:
        idx = candidates.index(selected)
        for method, values in scores.items():
            if idx < len(values):
                row[f"{method}_at_k"] = float(values[idx])
    return row


def compare_after_match(
    subject_id: str,
    reference: HabitatMap,
    moving: HabitatMap,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Match ``moving`` ids onto ``reference``, then score Dice / ARI.

    Same-grid, same-tumour maps use
    :func:`~habit.kernels.habitat_label_match.match_labels_by_overlap`
    (Prior Hungarian / voxel overlap). Dice needs that pairing; ARI does
    not (id permutation leaves it unchanged) but is reported on the same
    voxels.

    Args:
        subject_id: Cohort subject id (printed in the agreement tables).
        reference: k-means habitat map (id space we keep).
        moving: GMM habitat map (ids remapped onto ``reference``).

    Returns:
        ``(mapping_table, dice_table, summary_table, aligned_moving)``.
    """
    ref = np.asarray(reference.label_array)
    mov = np.asarray(moving.label_array)
    mapping = match_labels_by_overlap(ref, mov)
    reserved = [int(v) for v in present_habitat_ids(ref).tolist()]
    aligned = remap_label_array(mov, mapping, reserved_ids=reserved)
    mapping_table = pd.DataFrame(
        [
            {
                "subject_id": subject_id,
                "gmm_id": int(src),
                "kmeans_id": int(dst),
            }
            for src, dst in sorted(mapping.items(), key=lambda item: item[1])
        ]
    )
    dice_rows: List[Dict[str, Any]] = []
    dice_values: List[float] = []
    for habitat_id, matched_id, dice, n_ref, n_mov in habitat_dice_from_mapping(
        ref, mov, mapping
    ):
        dice_values.append(float(dice))
        dice_rows.append(
            {
                "subject_id": subject_id,
                "kmeans_id": int(habitat_id),
                "gmm_id": None if matched_id is None else int(matched_id),
                "dice": float(dice),
                "n_kmeans": int(n_ref),
                "n_gmm": int(n_mov),
            }
        )
    mean_dice = float(np.mean(dice_values)) if dice_values else float("nan")
    summary_table = pd.DataFrame(
        [
            {
                "subject_id": subject_id,
                "n_kmeans_habitats": int(len(present_habitat_ids(ref))),
                "n_gmm_habitats": int(len(present_habitat_ids(mov))),
                "mean_dice": mean_dice,
                "ari": float(adjusted_rand_index(ref, mov)),
            }
        ]
    )
    return mapping_table, pd.DataFrame(dice_rows), summary_table, aligned


# %%
# Fit: k-means vs GMM on the **same** k-means parcels. Auto-K over 2–4.
# Elbow (k-means) and BIC (GMM) are not interchangeable scores; selected
# K may differ. The overlay is the habitat definition, not the parceler.
kmeans_fit_spec = two_step_spec(
    "fit_kmeans_elbow_silhouette",
    fitter="kmeans",
    validation=["elbow", "silhouette"],
)
gmm_fit_spec = two_step_spec(
    "fit_gmm_bic",
    fitter="gmm",
    validation="bic",
)
kmeans_fit_result = Study(spec=kmeans_fit_spec).fit_predict(cohort)
gmm_fit_result = Study(spec=gmm_fit_spec).fit_predict(cohort)
fit_table = pd.DataFrame(
    [
        selection_table_row(kmeans_fit_spec, kmeans_fit_result),
        selection_table_row(gmm_fit_spec, gmm_fit_result),
    ]
)
print(fit_table.to_string(index=False))
fit_table

# %%
# Match GMM ids onto the k-means id space (same grid → overlap Hungarian),
# then score per-habitat Dice and ARI. The overlay uses the remapped GMM
# map (``align_labels=False``) so the colours match this table.
Path("out").mkdir(exist_ok=True)
mapping_frames: List[pd.DataFrame] = []
dice_frames: List[pd.DataFrame] = []
summary_frames: List[pd.DataFrame] = []
aligned_gmm: Optional[np.ndarray] = None
for subject, kmeans_map, gmm_map in zip(
    cohort,
    kmeans_fit_result.habitat_maps,
    gmm_fit_result.habitat_maps,
):
    mapping_table, dice_table, summary_table, aligned = compare_after_match(
        subject.subject_id, kmeans_map, gmm_map
    )
    mapping_frames.append(mapping_table)
    dice_frames.append(dice_table)
    summary_frames.append(summary_table)
    if aligned_gmm is None:
        aligned_gmm = aligned
if aligned_gmm is None:
    raise RuntimeError("compare_after_match produced no remapped GMM map")
mapping_all = pd.concat(mapping_frames, ignore_index=True)
dice_all = pd.concat(dice_frames, ignore_index=True)
summary_all = pd.concat(summary_frames, ignore_index=True)
print("GMM id -> k-means id (overlap Hungarian):")
print(mapping_all.to_string(index=False))
print("Per-habitat Dice after match:")
print(dice_all.to_string(index=False))
print("Summary (mean Dice after match; ARI is permutation-invariant):")
print(summary_all.to_string(index=False))
mapping_all
dice_all
summary_all

fig_fit = plot_habitat_label_compare(
    cohort[0].image(ROI),
    kmeans_fit_result.habitat_maps[0],
    aligned_gmm,
    titles=("k-means habitats", "GMM habitats (matched)"),
    align_labels=False,
)
fig_fit.savefig("out/choose_kmeans_vs_gmm.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Validation curves from each ``selection_report``. k-means shows elbow
# vs silhouette on one figure; GMM shows BIC (no inertia / kneedle).
kmeans_report = (kmeans_fit_result.habitat_model.preprocessing_state or {}).get(
    "selection_report"
)
gmm_report = (gmm_fit_result.habitat_model.preprocessing_state or {}).get(
    "selection_report"
)
if kmeans_report:
    fig_k = plot_cluster_validation_from_report(
        kmeans_report, title="k-means: elbow vs silhouette"
    )
    fig_k.savefig("out/choose_kmeans_validation.png", dpi=150, bbox_inches="tight")
    plt.show()
if gmm_report:
    fig_g = plot_cluster_validation_from_report(gmm_report, title="GMM: BIC")
    fig_g.savefig("out/choose_gmm_validation.png", dpi=150, bbox_inches="tight")
    plt.show()
print("k-means report methods:", (kmeans_report or {}).get("methods"))
print("GMM report methods:", (gmm_report or {}).get("methods"))
