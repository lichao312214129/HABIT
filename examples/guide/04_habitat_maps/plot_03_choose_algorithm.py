"""
Choose a habitat algorithm
==========================

The habitat **definition** is the cohort fitter: ``kmeans`` vs ``gmm``.
The assigner in this build is ``nearest_centroid`` only. ``slic`` is a
partition name (oversegmentation), not a fitter — parcels stay
``kmeans`` so the overlay compares only the fitter.

Compare maps **after** habitat-id matching. Independent fitters permute
integers: Dice needs Hungarian overlap; ARI is permutation-invariant.
"""

# sphinx_gallery_thumbnail_number = 1

# %%
# Load two demo subjects and list registered components.
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from habit.contracts import cohort_from_directory
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
Path("out").mkdir(exist_ok=True)
print(f"Cohort: {list(cohort.subject_ids)}")
print("fitter:", HabitatModelFitterRegistry.available())
print("assigner:", HabitatAssignerRegistry.available())
print("supervoxelizer:", SupervoxelizerRegistry.available())

# %%
# Shared two-step stages; only the ``fit`` stage differs.
shared_stages = (
    Stage("extract_voxel_features", Spec("raw", {"modalities": list(MODALITIES)})),
    Stage(
        "preprocess1",
        Spec("winsorize", {"winsor_limits": (0.05, 0.05), "across_features": False}),
    ),
    Stage("preprocess2", Spec("minmax", {"across_features": False})),
    Stage("partition", Spec("kmeans", {"n_supervoxels": 6, "n_init": 3})),
    Stage("pool", Spec("pool")),
)
kmeans_spec = HabitatSpec(
    name="fit_kmeans_elbow_silhouette",
    stages=shared_stages
    + (
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 4,
                    "validation": ["elbow", "silhouette"],
                    "n_init": 3,
                },
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
    ),
    random_seed=RANDOM_SEED,
)
gmm_spec = HabitatSpec(
    name="fit_gmm_bic",
    stages=shared_stages
    + (
        Stage(
            "fit",
            Spec(
                "gmm",
                {"min_habitats": 2, "max_habitats": 4, "validation": "bic", "n_init": 3},
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
    ),
    random_seed=RANDOM_SEED,
)
kmeans_result = Study(spec=kmeans_spec).fit_predict(cohort)
gmm_result = Study(spec=gmm_spec).fit_predict(cohort)


def _selection_row(name: str, fitter: str, result: StudyResult) -> Dict[str, Any]:
    """One row: selected K and scores at that K from ``selection_report``."""
    model = result.habitat_model
    report = (model.preprocessing_state or {}).get("selection_report") or {}
    selected = int(report.get("selected", model.n_habitats if model else 0))
    candidates = [int(v) for v in report.get("candidates", [])]
    row: Dict[str, Any] = {
        "spec_name": name,
        "fitter": fitter,
        "n_habitats": selected,
        "n_units": int(len(result.units[0].features)) if result.units else 0,
    }
    scores = report.get("scores") or {}
    if candidates and selected in candidates:
        idx = candidates.index(selected)
        for method, values in scores.items():
            if idx < len(values):
                row[f"{method}_at_k"] = float(values[idx])
    return row


fit_table = pd.DataFrame(
    [
        _selection_row(kmeans_spec.name, "kmeans", kmeans_result),
        _selection_row(gmm_spec.name, "gmm", gmm_result),
    ]
)
print(fit_table.to_string(index=False))
fit_table

# %%
# Match GMM ids onto k-means (subject 0), then Dice / ARI + overlay.
ref = np.asarray(kmeans_result.habitat_maps[0].label_array)
mov = np.asarray(gmm_result.habitat_maps[0].label_array)
mapping = match_labels_by_overlap(ref, mov)
aligned = remap_label_array(
    mov, mapping, reserved_ids=[int(v) for v in present_habitat_ids(ref)]
)
mapping_table = pd.DataFrame(
    [{"gmm_id": int(s), "kmeans_id": int(d)} for s, d in sorted(mapping.items(), key=lambda x: x[1])]
)
dice_table = pd.DataFrame(
    [
        {
            "kmeans_id": int(hid),
            "gmm_id": None if mid is None else int(mid),
            "dice": float(dice),
            "n_kmeans": int(n_ref),
            "n_gmm": int(n_mov),
        }
        for hid, mid, dice, n_ref, n_mov in habitat_dice_from_mapping(ref, mov, mapping)
    ]
)
summary = pd.DataFrame(
    [
        {
            "subject_id": cohort[0].subject_id,
            "mean_dice": float(dice_table["dice"].mean()) if len(dice_table) else float("nan"),
            "ari": float(adjusted_rand_index(ref, mov)),
            "n_kmeans": int(len(present_habitat_ids(ref))),
            "n_gmm": int(len(present_habitat_ids(mov))),
        }
    ]
)
print(mapping_table.to_string(index=False))
print(dice_table.to_string(index=False))
print(summary.to_string(index=False))
summary

# %%
fig_fit = plot_habitat_label_compare(
    cohort[0].image(ROI),
    kmeans_result.habitat_maps[0],
    aligned,
    titles=("k-means habitats", "GMM habitats (matched)"),
    align_labels=False,
)
fig_fit.savefig("out/choose_kmeans_vs_gmm.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Validation curves from each ``selection_report``.
kmeans_report = (kmeans_result.habitat_model.preprocessing_state or {}).get(
    "selection_report"
)
gmm_report = (gmm_result.habitat_model.preprocessing_state or {}).get("selection_report")
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
print("k-means methods:", (kmeans_report or {}).get("methods"))
print("GMM methods:", (gmm_report or {}).get("methods"))
