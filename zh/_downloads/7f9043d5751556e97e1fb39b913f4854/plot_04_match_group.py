"""
Group one-step fits: prototype matching
=======================================

Fit habitats **inside each subject** (one-step), then name the whole
group with :func:`~habit.precision.align_habitat_maps_to_prototypes` so
habitat ids mean the same prototype across people.

Two-step and pooled-voxel fits assign labels from one shared model, so
those habitat ids are already comparable across subjects and do **not**
need matching. Matching is for separate fits (two seeds, or one-step
per subject as here).

Subject-level winsorize 1 % then z-score; fixed ``n_habitats=3``. Small
demo cohort (three patients).
"""

# %%
# Load a small cohort
# -------------------
# sphinx_gallery_thumbnail_number = 2
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.execution import backend_from_policy
from habit.kernels import habitat_volume_fractions
from habit.precision import align_habitat_maps_to_prototypes
from habit.recipes import Study
from habit.spec import HabitatSpec, Spec, Stage
from habit.spec.policy import RunPolicy
from habit.viz import plot_habitat_overlay

# Change DATA / MODALITIES / ROI to your preprocessed layout.
DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:3]
Path("out").mkdir(exist_ok=True)

spec = HabitatSpec(
    name="group_one_step_match",
    stages=(
        Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
        Stage("preprocess", Spec("winsorize", {"winsor_limits": [0.01, 0.01]})),
        Stage("preprocess2", Spec("zscore")),
        Stage("fit", Spec("kmeans", {"n_habitats": 3, "n_init": 10})),
        Stage("assign", Spec("nearest_centroid")),
        Stage("volume", Spec("volume")),
    ),
    random_seed=0,
    pooling="none",
)
backend = backend_from_policy(RunPolicy(backend="serial", on_subject_failure="continue"))
result = Study(spec).fit_predict(cohort, backend=backend)

maps = list(result.habitat_maps)
models = [result.subject_models[sid] for sid in cohort.subject_ids]


def volume_row(habitat_map) -> dict:
    """Volume-fraction columns keyed by habitat id.

    Args:
        habitat_map: HabitatMap whose ids define the column names.

    Returns:
        Flat dict of ``habitat_<id>_volume_fraction`` values.
    """
    ids = tuple(int(h) for h in habitat_map.habitat_ids)
    frac = habitat_volume_fractions(habitat_map.label_array, ids)
    return {f"habitat_{hid}_volume_fraction": float(frac[hid]) for hid in ids}


# %%
# Before matching
# ---------------
print("features before matching:")
print(pd.DataFrame([volume_row(m) for m in maps], index=list(cohort.subject_ids)))

for subject, habitat_map in zip(cohort, maps):
    fig = plot_habitat_overlay(
        subject.image(ROI),
        habitat_map,
        title=f"{subject.subject_id}: before match",
        crop_to="labels",
    )
    fig.savefig(
        f"out/match_group_{subject.subject_id}_before.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

# %%
# Prototype-match the group
# -------------------------
aligned = align_habitat_maps_to_prototypes(maps, models=models)
maps_matched = list(aligned.habitat_maps)

print("features after prototype matching:")
print(
    pd.DataFrame(
        [volume_row(m) for m in maps_matched],
        index=list(cohort.subject_ids),
    )
)

for subject, habitat_map in zip(cohort, maps_matched):
    fig = plot_habitat_overlay(
        subject.image(ROI),
        habitat_map,
        title=f"{subject.subject_id}: after match",
        crop_to="labels",
    )
    fig.savefig(
        f"out/match_group_{subject.subject_id}_after.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()
