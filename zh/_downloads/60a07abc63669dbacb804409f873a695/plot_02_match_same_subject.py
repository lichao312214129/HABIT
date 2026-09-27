"""
Same subject, two fits: prototype matching
==========================================

Independent fits of the **same** patient permute habitat ids. Change
only ``random_seed``, then name both maps with
:func:`~habit.precision.align_habitat_maps_to_prototypes` (centroid
prototypes). Show overlays and volume-fraction columns before and after.

Two-step and pooled-voxel fits already share one model: their habitat
ids are comparable across subjects and do **not** need matching.
Matching is for separate fits (two seeds here, or one-step per subject).

Subject-level winsorize 1 % then z-score; fixed ``n_habitats=3``.
"""

# %%
# Load one subject
# ----------------
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
from habit.viz import plot_habitat_label_compare, plot_habitat_overlay

# Change DATA / MODALITIES / ROI to your preprocessed layout.
DATA = fetch_demo()
MODALITIES = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
subject = cohort[0]
image = subject.image(ROI)
Path("out").mkdir(exist_ok=True)


def one_step_spec(seed: int) -> HabitatSpec:
    """One-step k-means on this subject with a fixed habitat count.

    Args:
        seed: Spec ``random_seed`` (changes centroid init only).

    Returns:
        HabitatSpec ready for ``Study(...).fit_predict``.
    """
    return HabitatSpec(
        name=f"same_subject_seed_{seed}",
        stages=(
            Stage("extract", Spec("raw", {"modalities": list(MODALITIES), "roi": ROI})),
            Stage("preprocess", Spec("winsorize", {"winsor_limits": [0.01, 0.01]})),
            Stage("preprocess2", Spec("zscore")),
            Stage("fit", Spec("kmeans", {"n_habitats": 3, "n_init": 10})),
            Stage("assign", Spec("nearest_centroid")),
            Stage("volume", Spec("volume")),
        ),
        random_seed=seed,
        pooling="none",
    )


def volume_row(habitat_map) -> dict:
    """Volume-fraction columns keyed by habitat id (label-dependent).

    Args:
        habitat_map: HabitatMap whose ids define the column names.

    Returns:
        Flat dict of ``habitat_<id>_volume_fraction`` values.
    """
    ids = tuple(int(h) for h in habitat_map.habitat_ids)
    frac = habitat_volume_fractions(habitat_map.label_array, ids)
    return {f"habitat_{hid}_volume_fraction": float(frac[hid]) for hid in ids}


backend = backend_from_policy(RunPolicy(backend="serial", on_subject_failure="continue"))

# %%
# Fit twice with different seeds
# ------------------------------
# Same Spec and data; only the seed changes.
result_a = Study(one_step_spec(0)).fit_predict(cohort, backend=backend)
result_b = Study(one_step_spec(1)).fit_predict(cohort, backend=backend)
sid = subject.subject_id
map_a = result_a.habitat_maps[0]
map_b = result_b.habitat_maps[0]
model_a = result_a.subject_models[sid]
model_b = result_b.subject_models[sid]

print("features before matching (seed 0 / seed 1):")
print(pd.DataFrame([volume_row(map_a), volume_row(map_b)], index=["seed_0", "seed_1"]))

# Same anatomy under both label maps (before matching).
fig = plot_habitat_label_compare(
    image,
    map_a,
    map_b,
    titles=("seed 0 (before)", "seed 1 (before)"),
    crop_to="labels",
)
fig.savefig("out/match_same_subject_before.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Prototype-match both fits
# -------------------------
# Shared prototypes from clustering centroids (not voxel-overlap
# Hungarian alone). After matching, the same id names the same prototype.
aligned = align_habitat_maps_to_prototypes(
    [map_a, map_b], models=[model_a, model_b]
)
map_a_m, map_b_m = aligned.habitat_maps
print("features after prototype matching:")
print(
    pd.DataFrame(
        [volume_row(map_a_m), volume_row(map_b_m)],
        index=["seed_0_matched", "seed_1_matched"],
    )
)

# Same anatomy again after prototype matching.
fig = plot_habitat_label_compare(
    image,
    map_a_m,
    map_b_m,
    titles=("seed 0 (after)", "seed 1 (after)"),
    crop_to="labels",
)
fig.savefig("out/match_same_subject_after.png", dpi=150, bbox_inches="tight")
plt.show()

# Single-map overlays on the same image (matched ids only).
for one_map, title, stem in (
    (map_a_m, f"{sid}: seed 0 after match", "match_same_subject_overlay_seed0"),
    (map_b_m, f"{sid}: seed 1 after match", "match_same_subject_overlay_seed1"),
):
    fig = plot_habitat_overlay(
        image, one_map, title=title, crop_to="labels"
    )
    fig.savefig(f"out/{stem}.png", dpi=150, bbox_inches="tight")
    plt.show()
