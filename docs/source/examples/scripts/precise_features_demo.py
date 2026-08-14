"""Precise-feature screen demo: which voxel features deserve to define habitats.

Runs the Prior et al. (2024) precision screen on one subject from
``demo_data/preprocessed`` with fast raw-intensity features (so the demo
needs no PyRadiomics), then shows the two artefacts of the workflow: the
PreciseFeatureSet evidence table and the feature whitelist that restricts
a later habitat run to the precise set.

The scientific figure is the ICC lower-confidence-limit panel; the overlay
is the habitat map after clustering only those precise features.
Per-perturbation anatomy figures live in
``precise_screening_tutorial_demo.py``.

Run from the repository root::

    python docs/source/examples/scripts/precise_features_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from habit import HabitatSpec, Spec, cohort_from_directory
from habit.domain import PreciseFeatureSet, RawVoxelFeatures
from habit.recipes import Study, identify_precise_voxel_features

# BEGIN example
# Change DATA / MODALITIES / ROI to your preprocessed layout
DATA = "demo_data/preprocessed"
MODALITIES = ("LAP",)
ROI = "LAP"

cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]


def raw_factory(kernel_radius: int, bin_width: float) -> RawVoxelFeatures:
    """Map the settings grid onto an extractor.

    Raw intensities have no kernel-radius or bin-width setting, so the grid
    point is ignored; with voxel radiomics this is where ``kernel_radius``
    and ``bin_width`` would be forwarded (the default factory does exactly
    that with the bundled CT preset).
    """
    return RawVoxelFeatures(modalities=list(MODALITIES))


print("=== precision screen (repeatability experiment) ===")
precise = identify_precise_voxel_features(
    cohort,
    extractor_factory=raw_factory,
    kernel_radii=(3,),  # one radius: skip the kernel-radius experiment
    bin_widths=(12,),  # one width: skip the bin-width experiment
    seed=7,
    show_progress=False,
)
evidence = precise.to_frame().round(3)
screened = list(precise.feature_names)
all_names = list(dict.fromkeys(evidence["feature"].astype(str).tolist()))
unstable = [name for name in all_names if name not in set(screened)]
print(f"  precise (stable) features: {screened}")
print(f"  unstable features: {unstable}")
print(evidence.to_string(index=False))

print("\n=== whitelist bridge into a habitat spec ===")
whitelist = precise.preprocessor()
print(f"  whitelist spec: {whitelist.spec.to_dict()}")
spec = HabitatSpec(
    name="precise_demo",
    voxel_feature_extractor=Spec("raw", {"modalities": list(MODALITIES)}),
    # The whitelist goes FIRST: only precise features reach scaling/clustering.
    voxel_feature_preprocessors=(
        whitelist.spec,
        Spec("minmax", {"across_features": False}),
    ),
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 6, "n_init": 3}),
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "elbow", "n_init": 3},
    ),
    habitat_assigner=Spec("nearest_centroid"),
    random_seed=11,
)
print(f"  habitat spec fingerprint: {spec.fingerprint()}")

result = Study(spec).fit_predict(cohort)
print(f"  habitat maps: {len(result.habitat_maps)}")

print("\n=== artefact round trip ===")
with tempfile.TemporaryDirectory(prefix="habit_precise_") as tmp:
    path = precise.save(Path(tmp) / "precise_features.json")
    reloaded = PreciseFeatureSet.load(path)
    print(f"  saved {path.name}; reloaded features: {list(reloaded.feature_names)}")
# END example

# BEGIN figures
# Paste after the Script block. Uses precise, evidence, cohort, result, MODALITIES.
from habit.viz import plot_habitat_overlay

# Evidence figure: ICC and LCL per feature, coloured by precise vs unstable.
fig_icc, ax = plt.subplots(figsize=(6.2, 3.2))
x = np.arange(len(evidence))
width = 0.35
precise_flag = evidence["precise"].to_numpy(dtype=bool)
icc_colors = np.where(precise_flag, "#0072B2", "#9AA0A6")
lcl_colors = np.where(precise_flag, "#56B4E9", "#D0D3D8")
ax.bar(x - width / 2, evidence["value"], width, color=icc_colors, label="ICC")
ax.bar(x + width / 2, evidence["lcl"], width, color=lcl_colors, label="LCL")
ax.axhline(
    precise.lcl_threshold,
    color="0.25",
    linestyle="--",
    linewidth=1.2,
    label=f"LCL threshold ({precise.lcl_threshold})",
)
ax.set_xticks(list(x))
ax.set_xticklabels(
    [
        f"{row.feature}\n{row.experiment}\n"
        f"{'precise' if row.precise else 'unstable'}"
        for row in evidence.itertuples()
    ],
    fontsize=8,
)
ax.set_ylabel("ICC")
ax.set_ylim(0.0, 1.05)
ax.set_title("Precision screen: precise vs unstable")
ax.legend(frameon=False, loc="lower right")
fig_icc.tight_layout()
Path("out").mkdir(exist_ok=True)
fig_icc.savefig("out/precise_features_icc_lcl.png", dpi=150, bbox_inches="tight")

fig_overlay = plot_habitat_overlay(
    cohort[0].image(MODALITIES[0]),
    result.habitat_maps[0],
    title="Habitats after precise-feature whitelist",
)
fig_overlay.savefig("out/precise_features_overlay.png", dpi=150, bbox_inches="tight")
print("Wrote out/precise_features_icc_lcl.png and out/precise_features_overlay.png")
# END figures

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import copy_out_figures_to_gallery

    copy_out_figures_to_gallery(
        ("precise_features_icc_lcl.png", "precise_features_overlay.png")
    )
