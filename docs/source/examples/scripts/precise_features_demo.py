"""Precise-feature screen demo: which voxel textures deserve to define habitats.

Runs the Prior et al. (2024) precision screen on one subject from
``demo_data/preprocessed`` with a small first-order + GLCM voxel-radiomics
set (repeatability only), then shows the two artefacts of the workflow:
the PreciseFeatureSet evidence table and the feature whitelist that
restricts a later habitat run to the precise set.

The scientific figure is ICC + 95% CI per texture feature; the overlay
is the habitat map after clustering only those precise features.
Per-perturbation anatomy figures live in
``precise_screening_tutorial_demo.py``.

Run from the repository root::

    python docs/source/examples/scripts/precise_features_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np

from habit import Cohort, HabitatSpec, Spec, Subject, cohort_from_directory
from habit.contracts import ArrayImageRef, Geometry
from habit.domain import PreciseFeatureSet
from habit.domain.voxel_features import VoxelRadiomicsFeatures
from habit.recipes import Study, identify_precise_voxel_features

# BEGIN example
# Change DATA / MODALITIES / ROI to your preprocessed layout
DATA = "demo_data/preprocessed"
MODALITIES = ("LAP",)
ROI = "LAP"

cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]


def _crop_to_roi(subject: Subject, modality: str, roi: str, pad: int = 8) -> Subject:
    """Crop one subject to the ROI bounding box plus pad (demo speed)."""
    mask = np.asarray(subject.mask(roi).data)
    image = np.asarray(subject.image(modality).data)
    nz = np.argwhere(mask > 0)
    lo = np.maximum(nz.min(axis=0) - pad, 0)
    hi = np.minimum(nz.max(axis=0) + pad + 1, mask.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    src = subject.image(modality).geometry
    geom = Geometry.from_array(
        image[sl].shape, spacing=src.spacing, direction=src.direction
    )
    return type(subject)(
        subject_id=subject.subject_id,
        images={modality: ArrayImageRef(array=image[sl], geometry=geom)},
        masks={roi: ArrayImageRef(array=mask[sl], geometry=geom)},
    )


cohort = Cohort(subjects=(_crop_to_roi(cohort[0], MODALITIES[0], ROI),))


def texture_params(bin_width: float) -> Dict[str, Any]:
    """Inline PyRadiomics settings: first-order + a few stable GLCM names."""
    return {
        "imageType": {"Original": {}},
        "featureClass": {
            "firstorder": ["Entropy", "Mean", "Variance", "Skewness", "Kurtosis"],
            "glcm": [
                "Contrast",
                "Correlation",
                "JointEntropy",
                "Idm",
                "DifferenceEntropy",
            ],
        },
        "setting": {"binWidth": float(bin_width), "normalize": False},
    }


def texture_factory(kernel_radius: int, bin_width: float) -> VoxelRadiomicsFeatures:
    """Map the settings grid onto a small voxel-texture extractor."""
    return VoxelRadiomicsFeatures(
        modalities=list(MODALITIES),
        kernel_radius=int(kernel_radius),
        params=texture_params(bin_width),
    )


print("=== precision screen (repeatability experiment) ===")
precise = identify_precise_voxel_features(
    cohort,
    extractor_factory=texture_factory,
    kernel_radii=(1,),  # one radius: skip the kernel-radius experiment
    bin_widths=(12,),  # one width: skip the bin-width experiment
    base_kernel_radius=1,
    base_bin_width=12,
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
    voxel_feature_extractor=Spec(
        "voxel_radiomics",
        {
            "modalities": list(MODALITIES),
            "kernel_radius": 1,
            "params": texture_params(12),
        },
    ),
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
from habit.viz import plot_habitat_overlay, plot_precision_icc

fig_icc = plot_precision_icc(
    evidence,
    lcl_threshold=precise.lcl_threshold,
    title="Precision screen: ICC and 95% CI",
)
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
