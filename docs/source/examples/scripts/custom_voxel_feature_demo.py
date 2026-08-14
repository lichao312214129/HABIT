#!/usr/bin/env python
"""
DIY voxel features: built-in ``expression`` formulas + a custom registry plugin.

Two complementary routes for formulas that ``raw`` / ``concat`` cannot express:

1. **Built-in ``expression``** — restricted arithmetic over modality intensities
   (ratios, powers, ``square`` / ``log`` / ...). Safe AST evaluation, no
   arbitrary Python.
2. **Custom plugin** — register any :class:`~habit.domain.protocols.VoxelFeatureExtractor`
   under ``habit.voxel_feature_extractor`` (decorator in-process, or an
   entry point in a third-party package). Use this when the formula needs
   neighbourhoods, learned embeddings, or logic beyond the expression DSL.

This script accompanies ``docs/source/examples/custom_voxel_features.rst``.

Run from the repository root::

    python docs/source/examples/scripts/custom_voxel_feature_demo.py
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from habit import HabitatSpec, Spec, make_synthetic_cohort
from habit.contracts import VoxelFeatureField
from habit.contracts.subject import Subject
from habit.domain.assembly import build_habitat_components
from habit.domain.voxel_features import (
    VoxelFeatureExtractorRegistry,
    aligned_image,
    build_voxel_field,
    roi_voxels,
)
from habit.spec.specs import Spec as ComponentSpec
import habit.recipes as recipes


# ---------------------------------------------------------------------------
# Custom plugin (in-process registration). A third-party package would instead
# declare in its pyproject.toml:
#
#   [project.entry-points."habit.voxel_feature_extractor"]
#   t1_t2_contrast = "my_pkg.features:register"
#
# where ``register()`` performs the decorator registration below, then call
# ``habit.load_plugins()`` before building the Spec.
# ---------------------------------------------------------------------------
@VoxelFeatureExtractorRegistry.register("t1_t2_contrast")
class T1T2Contrast:
    """
    Example DIY extractor: per-voxel ``(T1 - T2) / (T1 + T2 + eps)``.

    Mirrors the registry plugin pattern, targeting the
    :class:`~habit.domain.protocols.VoxelFeatureExtractor` protocol.
    """

    def __init__(
        self,
        modalities: Sequence[str] = ("T1", "T2"),
        roi: Optional[str] = None,
        eps: float = 1e-8,
    ) -> None:
        if len(modalities) != 2:
            raise ValueError("t1_t2_contrast expects exactly two modalities.")
        self.modalities: Tuple[str, ...] = tuple(modalities)
        self.roi = roi
        self.eps = float(eps)

    @property
    def spec(self) -> ComponentSpec:
        """Return the algorithm specification used for provenance."""
        return ComponentSpec(
            name="t1_t2_contrast",
            params={
                "modalities": list(self.modalities),
                "roi": self.roi,
                "eps": self.eps,
            },
        )

    def __call__(self, subject: Subject) -> VoxelFeatureField:
        """Compute the contrast feature inside the ROI."""
        mask, inside, voxel_index = roi_voxels(subject, self.roi)
        a = aligned_image(subject, self.modalities[0], mask, owner="t1_t2_contrast")
        b = aligned_image(subject, self.modalities[1], mask, owner="t1_t2_contrast")
        numerator = a[inside] - b[inside]
        denominator = a[inside] + b[inside] + self.eps
        values = np.asarray(numerator / denominator, dtype=np.float64).reshape(-1, 1)
        return build_voxel_field(
            subject, mask, voxel_index, ("t1_t2_contrast",), values, self.spec
        )


# BEGIN example
cohort = make_synthetic_cohort(n_subjects=3, shape=(14, 14, 14), rng=21)
subject = cohort[0]

# --- A. Built-in expression DSL ---------------------------------------------
expression_spec = HabitatSpec(
    name="expression_demo",
    # Runtime order: voxel features -> (no prep) -> supervoxels -> fit -> ...
    voxel_feature_extractor=Spec(
        "expression",
        {
            "features": {
                # User request: square(T1 / T2^3). ``^`` is accepted as power.
                "t1_over_t2_sq": "square(T1 / (T2 ^ 3 + eps))",
            },
        },
    ),
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 5, "n_init": 3}),
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "elbow", "n_init": 3},
    ),
    habitat_assigner=Spec("nearest_centroid"),
    habitat_features=(
        Spec("volume"),
        Spec("msi"),
        Spec("ith_score"),
        Spec("non_radiomics"),
        # Heavy PyRadiomics families (opt-in; require pyradiomics):
        # Spec("traditional"),
        # Spec("whole_habitat"),
        # Spec("each_habitat"),
    ),
    random_seed=21,
)

units = build_habitat_components(expression_spec).pipeline(assigner=None).units(subject)
print("=== expression: square(T1 / (T2^3 + eps)) ===")
print(f"  atomic features: {units.feature_frame().columns.tolist()}")
print(f"  n_voxels x n_features: {units.features.shape}")

result = recipes.Study(spec=expression_spec).fit_predict(cohort)
print(f"  batch: {len(result.habitat_maps)} maps, "
      f"{result.habitat_model.n_habitats} habitats")

# --- B. Custom registered extractor -----------------------------------------
custom_spec = HabitatSpec(
    name="custom_plugin_demo",
    voxel_feature_extractor=Spec(
        "t1_t2_contrast",
        {"modalities": ["T1", "T2"], "eps": 1e-8},
    ),
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 5, "n_init": 3}),
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "elbow", "n_init": 3},
    ),
    habitat_assigner=Spec("nearest_centroid"),
    habitat_features=(
        Spec("volume"),
        Spec("msi"),
        Spec("ith_score"),
        Spec("non_radiomics"),
        # Heavy PyRadiomics families (opt-in; require pyradiomics):
        # Spec("traditional"),
        # Spec("whole_habitat"),
        # Spec("each_habitat"),
    ),
    random_seed=21,
)

custom_units = (
    build_habitat_components(custom_spec).pipeline(assigner=None).units(subject)
)
print("\n=== custom plugin: t1_t2_contrast ===")
print(f"  registered names include: "
      f"{[n for n in VoxelFeatureExtractorRegistry.available() if 't1' in n or n == 'expression']}")
print(f"  atomic features: {custom_units.feature_frame().columns.tolist()}")
custom_result = recipes.Study(spec=custom_spec).fit_predict(cohort)
print(f"  batch: {len(custom_result.habitat_maps)} maps, "
      f"{custom_result.habitat_model.n_habitats} habitats")
# END example

# BEGIN figures
# Paste after the Script block. Uses subject and custom_result.
from pathlib import Path

from habit.viz import plot_habitat_overlay

Path("out").mkdir(exist_ok=True)
fig = plot_habitat_overlay(
    subject.image("T1"),
    custom_result.habitat_maps[0],
    title="habitats",
)
fig.savefig("out/custom_voxel_overlay.png", dpi=150, bbox_inches="tight")
print("Wrote out/custom_voxel_overlay.png")
# END figures

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _habitat_eye_check import eye_check_study

    eye_check_study(cohort, custom_result)
