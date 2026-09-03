"""
Custom features
===============

When ``raw`` / ``concat`` are not enough — for example
``square(LAP / PVP^3)``, or a neighbourhood / embedding feature — use a
built-in ``expression`` or a registered voxel extractor.

Both plug into ``HabitatSpec.voxel_feature_extractor`` and the same recipes.
"""

# %%
# Register an in-process plugin. A third-party package would instead
# declare an entry point under ``habit.voxel_feature_extractor``.
from pathlib import Path
import os
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import VoxelFeatureField, cohort_from_directory
from habit.contracts.subject import Subject
from habit.datasets import fetch_demo
from habit.pipeline.assembly import build_habitat_components
from habit.spec import HabitatSpec, Spec
from habit.spec.specs import Spec as ComponentSpec
from habit.viz import plot_habitat_overlay
from habit.voxel_features import (
    VoxelFeatureExtractorRegistry,
    aligned_image,
    build_voxel_field,
    roi_voxels,
)
import habit.recipes as recipes


@VoxelFeatureExtractorRegistry.register("t1_t2_contrast")
class T1T2Contrast:
    """Per-voxel ``(T1 - T2) / (T1 + T2 + eps)`` (here LAP vs PVP)."""

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


DATA = fetch_demo()
MODALITIES = ("LAP", "PVP")
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:2]
subject = cohort[0]
print(f"Cohort: {len(cohort)} subjects -> {list(cohort.subject_ids)}")

# %%
# Built-in ``expression`` DSL: restricted arithmetic, no arbitrary Python.
# Print the voxel-feature head so the formula is a visible column.
expression_spec = HabitatSpec(
    name="expression_demo",
    voxel_feature_extractor=Spec(
        "expression",
        {
            "features": {
                "lap_over_pvp_sq": "square(LAP / (PVP ^ 3 + eps))",
            },
        },
    ),
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 5, "n_init": 3}),
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "elbow", "n_init": 3},
    ),
    habitat_assigner=Spec("nearest_centroid"),
    habitat_features=(Spec("volume"), Spec("msi"), Spec("ith_score")),
    random_seed=21,
)
expr_field = (
    build_habitat_components(expression_spec)
    .pipeline(assigner=None)
    .voxel_feature_extractor(subject)
)
print("expression feature table:")
print(expr_field.feature_frame().head())
expr_field.feature_frame().head()

# %%
# Fit habitats from the expression field and overlay.
expr_result = recipes.Study(spec=expression_spec).fit_predict(cohort)
fig = plot_habitat_overlay(
    subject.image("LAP"),
    expr_result.habitat_maps[0],
    title="habitats (expression)",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/custom_voxel_overlay.png", dpi=150, bbox_inches="tight")
if os.environ.get("HABIT_NO_VIEW") != "1":
    plt.show()

# %%
# The registered ``t1_t2_contrast`` plugin, called as a ``Spec`` name.
plugin_spec = HabitatSpec(
    name="plugin_demo",
    voxel_feature_extractor=Spec(
        "t1_t2_contrast",
        {"modalities": list(MODALITIES), "roi": ROI},
    ),
    supervoxelizer=Spec("kmeans", {"n_supervoxels": 5, "n_init": 3}),
    habitat_model_fitter=Spec(
        "kmeans",
        {"min_habitats": 2, "max_habitats": 3, "validation": "elbow", "n_init": 3},
    ),
    habitat_assigner=Spec("nearest_centroid"),
    habitat_features=(Spec("volume"),),
    random_seed=21,
)
plugin_field = (
    build_habitat_components(plugin_spec)
    .pipeline(assigner=None)
    .voxel_feature_extractor(subject)
)
print("plugin feature table:")
print(plugin_field.feature_frame().head())
plugin_field.feature_frame().head()
