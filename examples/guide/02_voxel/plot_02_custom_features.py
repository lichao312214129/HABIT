"""
Custom features
===============

Three liver DCE maps: arterial relative enhancement, arterial-to-portal
wash-out, and arterial-to-delayed wash-out.
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from habit.contracts import VoxelFeatureField, cohort_from_directory
from habit.contracts.subject import Subject
from habit.datasets import fetch_demo
from habit.feature_preprocessing import ZScoreScaling
from habit.spec import HabitatSpec, Spec, Stage
from habit.spec.specs import Spec as ComponentSpec
from habit.viz import plot_habitat_overlay
from habit.voxel_features import (
    VoxelFeatureExtractorRegistry,
    aligned_image,
    build_voxel_field,
    roi_voxels,
)
import habit.recipes as recipes

# Unenhanced, late-arterial, portal-venous, delayed (demo pack keys).
PHASES: Tuple[str, ...] = ("pre_contrast", "LAP", "PVP", "delay_3min")
ROI = "LAP"
DCE_FEATURES = {
    "relative_enhancement_lap": "(LAP - pre_contrast) / (pre_contrast + eps)",
    "relative_washout_pvp": "(LAP - PVP) / (LAP - pre_contrast + eps)",
    "relative_washout_delay": "(LAP - delay_3min) / (LAP - pre_contrast + eps)",
}


@VoxelFeatureExtractorRegistry.register("dce_hemodynamics")
class DCEHemodynamics:
    """Per-voxel DCE maps from four liver phases."""

    def __init__(
        self,
        phases: Sequence[str] = PHASES,
        roi: Optional[str] = None,
        eps: float = 1e-8,
    ) -> None:
        if len(phases) != 4:
            raise ValueError(
                "dce_hemodynamics expects four phases: unenhanced, "
                "arterial, portal, delayed."
            )
        self.phases: Tuple[str, ...] = tuple(phases)
        self.roi = roi
        self.eps = float(eps)

    @property
    def spec(self) -> ComponentSpec:
        """Return the algorithm specification used for provenance."""
        return ComponentSpec(
            name="dce_hemodynamics",
            params={
                "phases": list(self.phases),
                "roi": self.roi,
                "eps": self.eps,
            },
        )

    def __call__(self, subject: Subject) -> VoxelFeatureField:
        """Compute the three DCE columns inside the ROI."""
        mask, inside, voxel_index = roi_voxels(subject, self.roi)
        owner = "dce_hemodynamics"
        pre, lap, pvp, delay = (
            aligned_image(subject, phase, mask, owner=owner)[inside]
            for phase in self.phases
        )
        values = np.column_stack(
            [
                (lap - pre) / (pre + self.eps),
                (lap - pvp) / (lap - pre + self.eps),
                (lap - delay) / (lap - pre + self.eps),
            ]
        ).astype(np.float64, copy=False)
        return build_voxel_field(
            subject,
            mask,
            voxel_index,
            tuple(DCE_FEATURES),
            values,
            self.spec,
        )


DATA = fetch_demo()
cohort = cohort_from_directory(DATA, modalities=PHASES, roi=ROI)[:2]
subject = cohort[0]

before = DCEHemodynamics(phases=PHASES, roi=ROI)(subject).feature_frame()
print("before zscore (mean / std):")
print(before.agg(["mean", "std"]).round(4))
print(before.head())
scaler = ZScoreScaling(across_features=False)
after = scaler.transform(before, scaler.fit(before))
print("after zscore (mean / std):")
print(after.agg(["mean", "std"]).round(4))
print(after.head())

spec = HabitatSpec(
    name="dce_hemodynamics_demo",
    stages=(
        Stage(
            "extract_voxel_features",
            Spec("dce_hemodynamics", {"phases": list(PHASES), "roi": ROI}),
        ),
        Stage("preprocess", Spec("zscore", {"across_features": False})),
        Stage(
            "fit",
            Spec(
                "kmeans",
                {
                    "min_habitats": 2,
                    "max_habitats": 5,
                    "validation": "elbow",
                    "n_init": 3,
                },
            ),
        ),
        Stage("assign", Spec("nearest_centroid")),
        Stage("quantify", Spec("volume")),
    ),
    random_seed=21,
)
result = recipes.Study(spec=spec).fit_predict(cohort)
for sid, model in result.subject_models.items():
    print(f"{sid}: n_habitats={model.n_habitats}")
labels = np.unique(result.habitat_maps[0].label_array)
print("overlay labels", [int(v) for v in labels if v > 0])
fig = plot_habitat_overlay(
    subject.image("LAP"),
    result.habitat_maps[0],
    title="habitats",
)
Path("out").mkdir(exist_ok=True)
fig.savefig("out/custom_voxel_overlay.png", dpi=150, bbox_inches="tight")
plt.show()
