# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""L4 recipe: the precise-feature screen of Prior et al. 2024.

Assembles the L3 precision components into the published study design
(Radiol Artif Intell 2024;6(2):e230118): per subject, voxel features are
extracted at the base setting (R3B12), on a simulated-retest copy of the
image, and at the alternative kernel radii (R1) and bin widths (B25); the
three experiments yield per-subject ICC panels whose cohort medians must
all clear the LCL threshold for a feature to be called *precise*.

The recipe holds no engine of its own: extraction is the registered voxel
feature extractor, perturbation is the registered perturbation chain, and
the statistics live in :mod:`habit.domain.precision`. Everything is
in-memory; the returned :class:`PreciseFeatureSet` is the serialisable
artefact a habitat study publishes so others cluster the SAME features.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from habit.contracts.subject import Cohort, Subject
from habit.domain.precision import (
    GaussianNoisePerturbation,
    PerturbationChain,
    PreciseFeatureSet,
    RotationPerturbation,
    TranslationPerturbation,
    aggregate_panels,
    identify_precise_features,
    precision_panel,
)
from habit.domain.protocols import ImagePerturbation, VoxelFeatureExtractor
from habit.domain.voxel_features import VoxelRadiomicsFeatures
from habit.exceptions import HABITAPIError
from habit.spec.specs import Spec
from habit.utils.progress_utils import CustomTqdm

__all__ = ["identify_precise_voxel_features", "voxel_radiomics_factory"]

#: Experiment names used in the result's ``panels`` / ``experiments``.
EXPERIMENT_REPEATABILITY = "repeatability"
EXPERIMENT_KERNEL_RADIUS = "reproducibility_kernel_radius"
EXPERIMENT_BIN_WIDTH = "reproducibility_bin_width"


def voxel_radiomics_factory(
    kernel_radius: int, bin_width: float
) -> VoxelRadiomicsFeatures:
    """
    Build the default extractor: the bundled voxel preset at one grid point.

    The preset is the CT habitat setting (R3B12, 21 stable GLCM features);
    only ``binWidth`` varies across the grid, exactly as the paper's
    ``ROI_R*r*B*.yaml`` files do.

    Args:
        kernel_radius: Neighbourhood radius in voxels (the ``R`` of the
            paper's settings grid).
        bin_width: Discretisation bin width in HU (the ``B``).

    Returns:
        The configured voxel radiomics extractor.
    """
    from habit.utils.radiomics_params_utils import load_radiomics_params_yaml
    from habit.utils.radiomics_preset_utils import get_preset_path

    params = load_radiomics_params_yaml(get_preset_path("voxel"))
    params.setdefault("setting", {})["binWidth"] = bin_width
    return VoxelRadiomicsFeatures(kernel_radius=kernel_radius, params=params)


def _default_perturbation() -> PerturbationChain:
    """Return the paper's simulated-retest chain: noise + shift + rotation."""
    return PerturbationChain(
        [
            GaussianNoisePerturbation(),
            TranslationPerturbation(),
            RotationPerturbation(),
        ]
    )


def identify_precise_voxel_features(
    cohort: Cohort,
    *,
    extractor_factory: Optional[Callable[[int, float], VoxelFeatureExtractor]] = None,
    kernel_radii: Sequence[int] = (1, 3),
    bin_widths: Sequence[float] = (12, 25),
    base_kernel_radius: int = 3,
    base_bin_width: float = 12,
    perturbation: Optional[ImagePerturbation] = None,
    lcl_threshold: float = 0.5,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    alpha: float = 0.05,
    min_voxels: int = 10,
    seed: int = 0,
    show_progress: bool = True,
) -> PreciseFeatureSet:
    """
    Identify the voxel features precise enough to define habitats.

    Per subject, up to three experiments are run:

    * ``repeatability`` -- ICC(3A,1) between the base-setting feature maps
      of the original and of one perturbed (simulated retest) image;
    * ``reproducibility_kernel_radius`` -- ICC(3C,1) between the feature
      maps at ``kernel_radii`` with the bin width fixed at
      ``base_bin_width`` (skipped when fewer than two radii are given);
    * ``reproducibility_bin_width`` -- ICC(3C,1) between the feature maps
      at ``bin_widths`` with the radius fixed at ``base_kernel_radius``
      (skipped when fewer than two widths are given).

    Per-feature per-subject ICCs are aggregated by the cohort median (the
    paper's aggregation), and a feature is precise when its median LCL
    reaches ``lcl_threshold`` in EVERY experiment, subject to the
    ``include`` / ``exclude`` expert overrides (the paper used ``include``
    for NGTDM Coarseness).

    Args:
        cohort: Subjects to screen on; their ROIs define the voxel pools.
        extractor_factory: ``(kernel_radius, bin_width) -> extractor``;
            ``None`` selects :func:`voxel_radiomics_factory` (the bundled
            CT preset). Custom extractors need a factory mapping the grid
            point onto their own settings.
        kernel_radii: Reproducibility grid of neighbourhood radii; the
            paper contrasts R1 with R3.
        bin_widths: Reproducibility grid of bin widths; the paper
            contrasts B12 with B25.
        base_kernel_radius: Radius of the base (repeatability) setting.
        base_bin_width: Bin width of the base (repeatability) setting.
        perturbation: Simulated-retest perturbation; ``None`` selects the
            paper's chain (Chang-estimated Gaussian noise, sub-voxel
            translation up to one voxel, 0.5-degree in-plane rotation).
        lcl_threshold: Lower-confidence-limit cutoff; ``0.5`` is the
            paper's "at least good" boundary.
        include: Expert overrides added regardless of the criteria.
        exclude: Features removed regardless of the criteria.
        alpha: Two-sided significance level of the confidence limits.
        min_voxels: Minimum paired-voxel count per subject; below it a
            subject's feature is unmeasurable and does not veto the median.
        seed: Master seed; each subject's perturbation draws from its own
            spawned child sequence, so the screen is fully reproducible.
        show_progress: Show a progress bar over the cohort.

    Returns:
        The precise feature set with the cohort-level evidence panels.

    Raises:
        HABITAPIError: If the cohort is empty.
    """
    if len(cohort) == 0:
        raise HABITAPIError("identify_precise_voxel_features: the cohort is empty.")
    factory = extractor_factory or voxel_radiomics_factory
    chain = perturbation if perturbation is not None else _default_perturbation()
    radii = tuple(int(r) for r in kernel_radii)
    widths = tuple(float(b) for b in bin_widths)

    base_extractor = factory(base_kernel_radius, base_bin_width)
    # Grid extractors are built once; the base setting is reused whenever it
    # sits on a grid point so no extraction runs twice.
    radius_extractors: Dict[int, VoxelFeatureExtractor] = {
        r: (base_extractor if r == base_kernel_radius else factory(r, base_bin_width))
        for r in radii
    }
    width_extractors: Dict[float, VoxelFeatureExtractor] = {
        b: (base_extractor if b == base_bin_width else factory(base_kernel_radius, b))
        for b in widths
    }

    child_seeds = np.random.SeedSequence(seed).spawn(len(cohort))
    repeat_panels: List = []
    radius_panels: List = []
    width_panels: List = []
    subjects: List[Subject] = list(cohort)
    for index, subject in enumerate(
        CustomTqdm(
            subjects,
            total=len(subjects),
            desc="Precision screen",
            disable=not show_progress,
        )
    ):
        rng = np.random.default_rng(child_seeds[index])
        base_field = base_extractor(subject)
        perturbed_field = base_extractor(chain(subject, rng=rng))
        repeat_panels.append(
            precision_panel(
                {"original": base_field, "perturbed": perturbed_field},
                agreement="absolute",
                alpha=alpha,
                min_voxels=min_voxels,
            )
        )
        if len(radii) >= 2:
            conditions = {
                f"R{r}": (
                    base_field if r == base_kernel_radius else radius_extractors[r](subject)
                )
                for r in radii
            }
            radius_panels.append(
                precision_panel(
                    conditions,
                    agreement="consistency",
                    alpha=alpha,
                    min_voxels=min_voxels,
                )
            )
        if len(widths) >= 2:
            conditions = {
                f"B{b:g}": (
                    base_field if b == base_bin_width else width_extractors[b](subject)
                )
                for b in widths
            }
            width_panels.append(
                precision_panel(
                    conditions,
                    agreement="consistency",
                    alpha=alpha,
                    min_voxels=min_voxels,
                )
            )

    experiments = {EXPERIMENT_REPEATABILITY: aggregate_panels(repeat_panels)}
    if radius_panels:
        experiments[EXPERIMENT_KERNEL_RADIUS] = aggregate_panels(radius_panels)
    if width_panels:
        experiments[EXPERIMENT_BIN_WIDTH] = aggregate_panels(width_panels)
    result = identify_precise_features(
        experiments,
        lcl_threshold=lcl_threshold,
        include=include,
        exclude=exclude,
    )
    recipe_spec = Spec(
        name="identify_precise_voxel_features",
        params={
            "extractor_factory": getattr(factory, "__name__", str(factory)),
            "kernel_radii": list(radii),
            "bin_widths": list(widths),
            "base_kernel_radius": base_kernel_radius,
            "base_bin_width": base_bin_width,
            "perturbation": chain.spec.to_dict(),
            "lcl_threshold": float(lcl_threshold),
            "include": list(include),
            "exclude": list(exclude),
            "alpha": float(alpha),
            "min_voxels": int(min_voxels),
            "n_subjects": len(cohort),
        },
    )
    provenance = result.provenance.derive(
        produced_by="recipe.identify_precise_voxel_features",
        spec_fingerprint=recipe_spec.fingerprint(),
        random_seed=int(seed),
    )
    return dataclasses.replace(result, provenance=provenance)
