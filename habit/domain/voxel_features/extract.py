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
"""Volume-level voxel-texture extraction atom (L3).

:func:`extract_voxel_texture` is the single-volume entry point for
per-voxel radiomics. Callers pass one image, one mask, and the paper
knobs (kernel radius, bin width, feature classes) -- no
:class:`~habit.contracts.subject.Cohort`, YAML, or precision recipe.
Combinations such as R1 vs R3 or B12 vs B25 are two calls of this
function on the same volumes.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np

from habit.api.image import ImageVolume, MaskVolume
from habit.contracts.geometry import Geometry
from habit.contracts.habitat import VoxelFeatureField
from habit.contracts.image import ArrayImageRef
from habit.contracts.subject import Subject
from habit.domain.voxel_features.voxel_radiomics import (
    DEFAULT_VOXEL_BATCH,
    VoxelRadiomicsFeatures,
)
from habit.exceptions import HABITAPIError

__all__ = ["extract_voxel_texture"]

_DEFAULT_MODALITY = "image"
_DEFAULT_ROI = "roi"


def _geometry_of(volume: ImageVolume) -> Geometry:
    """
    Build a :class:`Geometry` from a public or contracts volume.

    Args:
        volume: Intensity or mask volume with spacing / origin / direction.

    Returns:
        Geometry describing ``volume.data``.
    """
    existing = getattr(volume, "geometry", None)
    if isinstance(existing, Geometry):
        return existing
    return Geometry(
        shape=tuple(int(v) for v in volume.data.shape),
        spacing=tuple(float(v) for v in volume.spacing),
        origin=tuple(float(v) for v in volume.origin),
        direction=tuple(float(v) for v in volume.direction),
    )


def _subject_from_volumes(
    image: ImageVolume,
    mask: MaskVolume,
    *,
    modality: str,
    roi: str,
) -> Subject:
    """
    Wrap one image and mask as a single-modality Subject.

    Args:
        image: Intensity volume.
        mask: ROI mask on the same grid.
        modality: Synthetic image key inside the subject.
        roi: Synthetic mask key inside the subject.

    Returns:
        An in-memory subject the voxel-radiomics extractor can call.
    """
    geometry = _geometry_of(image)
    return Subject(
        subject_id=str(image.subject_id or "image"),
        images={
            modality: ArrayImageRef(array=np.asarray(image.data), geometry=geometry)
        },
        masks={
            roi: ArrayImageRef(array=np.asarray(mask.data), geometry=_geometry_of(mask))
        },
    )


def _params_with_bin_width(
    bin_width: float,
    *,
    feature_classes: Optional[Mapping[str, Sequence[str]]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build in-memory PyRadiomics settings with a fixed bin width.

    Args:
        bin_width: Discretisation bin width (the paper's ``B``).
        feature_classes: Optional ``{class_name: [feature, ...]}`` map
            (PyRadiomics ``featureClass``).
        params: Optional full settings mapping. Mutually exclusive with
            ``feature_classes``.

    Returns:
        Settings dict with ``setting.binWidth`` set to ``bin_width``.

    Raises:
        HABITAPIError: When both ``feature_classes`` and ``params`` are set.
    """
    if feature_classes is not None and params is not None:
        raise HABITAPIError(
            "extract_voxel_texture: pass feature_classes or params, not both."
        )
    if params is not None:
        built = dict(params)
        setting = dict(built.get("setting") or {})
        setting["binWidth"] = float(bin_width)
        built["setting"] = setting
        return built
    if feature_classes is not None:
        return {
            "imageType": {"Original": {}},
            "featureClass": {str(key): list(values) for key, values in feature_classes.items()},
            "setting": {"binWidth": float(bin_width), "normalize": False},
        }
    # Default: bundled voxel preset (paper CT habitat setting) with this B.
    from habit.utils.radiomics_params_utils import load_radiomics_params_yaml
    from habit.utils.radiomics_preset_utils import get_preset_path

    built = dict(load_radiomics_params_yaml(get_preset_path("voxel")))
    setting = dict(built.get("setting") or {})
    setting["binWidth"] = float(bin_width)
    built["setting"] = setting
    return built


def extract_voxel_texture(
    image: ImageVolume,
    mask: MaskVolume,
    *,
    kernel_radius: int = 3,
    bin_width: float = 12.0,
    feature_classes: Optional[Mapping[str, Sequence[str]]] = None,
    params: Optional[Dict[str, Any]] = None,
    voxel_batch: int = DEFAULT_VOXEL_BATCH,
    use_torch_radiomics: Union[str, bool] = "auto",
    use_gpu_matrices: Union[str, bool] = "auto",
) -> VoxelFeatureField:
    """
    Extract a per-voxel texture table from one image and one mask.

    This is the atomic teaching / experiment call. Paper combinations are
    repeated calls on the same volumes with different knobs, for example
    ``kernel_radius=1`` vs ``kernel_radius=3`` (R1 vs R3) or
    ``bin_width=12`` vs ``bin_width=25`` (B12 vs B25). The returned
    :class:`~habit.contracts.habitat.VoxelFeatureField` is the input to
    :func:`~habit.domain.precision.precision_panel`.

    Args:
        image: Intensity volume.
        mask: ROI mask; one row is emitted per foreground voxel.
        kernel_radius: Neighbourhood radius in voxels (the paper's ``R``).
            Radius 1 is a 3x3x3 cube; radius 3 is 7x7x7.
        bin_width: Grey-level discretisation width (the paper's ``B``).
        feature_classes: Optional PyRadiomics class-to-names map, e.g.
            ``{"firstorder": ["Entropy", "Mean"], "glcm": ["Contrast"]}``.
            When omitted (and ``params`` is omitted) the bundled voxel
            preset is used.
        params: Full in-memory PyRadiomics settings. Mutually exclusive
            with ``feature_classes``. ``bin_width`` still overwrites
            ``setting.binWidth``.
        voxel_batch: ROI voxels processed per batch.
        use_torch_radiomics: ``"auto"``, ``True``, or ``False``.
        use_gpu_matrices: ``"auto"``, ``True``, or ``False`` -- build the
            TorchRadiomics texture matrices on GPU (bit-identical counts).

    Returns:
        One row per ROI voxel, one column per enabled feature. Provenance
        carries the extractor :class:`~habit.spec.specs.Spec`.

    Raises:
        HABITAPIError: When ``feature_classes`` and ``params`` are both
            set, or ``kernel_radius`` is not positive.
    """
    if kernel_radius < 1:
        raise HABITAPIError(
            f"extract_voxel_texture: kernel_radius must be positive; "
            f"got {kernel_radius}."
        )
    built = _params_with_bin_width(
        bin_width, feature_classes=feature_classes, params=params
    )
    modality = str(image.modality or _DEFAULT_MODALITY)
    roi = str(getattr(mask, "modality", None) or _DEFAULT_ROI)
    extractor = VoxelRadiomicsFeatures(
        modalities=[modality],
        roi=roi,
        params=built,
        kernel_radius=int(kernel_radius),
        voxel_batch=int(voxel_batch),
        use_torch_radiomics=use_torch_radiomics,
        use_gpu_matrices=use_gpu_matrices,
    )
    subject = _subject_from_volumes(image, mask, modality=modality, roi=roi)
    return extractor(subject)
