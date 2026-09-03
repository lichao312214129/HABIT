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
"""Shared PyRadiomics machinery for the radiomics habitat feature families.

The three radiomics families (``traditional`` / ``whole_habitat`` /
``each_habitat``) re-implement the v0.1
``HabitatRadiomicsExtractor`` semantics on the v1 in-memory contracts:
SimpleITK images are assembled from the contract arrays and geometries
instead of read from files, while the mask construction, the metadata
harmonisation and the PyRadiomics calls themselves are unchanged so the
extracted numbers stay comparable with previously published results.

SimpleITK and PyRadiomics are heavy third-party libraries; every import of
them (direct or via ``habit.utils.radiomics_params_utils``) happens inside
function bodies, keeping the L3 module cheap to import.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from habit.exceptions import HABITAPIError
from habit.contracts.geometry import Geometry
from habit.contracts.subject import Subject

__all__ = [
    "resolve_modalities",
    "build_pyradiomics_extractor",
    "sitk_image_from_contract",
    "harmonize_mask_geometry",
    "binarized_habitat_mask",
    "execute_radiomics",
]

#: Default TorchRadiomics switch for habitat-level radiomics families.
#: ``False`` keeps the historical CPU PyRadiomics path (no silent numeric
#: drift). Callers may set ``"auto"`` / ``True`` to opt into TorchRadiomics;
#: scientific parameters (bin width, feature classes, ...) are never altered
#: by this switch.
DEFAULT_USE_TORCH_RADIOMICS: Union[str, bool] = False


def resolve_modalities(
    subject: Subject,
    modalities: Optional[Sequence[str]],
    *,
    owner: str,
) -> Tuple[str, ...]:
    """
    Resolve which image modalities of a subject to extract from.

    Args:
        subject: Subject whose images are considered.
        modalities: Requested modality names, or ``None`` for every image
            the subject carries (insertion order preserved).
        owner: Human-readable extractor name for the error message.

    Returns:
        The validated modality names.

    Raises:
        HABITAPIError: If a requested modality is absent from the subject.
    """
    if modalities is None:
        return tuple(subject.images.keys())
    resolved = tuple(modalities)
    missing = [name for name in resolved if name not in subject.images]
    if missing:
        raise HABITAPIError(
            f"{owner}: subject {subject.subject_id!r} does not provide "
            f"modalities {missing}; available: {sorted(subject.images)}."
        )
    return resolved


def build_pyradiomics_extractor(
    params_file: Optional[str],
    params: Optional[Dict[str, Any]],
    *,
    owner: str,
) -> Any:
    """
    Create the PyRadiomics feature extractor for one extraction call.

    Args:
        params_file: Path to a PyRadiomics parameter YAML; ``None`` selects
            PyRadiomics defaults (the v0.1 behaviour when no file was set).
        params: Inline parameter mapping, for API users who hold their
            settings in memory rather than in a YAML file.
        owner: Human-readable extractor name for the error message.

    Returns:
        An initialised ``radiomics.featureextractor.RadiomicsFeatureExtractor``.

    Raises:
        HABITAPIError: If both ``params_file`` and ``params`` are given.
    """
    if params_file is not None and params is not None:
        raise HABITAPIError(
            f"{owner}: params_file and params are mutually exclusive; "
            "pass the PyRadiomics settings as a file path OR a mapping."
        )
    import logging

    from habit.utils.radiomics_params_utils import create_radiomics_feature_extractor

    # PyRadiomics logs its default-settings mapping at INFO while constructing
    # an extractor.  ``logging.LogRecord`` treats one mapping argument as
    # interpolation data, which breaks pytest's capture handler for that
    # third-party message.  Construction is otherwise deterministic, so
    # silence only this informational external message and restore the caller's
    # logger configuration immediately afterwards.
    radiomics_logger = logging.getLogger("radiomics.featureextractor")
    was_disabled = radiomics_logger.disabled
    radiomics_logger.disabled = True
    try:
        return create_radiomics_feature_extractor(
            params if params is not None else params_file
        )
    finally:
        radiomics_logger.disabled = was_disabled


def sitk_image_from_contract(array: np.ndarray, geometry: Geometry) -> Any:
    """
    Build a SimpleITK image from a contract array and its geometry.

    Axis-order convention follows :class:`~habit.contracts.geometry.Geometry`:
    the array is NumPy ``(z, y, x)`` while spacing/origin/direction are
    already in SimpleITK ``(x, y, z)`` order, so they attach verbatim.

    Args:
        array: Voxel values, NumPy axis order ``(z, y, x)``.
        geometry: Spatial definition of ``array``.

    Returns:
        A ``SimpleITK.Image`` carrying the geometry metadata.
    """
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(np.asarray(array))
    image.SetSpacing(tuple(float(v) for v in geometry.spacing))
    image.SetOrigin(tuple(float(v) for v in geometry.origin))
    image.SetDirection(tuple(float(v) for v in geometry.direction))
    return image


def harmonize_mask_geometry(image_sitk: Any, mask_sitk: Any) -> None:
    """
    Align the mask's metadata to the raw image's, in place.

    This is the exact v0.1 ``HabitatRadiomicsExtractor`` behaviour: when the
    habitat map's direction, origin or spacing differ from the raw image's,
    the mask adopts the raw image's metadata before PyRadiomics runs. The
    v1 contracts carry geometry explicitly, yet migrated cohorts can still
    contain the same harmless metadata drift, so the harmonisation is kept
    rather than turned into an error, preserving v0.1 numerics.

    Args:
        image_sitk: Raw-image SimpleITK image (metadata source).
        mask_sitk: Habitat-map SimpleITK image (adjusted in place).
    """
    if image_sitk.GetDirection() != mask_sitk.GetDirection():
        mask_sitk.SetDirection(image_sitk.GetDirection())
    if image_sitk.GetOrigin() != mask_sitk.GetOrigin():
        mask_sitk.SetOrigin(image_sitk.GetOrigin())
    if image_sitk.GetSpacing() != mask_sitk.GetSpacing():
        mask_sitk.SetSpacing(image_sitk.GetSpacing())


def binarized_habitat_mask(habitat_sitk: Any) -> Any:
    """
    Threshold the multi-label habitat map into a single binary ROI mask.

    Every non-zero habitat label becomes ``1`` -- the exact
    ``sitk.BinaryThreshold(lowerThreshold=1, upperThreshold=<max label>,
    insideValue=1, outsideValue=0)`` construction the v0.1 traditional and
    whole-habitat paths used.

    Args:
        habitat_sitk: Habitat-map SimpleITK image with integer labels.

    Returns:
        Binary SimpleITK image of the whole ROI.

    Raises:
        HABITAPIError: If the map contains no habitat label at all.
    """
    import SimpleITK as sitk

    label_filter = sitk.LabelStatisticsImageFilter()
    label_filter.Execute(habitat_sitk, habitat_sitk)
    labels = [int(label) for label in label_filter.GetLabels() if label != 0]
    if not labels:
        raise HABITAPIError(
            "habitat map contains no non-zero label; the ROI is empty."
        )
    return sitk.BinaryThreshold(
        habitat_sitk,
        lowerThreshold=1,
        upperThreshold=float(max(labels)),
        insideValue=1,
        outsideValue=0,
    )


def execute_radiomics(
    extractor: Any,
    image_sitk: Any,
    mask_sitk: Any,
    label: int,
    *,
    use_torch_radiomics: Union[str, bool] = DEFAULT_USE_TORCH_RADIOMICS,
    torch_device: str = "auto",
    torch_dtype: str = "float32",
    subject_id: str = "",
) -> Dict[str, float]:
    """
    Run PyRadiomics once and clean its output into a numeric feature dict.

    Keys containing ``diagnostic`` are dropped (the v0.1 export did the
    same) and every remaining value is coerced to a plain float, so the
    resulting mapping can populate a feature table directly.

    When ``use_torch_radiomics`` resolves to the torch backend, TorchRadiomics
    is injected for this call only; bin width and enabled feature classes from
    the parameter file are left untouched.

    Args:
        extractor: Initialised PyRadiomics feature extractor.
        image_sitk: Intensity image (SimpleITK).
        mask_sitk: Mask image (SimpleITK).
        label: Mask label to extract within.
        use_torch_radiomics: ``"auto"``, ``True``/``"true"``, or
            ``False``/``"false"`` -- same switch as voxel/supervoxel radiomics.
        torch_device: Torch device string, or ``"auto"``.
        torch_dtype: ``"float32"`` or ``"float64"`` for the torch path.
        subject_id: Optional subject id for backend resolution logging.

    Returns:
        Feature name to float value mapping, PyRadiomics order preserved.
    """
    from habit.utils.torch_radiomics_utils import (
        injected_torch_radiomics,
        resolve_torch_dtype,
        resolve_voxel_radiomics_backend,
    )

    backend, device = resolve_voxel_radiomics_backend(
        use_torch_radiomics=use_torch_radiomics,
        torch_device=torch_device,
        subject=subject_id or None,
    )
    if backend == "torch" and device is not None:
        extractor.settings["device"] = device
        extractor.settings["dtype"] = resolve_torch_dtype(torch_dtype)

    with injected_torch_radiomics(enabled=(backend == "torch")):
        result = extractor.execute(
            imageFilepath=image_sitk,
            maskFilepath=mask_sitk,
            label=int(label),
        )
    features: Dict[str, float] = {}
    for key, value in result.items():
        if "diagnostic" in key:
            continue
        features[key] = float(value)
    return features
