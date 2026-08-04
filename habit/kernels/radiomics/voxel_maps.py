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
"""L0 kernels turning PyRadiomics voxel feature maps into a voxel table.

PyRadiomics ``execute(voxelBased=True)`` returns one image per feature, each
cropped to the ROI bounding box with padding that depends on ``kernelRadius``.
Recovering a voxel-by-feature matrix from those maps is where the subtle
mistakes live -- a crop offset ignored, a mask aligned by array shape instead
of physical coordinates, or a feature filtered by value instead of by mask --
so the alignment rules live here once and are shared by the v0.1 extractor and
the v1.0 domain operator.

The functions take an already-computed PyRadiomics result: running the
extractor (and any torch injection or logging around it) belongs to the caller,
which keeps this module free of habit imports and of any global state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

__all__ = [
    "enabled_voxel_feature_classes",
    "group_voxel_feature_keys_by_class",
    "mask_array_for_feature_map",
    "feature_values_in_mask",
    "voxel_feature_frame",
]


def enabled_voxel_feature_classes(enabled_features: Mapping[str, Any]) -> List[str]:
    """
    Return the sorted feature classes that voxel-based extraction can compute.

    Shape classes are excluded because PyRadiomics does not compute them per
    voxel.

    Args:
        enabled_features: ``RadiomicsFeatureExtractor.enabledFeatures``.

    Returns:
        Feature class names, e.g. ``["firstorder", "glcm"]``.
    """
    return sorted(
        feature_class
        for feature_class in enabled_features.keys()
        if not str(feature_class).startswith("shape")
    )


def group_voxel_feature_keys_by_class(
    feature_keys: List[str],
    feature_classes: List[str],
) -> Dict[str, List[str]]:
    """
    Group PyRadiomics result keys by feature class.

    Keys follow ``{imageType}_{featureClass}_{featureName}``.

    Args:
        feature_keys: Non-diagnostic keys of a voxel-based result.
        feature_classes: Feature class names to group by.

    Returns:
        Feature class name -> matching keys.
    """
    grouped: Dict[str, List[str]] = {name: [] for name in feature_classes}
    for key in feature_keys:
        for feature_class in feature_classes:
            if f"_{feature_class}_" in key:
                grouped[feature_class].append(key)
                break
    return grouped


def feature_values_in_mask(
    feature_array: np.ndarray,
    mask_array: np.ndarray,
) -> np.ndarray:
    """
    Select every voxel feature value inside the non-background mask.

    Zero and negative values are scientifically valid for many radiomics maps,
    so the mask -- never the feature value -- defines the voxel population.
    Filtering by value would yield a different row count per feature column.

    Args:
        feature_array: One voxel feature map as an array.
        mask_array: Spatially aligned mask on the same grid; zero is
            background.

    Returns:
        Feature values for all foreground voxels, in C order.

    Raises:
        ValueError: If the shapes differ or the mask has no foreground.
    """
    if feature_array.shape != mask_array.shape:
        raise ValueError(
            "Voxel feature map shape does not match mask shape: "
            f"{feature_array.shape} != {mask_array.shape}."
        )
    roi = mask_array != 0
    if not np.any(roi):
        raise ValueError("Voxel radiomics mask does not contain any foreground voxels.")
    return feature_array[roi]


def mask_array_for_feature_map(
    mask: sitk.Image,
    feature_map: sitk.Image,
    *,
    label: Optional[int] = None,
) -> np.ndarray:
    """
    Align a full-size mask onto the grid of a cropped voxel feature map.

    Alignment must go through the physical coordinate system: slicing by array
    shape would drop the crop offset and be wrong for non-unit spacing,
    non-zero origins, rotated directions, or radiomics resampling.

    Args:
        mask: Full-size mask handed to PyRadiomics.
        feature_map: Cropped voxel feature map returned by PyRadiomics.
        label: Mask label PyRadiomics selected; ``None`` treats every nonzero
            value as foreground.

    Returns:
        A binary mask on the feature map's exact array grid.

    Raises:
        ValueError: If dimensions differ, equally sampled grids are not
            lattice-aligned, or the crop loses ROI voxels.
    """
    if mask.GetDimension() != feature_map.GetDimension():
        raise ValueError(
            "Voxel feature map and mask dimensions do not match: "
            f"{feature_map.GetDimension()} != {mask.GetDimension()}."
        )

    spacing_matches = np.allclose(
        mask.GetSpacing(), feature_map.GetSpacing(), rtol=0.0, atol=1e-6
    )
    direction_matches = np.allclose(
        mask.GetDirection(), feature_map.GetDirection(), rtol=0.0, atol=1e-6
    )
    if spacing_matches and direction_matches:
        # Cropped maps with unchanged sampling must start exactly on the source
        # mask lattice. A fractional index means an origin or direction
        # metadata error that nearest-neighbour resampling would hide.
        start_index = np.asarray(
            mask.TransformPhysicalPointToContinuousIndex(feature_map.GetOrigin()),
            dtype=np.float64,
        )
        if not np.allclose(start_index, np.rint(start_index), rtol=0.0, atol=1e-5):
            raise ValueError(
                "Voxel feature map is not aligned to the mask voxel lattice: "
                f"continuous_start_index={tuple(start_index)}."
            )

    aligned_mask = sitk.Resample(
        mask,
        feature_map,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        mask.GetPixelID(),
    )
    aligned_array = sitk.GetArrayFromImage(aligned_mask)
    source_array = sitk.GetArrayViewFromImage(mask)

    if label is None:
        roi = aligned_array != 0
        source_roi_count = int(np.count_nonzero(source_array))
    else:
        roi = aligned_array == label
        source_roi_count = int(np.count_nonzero(source_array == label))

    # With unchanged sampling PyRadiomics only crops the grid; it does not
    # change the ROI voxel population. A count difference means the crop does
    # not fully cover the requested label and must not be accepted.
    if spacing_matches and direction_matches:
        aligned_roi_count = int(np.count_nonzero(roi))
        if aligned_roi_count != source_roi_count:
            raise ValueError(
                "Voxel feature map physical extent does not contain the complete "
                f"mask ROI: aligned_voxels={aligned_roi_count}, "
                f"source_voxels={source_roi_count}, label={label}."
            )

    return roi.astype(np.uint8, copy=False)


def voxel_feature_frame(
    result: Dict[str, Any],
    mask: sitk.Image,
    *,
    image_name: Optional[str] = None,
    mask_label: Optional[int] = 1,
    output_float32: bool = True,
) -> pd.DataFrame:
    """
    Assemble a voxel-by-feature table from a voxel-based PyRadiomics result.

    Rows are the ROI voxels in C order, which is the order
    ``np.argwhere(mask == label)`` produces, so callers can pair the table with
    voxel coordinates without extra bookkeeping.

    Args:
        result: Return value of ``execute(..., voxelBased=True)``. Feature maps
            are popped as they are consumed, so peak memory stays close to one
            map at a time; pass a dict the caller no longer needs.
        mask: Full-size mask handed to PyRadiomics.
        image_name: Modality name suffixed onto each column as
            ``{key}-{image_name}``, the v0.1 column scheme. ``None`` keeps the
            bare PyRadiomics key.
        mask_label: Mask label to select; ``None`` uses every nonzero voxel.
        output_float32: Downcast the table to float32, the v0.1 default that
            keeps large voxel tables manageable.

    Returns:
        One row per ROI voxel, one column per feature map.

    Raises:
        ValueError: If the maps disagree on the ROI row count, or if no
            feature map is present.
    """
    keys = [key for key in result.keys() if not str(key).startswith("diagnostic")]
    feature_names: List[str] = []
    columns: List[np.ndarray] = []
    masks_by_geometry: Dict[Tuple[Any, ...], np.ndarray] = {}

    for key in keys:
        value = result.pop(key, None)
        if not isinstance(value, sitk.Image):
            continue
        feature_names.append(f"{key}-{image_name}" if image_name else str(key))
        feature_array = sitk.GetArrayFromImage(value)
        geometry_key: Tuple[Any, ...] = (
            tuple(value.GetSize()),
            tuple(value.GetSpacing()),
            tuple(value.GetOrigin()),
            tuple(value.GetDirection()),
        )
        if geometry_key not in masks_by_geometry:
            masks_by_geometry[geometry_key] = mask_array_for_feature_map(
                mask, value, label=mask_label
            )
        columns.append(feature_values_in_mask(feature_array, masks_by_geometry[geometry_key]))
        del value, feature_array

    if not columns:
        raise ValueError(
            "Voxel radiomics produced no feature map; check that the parameter "
            "file enables at least one non-shape feature class."
        )
    row_counts = {column.shape[0] for column in columns}
    if len(row_counts) > 1:
        raise ValueError(
            "Voxel radiomics feature maps produced inconsistent ROI row "
            f"counts: {sorted(row_counts)}."
        )

    frame = pd.DataFrame(np.stack(columns, axis=1), columns=feature_names)
    if output_float32:
        frame = frame.astype(np.float32)
    return frame
