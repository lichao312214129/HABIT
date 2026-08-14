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
"""Reorient a SimpleITK volume to a canonical DICOM orientation."""

from __future__ import annotations

from typing import Any

__all__ = ["reorient_sitk_image"]


def reorient_sitk_image(
    sitk_image: Any,
    *,
    target_orientation: str = "LPS",
    mode: str = "closest",
    is_mask: bool = False,
) -> Any:
    """
    Reorient ``sitk_image`` to ``target_orientation``.

    Args:
        sitk_image: Input SimpleITK image.
        target_orientation: DICOM orientation code (e.g. ``LPS``, ``RAS``).
        mode: ``closest`` flips/permutes axes only; ``strict`` resamples
            onto an orthogonal grid (linear for images, nearest for masks).
        is_mask: When ``mode='strict'``, use nearest-neighbour interpolation.

    Returns:
        Reoriented SimpleITK image.

    Raises:
        ValueError: If ``mode`` is not ``closest`` or ``strict``.
    """
    import SimpleITK as sitk

    target = str(target_orientation).upper()
    mode_key = str(mode).lower()
    if mode_key not in {"closest", "strict"}:
        raise ValueError("mode must be either 'closest' or 'strict'")

    orient = sitk.DICOMOrientImageFilter()
    orient.SetDesiredCoordinateOrientation(target)
    if mode_key == "closest":
        return orient.Execute(sitk_image)

    dummy = sitk.Image(1, 1, 1, sitk.sitkUInt8)
    dummy.SetDirection(sitk_image.GetDirection())
    target_direction = orient.Execute(dummy).GetDirection()
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_image)
    resampler.SetOutputDirection(target_direction)
    resampler.SetTransform(sitk.Transform())
    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        mm = sitk.MinimumMaximumImageFilter()
        mm.Execute(sitk_image)
        resampler.SetDefaultPixelValue(mm.GetMinimum())
    return resampler.Execute(sitk_image)
