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
"""Resample a SimpleITK volume to a target spacing."""

from __future__ import annotations

from typing import Any, Sequence

__all__ = ["RESAMPLE_INTERPOLATORS", "resample_sitk_image"]

RESAMPLE_INTERPOLATORS: tuple[str, ...] = (
    "nearest",
    "linear",
    "bilinear",
    "bspline",
    "bicubic",
    "gaussian",
    "lanczos",
    "hamming",
    "cosine",
    "welch",
    "blackman",
)


def _interpolator(name: str) -> Any:
    """Map a HABIT interpolator name to a SimpleITK interpolator enum."""
    import SimpleITK as sitk

    mapping = {
        "nearest": sitk.sitkNearestNeighbor,
        "linear": sitk.sitkLinear,
        "bilinear": sitk.sitkLinear,
        "bspline": sitk.sitkBSpline,
        "bicubic": sitk.sitkBSpline,
        "gaussian": sitk.sitkGaussian,
        "lanczos": sitk.sitkLanczosWindowedSinc,
        "hamming": sitk.sitkHammingWindowedSinc,
        "cosine": sitk.sitkCosineWindowedSinc,
        "welch": sitk.sitkWelchWindowedSinc,
        "blackman": sitk.sitkBlackmanWindowedSinc,
    }
    return mapping.get(str(name).lower(), sitk.sitkLinear)


def resample_sitk_image(
    sitk_image: Any,
    target_spacing: Sequence[float],
    *,
    interpolator: str = "bilinear",
) -> Any:
    """
    Resample ``sitk_image`` onto ``target_spacing``, keeping origin/direction.

    Args:
        sitk_image: Input SimpleITK image.
        target_spacing: Target spacing ``(x, y, z)`` in millimetres.
        interpolator: HABIT interpolator name (see ``RESAMPLE_INTERPOLATORS``).

    Returns:
        Resampled SimpleITK image.
    """
    import SimpleITK as sitk

    original_spacing = sitk_image.GetSpacing()
    size = sitk_image.GetSize()
    zoom = [
        orig / tgt for orig, tgt in zip(original_spacing, tuple(target_spacing))
    ]
    new_size = [int(round(sz * factor)) for sz, factor in zip(size, zoom)]
    reference = sitk.Image(new_size, sitk_image.GetPixelID())
    reference.SetSpacing(tuple(float(v) for v in target_spacing))
    reference.SetOrigin(sitk_image.GetOrigin())
    reference.SetDirection(sitk_image.GetDirection())
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(_interpolator(interpolator))
    return resampler.Execute(sitk_image)
