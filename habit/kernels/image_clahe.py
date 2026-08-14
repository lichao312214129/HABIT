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
"""Adaptive histogram equalization (CLAHE-style) on a SimpleITK volume."""

from __future__ import annotations

from typing import Any, Tuple, Union

__all__ = ["adaptive_histogram_equalize_sitk_image"]


def adaptive_histogram_equalize_sitk_image(
    sitk_image: Any,
    *,
    alpha: float = 0.3,
    beta: float = 0.3,
    radius: Union[int, Tuple[int, int, int]] = 5,
) -> Any:
    """
    Apply SimpleITK adaptive histogram equalization.

    Args:
        sitk_image: Input SimpleITK image.
        alpha: How much the filter acts like classical HE (``[0, 1]``).
        beta: How much the filter adapts locally (``[0, 1]``).
        radius: Local-region radius in pixels (int or ``(x, y, z)``).

    Returns:
        Contrast-enhanced SimpleITK image (float32).

    Raises:
        ValueError: If ``alpha`` or ``beta`` is outside ``[0, 1]``.
    """
    import SimpleITK as sitk

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in range [0, 1], got {alpha}")
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError(f"beta must be in range [0, 1], got {beta}")
    if isinstance(radius, int):
        radius_xyz = (int(radius), int(radius), int(radius))
    else:
        radius_xyz = tuple(int(v) for v in radius)

    image = sitk.Cast(sitk_image, sitk.sitkFloat32)
    filt = sitk.AdaptiveHistogramEqualizationImageFilter()
    filt.SetAlpha(float(alpha))
    filt.SetBeta(float(beta))
    filt.SetRadius(radius_xyz)
    return filt.Execute(image)
