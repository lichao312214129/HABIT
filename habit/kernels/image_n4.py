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
"""N4 bias-field correction on a SimpleITK volume."""

from __future__ import annotations

from typing import Any, Optional, Sequence

__all__ = ["n4_correct_sitk_image"]


def n4_correct_sitk_image(
    sitk_image: Any,
    sitk_mask: Optional[Any] = None,
    *,
    num_fitting_levels: int = 4,
    num_iterations: Optional[Sequence[int]] = None,
    convergence_threshold: float = 0.001,
    shrink_factor: int = 4,
) -> Any:
    """
    Apply SimpleITK N4 bias-field correction.

    When ``shrink_factor > 1`` the filter runs on a downsampled grid and
    the log-bias field is applied back at full resolution (v0.1 behaviour).

    Args:
        sitk_image: Input SimpleITK image.
        sitk_mask: Optional mask restricting the fit.
        num_fitting_levels: N4 fitting levels.
        num_iterations: Iterations per level; default ``[50] * levels``.
        convergence_threshold: N4 convergence threshold.
        shrink_factor: Downsample factor used to speed up the fit.

    Returns:
        Bias-corrected SimpleITK image (float32).
    """
    import SimpleITK as sitk

    levels = int(num_fitting_levels)
    iterations = (
        list(num_iterations)
        if num_iterations is not None
        else [50] * levels
    )
    image = sitk.Cast(sitk_image, sitk.sitkFloat32)
    original = image
    mask = sitk.Cast(sitk_mask, sitk.sitkUInt8) if sitk_mask is not None else None
    factor = int(shrink_factor)
    if factor > 1:
        image = sitk.Shrink(original, [factor] * original.GetDimension())
        if mask is not None:
            mask = sitk.Shrink(mask, [factor] * original.GetDimension())

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(iterations)
    corrector.SetConvergenceThreshold(float(convergence_threshold))
    if mask is not None:
        corrected = corrector.Execute(image, mask)
    else:
        corrected = corrector.Execute(image)
    if factor > 1:
        log_bias = corrector.GetLogBiasFieldAsImage(original)
        corrected = original / sitk.Exp(log_bias)
    return corrected
