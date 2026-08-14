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
"""SimpleITK image-to-image registration (no HABIT imports)."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["register_sitk_image", "warp_sitk_mask"]


def _transform_kind(name: str) -> str:
    """Map an ANTs-style transform name onto a SimpleITK kind."""
    key = (name or "").strip().lower()
    if key == "rigid":
        return "rigid"
    if key in {"affine", "trsaa"}:
        return "affine"
    if key == "bspline":
        return "bspline"
    deformable = {
        "syn",
        "synra",
        "synonly",
        "elastic",
        "syncc",
        "synabp",
        "synbold",
        "synboldaff",
        "synaggro",
        "tvmsq",
    }
    if key in deformable or key.startswith("syn"):
        return "bspline"
    raise ValueError(
        f"type_of_transform={name!r} is not supported for backend='simpleitk'. "
        "Use Rigid, Affine, TRSAA, BSpline, or an ANTs deformable name "
        "(mapped to BSpline)."
    )


def _initial_transform(
    sitk: Any,
    fixed: Any,
    moving: Any,
    kind: str,
    sitk_params: Dict[str, Any],
) -> Any:
    """Build a centered initial transform of the requested kind."""
    dim = int(fixed.GetDimension())
    if kind == "rigid":
        if dim == 3:
            rigid = sitk.VersorRigid3DTransform()
        elif dim == 2:
            rigid = sitk.Euler2DTransform()
        else:
            raise ValueError(
                f"Rigid SimpleITK registration supports 2D or 3D, got {dim}"
            )
        return sitk.CenteredTransformInitializer(
            fixed,
            moving,
            rigid,
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
    if kind == "affine":
        return sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.AffineTransform(dim),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
    default_mesh = [8] * dim
    mesh = sitk_params.get("bspline_mesh_size", default_mesh)
    if isinstance(mesh, int):
        mesh_list = [int(mesh)] * dim
    elif isinstance(mesh, (list, tuple)):
        mesh_list = (
            [int(mesh[0])] * dim
            if len(mesh) == 1
            else [int(x) for x in mesh[:dim]]
        )
    else:
        mesh_list = list(default_mesh)
    return sitk.BSplineTransformInitializer(
        image1=fixed,
        transformDomainMeshSize=mesh_list,
        order=int(sitk_params.get("bspline_order", 3)),
    )


def register_sitk_image(
    fixed: Any,
    moving: Any,
    *,
    type_of_transform: str = "Rigid",
    metric: str = "MI",
    optimizer: Optional[str] = None,
    fixed_mask: Optional[Any] = None,
    moving_mask: Optional[Any] = None,
    sitk_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, List[str]]:
    """
    Register ``moving`` onto ``fixed`` with SimpleITK.

    Args:
        fixed: Fixed SimpleITK image.
        moving: Moving SimpleITK image.
        type_of_transform: ANTs-style name mapped onto rigid/affine/bspline.
        metric: ``MI``, ``CC``, or ``MeanSquares``.
        optimizer: ``lbfgs`` selects LBFGSB; otherwise gradient descent.
        fixed_mask: Optional fixed-image metric mask.
        moving_mask: Optional moving-image metric mask.
        sitk_params: Optional SimpleITK tuning dict (bins, shrink, …).

    Returns:
        ``(registered_image, [transform_path])``. The caller owns the
        temporary ``.tfm`` file.
    """
    import SimpleITK as sitk

    params = dict(sitk_params or {})
    kind = _transform_kind(type_of_transform)
    initial = _initial_transform(sitk, fixed, moving, kind, params)
    method = sitk.ImageRegistrationMethod()
    metric_key = (metric or "MI").strip().upper()
    bins = int(params.get("number_of_histogram_bins", 50))
    if metric_key == "MI":
        method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)
    elif metric_key == "CC":
        method.SetMetricAsCorrelation()
    elif metric_key in {"MEANSQUARES", "MEAN_SQUARES"}:
        method.SetMetricAsMeanSquares()
    else:
        method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(
        float(params.get("metric_sampling_percentage", 0.01))
    )
    method.SetShrinkFactorsPerLevel(
        tuple(int(x) for x in params.get("shrink_factors_per_level", [4, 2, 1]))
    )
    method.SetSmoothingSigmasPerLevel(
        tuple(float(x) for x in params.get("smoothing_sigmas_per_level", [2.1, 1.0, 0.0]))
    )
    method.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(True)
    if fixed_mask is not None:
        method.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        method.SetMetricMovingMask(moving_mask)
    method.SetInitialTransform(initial, inPlace=False)
    iterations = int(params.get("number_of_iterations", 100))
    opt = (optimizer or "gradient_descent").lower()
    if "lbfgs" in opt:
        method.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=iterations,
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=max(iterations, 100),
            costFunctionConvergenceFactor=1e7,
        )
    else:
        method.SetOptimizerAsGradientDescent(
            learningRate=float(params.get("learning_rate", 1.0)),
            numberOfIterations=iterations,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10,
        )
    method.SetOptimizerScalesFromPhysicalShift()
    final = method.Execute(fixed, moving)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetTransform(final)
    registered = resampler.Execute(moving)
    handle, path = tempfile.mkstemp(suffix=".tfm", prefix="habit_sitk_reg_")
    os.close(handle)
    sitk.WriteTransform(final, path)
    return registered, [path]


def warp_sitk_mask(
    fixed_reference: Any,
    moving_mask: Any,
    transform_files: List[str],
) -> Any:
    """
    Warp a label mask onto the fixed grid with nearest-neighbour resampling.

    Args:
        fixed_reference: Fixed SimpleITK image (grid source).
        moving_mask: Moving SimpleITK mask.
        transform_files: Paths written by :func:`register_sitk_image`.

    Returns:
        Warped mask on the fixed grid.

    Raises:
        ValueError: If ``transform_files`` is empty.
    """
    import SimpleITK as sitk

    if not transform_files:
        raise ValueError("transform_files must not be empty for mask warping")
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.ReadTransform(transform_files[0]))
    return resampler.Execute(moving_mask)
