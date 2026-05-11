from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import SimpleITK as sitk

from habit.core.preprocessing.registration.base import BaseRegistrationBackend
from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)


class SitkRegistrationBackend(BaseRegistrationBackend):
    """Registration backend using SimpleITK ImageRegistrationMethod."""

    def _map_ants_transform_name_to_sitk_kind(self, name: str) -> str:
        n = (name or "").strip().lower()
        if n == "rigid":
            return "rigid"
        if n in {"affine", "trsaa"}:
            return "affine"
        if n == "bspline":
            return "bspline"
        deformable_aliases = {
            "syn", "synra", "synonly", "elastic", "syncc",
            "synabp", "synbold", "synboldaff", "synaggro", "tvmsq",
        }
        if n in deformable_aliases or n.startswith("syn"):
            return "bspline"
        raise ValueError(
            f"type_of_transform={name!r} is not supported for backend='simpleitk'. "
            "Use Rigid, Affine, TRSAA, BSpline, or an ANTs deformable name (mapped to BSpline)."
        )

    def _build_initial_transform(
        self,
        fixed_image: sitk.Image,
        moving_image: sitk.Image,
        kind: str,
    ) -> sitk.Transform:
        dim = int(fixed_image.GetDimension())
        if kind == "rigid":
            if dim == 3:
                rigid = sitk.VersorRigid3DTransform()
            elif dim == 2:
                rigid = sitk.Euler2DTransform()
            else:
                raise ValueError(
                    f"Rigid SimpleITK registration supports 2D or 3D images, got dimension {dim}"
                )
            return sitk.CenteredTransformInitializer(
                fixed_image, moving_image, rigid,
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
        if kind == "affine":
            affine = sitk.AffineTransform(dim)
            return sitk.CenteredTransformInitializer(
                fixed_image, moving_image, affine,
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
        if kind == "bspline":
            default_mesh = [8] * dim
            mesh = self.sitk_reg_params.get("bspline_mesh_size", default_mesh)
            if isinstance(mesh, int):
                mesh_list = [int(mesh)] * dim
            elif isinstance(mesh, (list, tuple)):
                if len(mesh) == 1:
                    mesh_list = [int(mesh[0])] * dim
                else:
                    mesh_list = [int(x) for x in mesh[:dim]]
            else:
                mesh_list = list(default_mesh)
            order = int(self.sitk_reg_params.get("bspline_order", 3))
            return sitk.BSplineTransformInitializer(
                image1=fixed_image,
                transformDomainMeshSize=mesh_list,
                order=order,
            )
        raise ValueError(f"Unknown SimpleITK transform kind {kind!r}")

    def _configure_metric(self, method: sitk.ImageRegistrationMethod) -> None:
        m = (self.metric or "MI").strip().upper()
        if m == "MI":
            bins = int(self.sitk_reg_params.get("number_of_histogram_bins", 50))
            method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)
        elif m == "CC":
            method.SetMetricAsCorrelation()
        elif m in {"MEANSQUARES", "MEAN_SQUARES"}:
            method.SetMetricAsMeanSquares()
        else:
            logger.warning(
                "metric=%r is not mapped for backend='simpleitk'; using Mattes MI instead.",
                self.metric,
            )
            bins = int(self.sitk_reg_params.get("number_of_histogram_bins", 50))
            method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bins)

    def _configure_optimizer(self, method: sitk.ImageRegistrationMethod) -> None:
        opt = (self.optimizer or "gradient_descent").lower()
        iterations = int(self.sitk_reg_params.get("number_of_iterations", 100))
        if "lbfgs" in opt:
            method.SetOptimizerAsLBFGSB(
                gradientConvergenceTolerance=1e-5,
                numberOfIterations=iterations,
                maximumNumberOfCorrections=5,
                maximumNumberOfFunctionEvaluations=max(iterations, 100),
                costFunctionConvergenceFactor=1e7,
            )
        else:
            lr = float(self.sitk_reg_params.get("learning_rate", 1.0))
            method.SetOptimizerAsGradientDescent(
                learningRate=lr,
                numberOfIterations=iterations,
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=10,
            )
        method.SetOptimizerScalesFromPhysicalShift()

    def _run_registration_execute(
        self,
        fixed_image: sitk.Image,
        moving_image: sitk.Image,
        kind: str,
        fixed_mask: Optional[sitk.Image],
        moving_mask: Optional[sitk.Image],
    ) -> sitk.Transform:
        initial = self._build_initial_transform(fixed_image, moving_image, kind)
        method = sitk.ImageRegistrationMethod()
        self._configure_metric(method)
        method.SetInterpolator(sitk.sitkLinear)

        sampling = float(self.sitk_reg_params.get("metric_sampling_percentage", 0.01))
        method.SetMetricSamplingStrategy(method.RANDOM)
        method.SetMetricSamplingPercentage(sampling)

        shrink = self.sitk_reg_params.get("shrink_factors_per_level", [4, 2, 1])
        sigma = self.sitk_reg_params.get("smoothing_sigmas_per_level", [2.1, 1.0, 0.0])
        method.SetShrinkFactorsPerLevel(tuple(int(x) for x in shrink))
        method.SetSmoothingSigmasPerLevel(tuple(float(x) for x in sigma))
        method.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(True)

        if fixed_mask is not None:
            method.SetMetricFixedMask(fixed_mask)
        if moving_mask is not None:
            method.SetMetricMovingMask(moving_mask)

        method.SetInitialTransform(initial, inPlace=False)
        self._configure_optimizer(method)
        return method.Execute(fixed_image, moving_image)

    def register_image(
        self,
        fixed_image_sitk: sitk.Image,
        moving_image_sitk: sitk.Image,
        fixed_mask_sitk: Optional[sitk.Image] = None,
        moving_mask_sitk: Optional[sitk.Image] = None,
        fixed_image_ants: Optional[Any] = None,
    ) -> Tuple[sitk.Image, List[str]]:
        kind = self._map_ants_transform_name_to_sitk_kind(self.type_of_transform)
        final_transform = self._run_registration_execute(
            fixed_image_sitk, moving_image_sitk, kind, fixed_mask_sitk, moving_mask_sitk
        )

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        resampler.SetTransform(final_transform)
        registered = resampler.Execute(moving_image_sitk)

        fd, path = tempfile.mkstemp(suffix=".tfm", prefix="habit_sitk_reg_")
        os.close(fd)
        sitk.WriteTransform(final_transform, path)
        return registered, [path]

    def apply_transform_mask(
        self,
        fixed_reference_sitk: sitk.Image,
        moving_mask_sitk: sitk.Image,
        transform_files: List[str],
        fixed_image_ants: Optional[Any] = None,
    ) -> sitk.Image:
        if not transform_files:
            raise ValueError("transform_files must not be empty for SimpleITK mask warping")
        transform = sitk.ReadTransform(transform_files[0])
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_reference_sitk)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(transform)
        return resampler.Execute(moving_mask_sitk)
