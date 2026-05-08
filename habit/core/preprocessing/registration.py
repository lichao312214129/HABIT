from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import SimpleITK as sitk

from habit.utils.image_converter import ImageConverter
from .base_preprocessor import BasePreprocessor
from .preprocessor_factory import PreprocessorFactory

from ...utils.log_utils import get_module_logger

# Get module logger
logger = get_module_logger(__name__)

# Optional keyword args consumed by the SimpleITK backend only (not forwarded to ``ants.registration``).
_SITK_OPTION_KEYS: Tuple[str, ...] = (
    "number_of_histogram_bins",
    "metric_sampling_percentage",
    "shrink_factors_per_level",
    "smoothing_sigmas_per_level",
    "learning_rate",
    "number_of_iterations",
    "bspline_mesh_size",
    "bspline_order",
)


@PreprocessorFactory.register("registration")
class RegistrationPreprocessor(BasePreprocessor):
    """Register moving images to a fixed reference using ANTsPy or SimpleITK.

    Pipelines store ``sitk.Image`` volumes in the subject dictionary. The ``ants``
    backend converts to ``ANTsImage`` for ``ants.registration`` / ``ants.apply_transforms``.
    The ``simpleitk`` backend runs ``sitk.ImageRegistrationMethod`` and warps masks with
    ``sitk.ResampleImageFilter``, avoiding ANTsPy when it is unstable or unavailable.
    """

    def __init__(
        self,
        keys: Union[str, List[str]],
        fixed_image: str,
        mask_keys: Optional[Union[str, List[str]]] = None,
        type_of_transform: str = "SyN",
        metric: str = "MI",
        optimizer: Optional[str] = None,
        use_mask: bool = False,
        allow_missing_keys: bool = False,
        replace_by_fixed_image_mask: bool = True,
        backend: str = "ants",
        **kwargs: Any,
    ) -> None:
        """Initialize the registration preprocessor.

        Args:
            keys (Union[str, List[str]]): Keys of the images to be registered.
            fixed_image (str): Key of the reference image to register to.
            mask_keys (Optional[Union[str, List[str]]]): Keys of the masks to use for registration.
            type_of_transform (str): Transform / registration model name. For ``backend="ants"`` this is
                passed to ANTs (SyN, Rigid, Affine, etc.). For ``backend="simpleitk"``, names are mapped:
                ``Rigid``; ``Affine`` or ``TRSAA``; deformable ANTs-style names (e.g. ``SyN``) map to a
                BSpline approximation (not equivalent to ANTs SyN).
            metric (str): Similarity metric. ``MI``, ``CC``, ``MeanSquares`` are supported for
                SimpleITK; other values fall back to Mattes mutual information with a warning.
            optimizer (str): Optimizer hint. For SimpleITK, strings containing ``lbfgs`` select LBFGSB;
                otherwise gradient descent is used. ANTs uses its own optimiser names when
                ``backend="ants"``.
            use_mask (bool): If True, use mask for registration.
            allow_missing_keys (bool): If True, allows missing keys in the input data.
            replace_by_fixed_image_mask (bool): If True, use fixed image's mask to replace moving
                image's mask after registration.
            backend (str): ``"ants"`` (default) or ``"simpleitk"``.
            **kwargs: Extra parameters. SimpleITK-only tuning keys (see module constant
                ``_SITK_OPTION_KEYS``) are stripped and applied only when ``backend="simpleitk"``;
                remaining keys are forwarded to ``ants.registration`` when ``backend="ants"``.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)

        # Convert single key to list
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.fixed_image = fixed_image

        # Handle mask keys
        if mask_keys is None:
            self.mask_keys = None
        else:
            self.mask_keys = [mask_keys] if isinstance(mask_keys, str) else mask_keys

        # Set registration parameters
        self.type_of_transform = type_of_transform
        self.metric = metric
        self.optimizer = optimizer
        self.use_mask = use_mask
        self.replace_by_fixed_image_mask = replace_by_fixed_image_mask

        b = (backend or "ants").strip().lower()
        if b not in {"ants", "simpleitk"}:
            raise ValueError(
                f"backend must be 'ants' or 'simpleitk', got {backend!r}"
            )
        self.backend = b

        sitk_params: Dict[str, Any] = {}
        reg_params: Dict[str, Any] = dict(kwargs)
        for _key in _SITK_OPTION_KEYS:
            if _key in reg_params:
                sitk_params[_key] = reg_params.pop(_key)
        self.reg_params = reg_params
        self.sitk_reg_params: Dict[str, Any] = sitk_params

    def _register_image(
        self,
        fixed_image: Any,
        moving_image: Any,
        fixed_mask: Any = None,
        moving_mask: Any = None,
    ) -> Tuple[Any, List[str]]:
        """Register a moving image to a fixed image using ANTsPy.

        Args:
            fixed_image (Any): Reference ``ANTsImage`` (``ants.ANTsImage``).
            moving_image (Any): ``ANTsImage`` to be registered.
            fixed_mask (Optional[Any]): ``ANTsImage`` mask for the fixed image.
            moving_mask (Optional[Any]): ``ANTsImage`` mask for the moving image.

        Returns:
            Tuple[Any, List[str]]: Registered ``ANTsImage`` and list of forward transform paths.
        """
        import ants  # Lazy import so ``simpleitk``-only workflows avoid loading ANTsPy.

        reg_params: Dict[str, Any] = {
            "metric": self.metric,
            "optimizer": self.optimizer,
            **self.reg_params,
        }

        if fixed_mask is not None:
            reg_params["mask"] = fixed_mask
        if moving_mask is not None:
            reg_params["moving_mask"] = moving_mask

        reg_result: Dict[str, Any] = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            type_of_transform=self.type_of_transform,
            **reg_params,
        )

        return reg_result["warpedmovout"], reg_result["fwdtransforms"]

    def _map_ants_transform_name_to_sitk_kind(self, name: str) -> str:
        """Map an ANTs-style ``type_of_transform`` label to a SimpleITK registration family.

        Args:
            name (str): User ``type_of_transform`` string (ANTs naming).

        Returns:
            str: One of ``"rigid"``, ``"affine"``, or ``"bspline"``.
        """
        n = (name or "").strip().lower()
        if n == "rigid":
            return "rigid"
        if n in {"affine", "trsaa"}:
            return "affine"
        if n == "bspline":
            return "bspline"
        deformable_aliases = {
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
        if n in deformable_aliases or n.startswith("syn"):
            return "bspline"
        raise ValueError(
            f"type_of_transform={name!r} is not supported for backend='simpleitk'. "
            "Use Rigid, Affine, TRSAA, BSpline, or an ANTs deformable name (mapped to BSpline)."
        )

    def _build_initial_transform_sitk(
        self,
        fixed_image: sitk.Image,
        moving_image: sitk.Image,
        kind: str,
    ) -> sitk.Transform:
        """Create the initial transform used by ``ImageRegistrationMethod``.

        Args:
            fixed_image (sitk.Image): Fixed (reference) image.
            moving_image (sitk.Image): Moving image.
            kind (str): ``"rigid"``, ``"affine"``, or ``"bspline"``.

        Returns:
            sitk.Transform: Initial transform suitable for ``SetInitialTransform``.
        """
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
                fixed_image,
                moving_image,
                rigid,
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
        if kind == "affine":
            affine = sitk.AffineTransform(dim)
            return sitk.CenteredTransformInitializer(
                fixed_image,
                moving_image,
                affine,
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

    def _configure_metric_sitk(self, method: sitk.ImageRegistrationMethod) -> None:
        """Configure ``ImageRegistrationMethod`` similarity metric from ``self.metric``."""
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

    def _configure_optimizer_sitk(self, method: sitk.ImageRegistrationMethod) -> None:
        """Configure optimiser for ``ImageRegistrationMethod``."""
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

    def _register_image_sitk(
        self,
        fixed_image: sitk.Image,
        moving_image: sitk.Image,
        fixed_mask: Optional[sitk.Image] = None,
        moving_mask: Optional[sitk.Image] = None,
    ) -> Tuple[sitk.Image, List[str]]:
        """Register ``moving_image`` to ``fixed_image`` using ``ImageRegistrationMethod``.

        Args:
            fixed_image (sitk.Image): Fixed reference image (float32 recommended).
            moving_image (sitk.Image): Moving image (float32 recommended).
            fixed_mask (Optional[sitk.Image]): Optional uint8 fixed mask for metric ROI.
            moving_mask (Optional[sitk.Image]): Optional uint8 moving mask for metric ROI.

        Returns:
            Tuple[sitk.Image, List[str]]: Warped moving image on the fixed grid, and a one-element
            list containing the path to a temporary ``.tfm`` written with ``sitk.WriteTransform``.
        """
        kind = self._map_ants_transform_name_to_sitk_kind(self.type_of_transform)
        initial = self._build_initial_transform_sitk(fixed_image, moving_image, kind)

        method = sitk.ImageRegistrationMethod()
        self._configure_metric_sitk(method)
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
        self._configure_optimizer_sitk(method)

        final_transform = method.Execute(fixed_image, moving_image)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        resampler.SetTransform(final_transform)
        registered = resampler.Execute(moving_image)

        fd, path = tempfile.mkstemp(suffix=".tfm", prefix="habit_sitk_reg_")
        os.close(fd)
        sitk.WriteTransform(final_transform, path)
        return registered, [path]

    def _apply_transform_mask_ants(
        self,
        fixed_ants: Any,
        moving_mask_ants: Any,
        transform_files: List[str],
    ) -> sitk.Image:
        """Warp a mask with ``ants.apply_transforms`` and return a ``sitk.Image``."""
        import ants

        transformed_mask = ants.apply_transforms(
            fixed=fixed_ants,
            moving=moving_mask_ants,
            transformlist=transform_files,
            interpolator="nearestNeighbor",
        )
        return ImageConverter.ants_2_itk(transformed_mask)

    def _apply_transform_mask_sitk(
        self,
        fixed_reference: sitk.Image,
        moving_mask: sitk.Image,
        transform_files: List[str],
    ) -> sitk.Image:
        """Resample a moving mask onto the fixed grid using saved SimpleITK transforms."""
        if not transform_files:
            raise ValueError("transform_files must not be empty for SimpleITK mask warping")
        transform = sitk.ReadTransform(transform_files[0])
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_reference)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(transform)
        return resampler.Execute(moving_mask)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Register the specified images to the reference image.

        Args:
            data (Dict[str, Any]): Subject dictionary with ``sitk.Image`` volumes.

        Returns:
            Dict[str, Any]: Same dictionary with registered images and optional transform paths.
        """
        subj = data.get("subj", "unknown")
        logger.debug(f"[{subj}] Registering to {self.fixed_image} (backend={self.backend})")
        self._check_keys(data)

        if self.fixed_image not in data:
            raise KeyError(
                f"Reference key {self.fixed_image} not found in data dictionary"
            )

        fixed_image_sitk: sitk.Image = sitk.Cast(
            data[self.fixed_image], sitk.sitkFloat32
        )
        use_ants_backend = self.backend == "ants"
        fixed_image_ants: Optional[Any] = None
        if use_ants_backend:
            fixed_image_ants = ImageConverter.itk_2_ants(fixed_image_sitk)

        fixed_mask_sitk: Optional[sitk.Image] = None
        fixed_mask_ants: Optional[Any] = None
        if self.use_mask:
            mk_fixed = f"mask_{self.fixed_image}"
            if mk_fixed in data:
                fixed_mask_sitk = sitk.Cast(data[mk_fixed], sitk.sitkUInt8)
                if use_ants_backend:
                    fixed_mask_ants = ImageConverter.itk_2_ants(fixed_mask_sitk)

        for key in self.keys:
            if key == self.fixed_image:
                continue

            moving_sitk: sitk.Image = sitk.Cast(data[key], sitk.sitkFloat32)

            moving_mask_sitk: Optional[sitk.Image] = None
            moving_mask_ants: Optional[Any] = None
            if self.use_mask:
                mk_mov = f"mask_{key}"
                if mk_mov in data:
                    moving_mask_sitk = sitk.Cast(data[mk_mov], sitk.sitkUInt8)
                    if use_ants_backend:
                        moving_mask_ants = ImageConverter.itk_2_ants(moving_mask_sitk)

            try:
                if use_ants_backend:
                    assert fixed_image_ants is not None
                    mov_ants = ImageConverter.itk_2_ants(moving_sitk)
                    registered_image, transform_files = self._register_image(
                        fixed_image_ants,
                        mov_ants,
                        fixed_mask_ants,
                        moving_mask_ants,
                    )
                    registered_sitk: sitk.Image = ImageConverter.ants_2_itk(registered_image)
                else:
                    registered_sitk, transform_files = self._register_image_sitk(
                        fixed_image_sitk,
                        moving_sitk,
                        fixed_mask_sitk,
                        moving_mask_sitk,
                    )

                data[key] = registered_sitk

                transform_key = f"{key}_transform_files"
                data[transform_key] = transform_files

                meta_key = f"{key}_meta_dict"
                if meta_key not in data:
                    data[meta_key] = {}
                data[meta_key]["registered"] = True
                data[meta_key]["fixed_image"] = self.fixed_image
                data[meta_key]["type_of_transform"] = self.type_of_transform
                data[meta_key]["metric"] = self.metric
                data[meta_key]["optimizer"] = self.optimizer
                data[meta_key]["backend"] = self.backend

            except Exception as e:
                logger.error(f"Error registering image {key}: {e}")
                if not self.allow_missing_keys:
                    raise

        # Process each mask image
        for key in self.keys:
            if key == self.fixed_image:
                continue

            mask_key = f"mask_{key}"
            fixed_mask_key = f"mask_{self.fixed_image}"
            transform_key = f"{key}_transform_files"

            if mask_key not in data:
                continue

            if self.replace_by_fixed_image_mask and fixed_mask_key not in data:
                logger.warning(
                    "Warning: Cannot replace mask for %s because fixed mask %s not found.",
                    key,
                    fixed_mask_key,
                )
                continue

            if self.replace_by_fixed_image_mask:
                logger.debug(f"Replacing mask for {key} with fixed image mask")
                fixed_mask = data[fixed_mask_key]
                data[mask_key] = sitk.Cast(fixed_mask, sitk.sitkUInt8)

                meta_key = f"{mask_key}_meta_dict"
                if meta_key not in data:
                    data[meta_key] = {}
                data[meta_key]["registered"] = True
                data[meta_key]["fixed_image"] = self.fixed_image
                data[meta_key]["replaced_by_fixed_mask"] = True
                data[meta_key]["backend"] = self.backend
                continue

            if transform_key not in data:
                logger.warning(
                    f"Warning: No transform files found for {key}. Skipping mask registration."
                )
                continue

            moving_mask = data[mask_key]
            moving_mask = sitk.Cast(moving_mask, sitk.sitkUInt8)
            transform_files: List[str] = data[transform_key]

            try:
                if use_ants_backend:
                    assert fixed_image_ants is not None
                    mov_mask_ants = ImageConverter.itk_2_ants(moving_mask)
                    transformed_mask_sitk = self._apply_transform_mask_ants(
                        fixed_image_ants,
                        mov_mask_ants,
                        transform_files,
                    )
                else:
                    transformed_mask_sitk = self._apply_transform_mask_sitk(
                        fixed_image_sitk,
                        moving_mask,
                        transform_files,
                    )

                transformed_mask_sitk = sitk.Cast(transformed_mask_sitk, sitk.sitkUInt8)
                data[mask_key] = transformed_mask_sitk

                meta_key = f"{mask_key}_meta_dict"
                if meta_key not in data:
                    data[meta_key] = {}
                data[meta_key]["registered"] = True
                data[meta_key]["fixed_image"] = self.fixed_image
                data[meta_key]["type_of_transform"] = self.type_of_transform
                data[meta_key]["metric"] = self.metric
                data[meta_key]["optimizer"] = self.optimizer
                data[meta_key]["backend"] = self.backend

            except Exception as e:
                logger.error(f"Error applying transform to mask {mask_key}: {e}")
