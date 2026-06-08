# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import SimpleITK as sitk

from ..base_preprocessor import BasePreprocessor
from ..preprocessor_factory import PreprocessorFactory
from .base import BaseRegistrationBackend
from .ants_backend import AntsRegistrationBackend
from .elastix_backend import ElastixRegistrationBackend
from .sitk_backend import SitkRegistrationBackend
from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)

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
    """Register moving images to a fixed reference using ANTsPy, SimpleITK, or elastix CLI.

    Pipelines store ``sitk.Image`` volumes in the subject dictionary. The ``ants``
    backend converts to ``ANTsImage`` for ``ants.registration`` / ``ants.apply_transforms``.
    The ``simpleitk`` backend runs ``sitk.ImageRegistrationMethod`` and warps masks with
    ``sitk.ResampleImageFilter``, avoiding ANTsPy when it is unstable or unavailable.
    The ``elastix`` backend shells out to **elastix** / **transformix** (install the official
    binaries and optionally set ``elastix_path`` / ``transformix_path`` in YAML). Parameters can be
    supplied as a standard elastix ``.txt`` file (e.g. from the elastix Model Zoo) via
    ``elastix_parameter_files``, as an override dict via ``elastix_parameter_overrides``,
    or both combined.
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
            backend (str): ``"ants"`` (default), ``"simpleitk"``, or ``"elastix"``.
                ``elastix`` runs the **elastix** / **transformix** executables (see elastix manual;
                optional YAML keys ``elastix_path``, ``transformix_path``, ``elastix_threads``).
                Pass
                ``elastix_parameter_files`` (path or list of paths to elastix ``.txt`` files,
                e.g. from the elastix Model Zoo) and/or ``elastix_parameter_overrides`` (dict
                of parameter key → value) as additional kwargs to configure registration.
            **kwargs: Extra parameters. SimpleITK-only tuning keys (see module constant
                ``_SITK_OPTION_KEYS``) are stripped and applied when ``backend`` is
                ``"simpleitk"``; for ``"elastix"``, ``elastix_parameter_files`` and
                ``elastix_parameter_overrides`` are consumed by the elastix registration backend.
                Remaining keys go to ``ants.registration`` only for ``backend="ants"``.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)

        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.fixed_image = fixed_image

        if mask_keys is None:
            self.mask_keys = None
        else:
            self.mask_keys = [mask_keys] if isinstance(mask_keys, str) else mask_keys

        self.type_of_transform = type_of_transform
        self.metric = metric
        self.optimizer = optimizer
        self.use_mask = use_mask
        self.replace_by_fixed_image_mask = replace_by_fixed_image_mask

        b = (backend or "ants").strip().lower()
        if b == "elastic":
            logger.warning(
                "backend='elastic' is deprecated; use backend='elastix' (elastix / transformix CLI).",
            )
            b = "elastix"
        if b in {"elaxtic", "elactic"}:
            logger.warning(
                "backend=%r is not a supported name; using 'elastix' (elastix / transformix CLI).",
                backend,
            )
            b = "elastix"
        if b not in {"ants", "simpleitk", "elastix"}:
            raise ValueError(
                f"backend must be 'ants', 'simpleitk', or 'elastix', got {backend!r}"
            )
        self.backend = b

        sitk_params: Dict[str, Any] = {}
        reg_params: Dict[str, Any] = dict(kwargs)
        for _key in _SITK_OPTION_KEYS:
            if _key in reg_params:
                sitk_params[_key] = reg_params.pop(_key)

        self._backend_impl: BaseRegistrationBackend = self._build_backend(
            b, fixed_image, type_of_transform, metric, optimizer, reg_params, sitk_params
        )

    @staticmethod
    def _build_backend(
        name: str,
        fixed_image_key: str,
        type_of_transform: str,
        metric: str,
        optimizer: Optional[str],
        reg_params: Dict[str, Any],
        sitk_params: Dict[str, Any],
    ) -> BaseRegistrationBackend:
        common = dict(
            fixed_image_key=fixed_image_key,
            type_of_transform=type_of_transform,
            metric=metric,
            optimizer=optimizer,
            reg_params=reg_params,
            sitk_reg_params=sitk_params,
        )
        if name == "ants":
            return AntsRegistrationBackend(**common)
        if name == "simpleitk":
            return SitkRegistrationBackend(**common)
        return ElastixRegistrationBackend(**common)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Register the specified images to the reference image.

        Args:
            data (Dict[str, Any]): Subject dictionary with ``sitk.Image`` volumes.

        Returns:
            Dict[str, Any]: Same dictionary with registered images and optional transform paths.
            For ``backend: elastix``, ``*_transform_files`` are removed after this step because
            transform parameters lived only under deleted scratch directories.
        """
        try:
            subj = data.get("subj", "unknown")
            logger.debug(f"[{subj}] Registering to {self.fixed_image} (backend={self.backend})")
            self._check_keys(data)

            if self.fixed_image not in data:
                raise KeyError(
                    f"Reference key {self.fixed_image} not found in data dictionary"
                )

            fixed_image_sitk: sitk.Image = sitk.Cast(data[self.fixed_image], sitk.sitkFloat32)

            fixed_image_ants: Optional[Any] = None
            if self.backend == "ants":
                from habit.utils.image_converter import ImageConverter
                fixed_image_ants = ImageConverter.itk_2_ants(fixed_image_sitk)

            fixed_mask_sitk: Optional[sitk.Image] = None
            if self.use_mask:
                mk_fixed = f"mask_{self.fixed_image}"
                if mk_fixed in data:
                    fixed_mask_sitk = sitk.Cast(data[mk_fixed], sitk.sitkUInt8)

            for key in self.keys:
                if key == self.fixed_image:
                    continue

                try:
                    moving_sitk: sitk.Image = sitk.Cast(data[key], sitk.sitkFloat32)

                    moving_mask_sitk: Optional[sitk.Image] = None
                    if self.use_mask:
                        mk_moving = f"mask_{key}"
                        if mk_moving in data:
                            moving_mask_sitk = sitk.Cast(data[mk_moving], sitk.sitkUInt8)

                    registered_sitk, transform_files = self._backend_impl.register_image(
                        fixed_image_sitk=fixed_image_sitk,
                        moving_image_sitk=moving_sitk,
                        fixed_mask_sitk=fixed_mask_sitk,
                        moving_mask_sitk=moving_mask_sitk,
                        fixed_image_ants=fixed_image_ants,
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
                    transformed_mask_sitk = self._backend_impl.apply_transform_mask(
                        fixed_reference_sitk=fixed_image_sitk,
                        moving_mask_sitk=moving_mask,
                        transform_files=transform_files,
                        fixed_image_ants=fixed_image_ants,
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
                    if not self.allow_missing_keys:
                        raise

            return data
        finally:
            cleanup = getattr(self._backend_impl, "cleanup_elastix_work_dirs", None)
            if callable(cleanup):
                cleanup()
                if self.backend == "elastix":
                    for key in self.keys:
                        if key == self.fixed_image:
                            continue
                        data.pop(f"{key}_transform_files", None)
