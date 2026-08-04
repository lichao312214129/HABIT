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
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import SimpleITK as sitk

from habit.exceptions import OptionalDependencyError
from habit.utils.image_converter import ImageConverter
from habit.core.preprocessing.registration.base import BaseRegistrationBackend
from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)

class AntsRegistrationBackend(BaseRegistrationBackend):
    """Registration backend using ANTsPy."""

    def register_image(
        self,
        fixed_image_sitk: sitk.Image,
        moving_image_sitk: sitk.Image,
        fixed_mask_sitk: Optional[sitk.Image] = None,
        moving_mask_sitk: Optional[sitk.Image] = None,
        fixed_image_ants: Optional[Any] = None,
    ) -> Tuple[sitk.Image, List[str]]:
        try:
            import ants
        except ImportError as exc:
            raise OptionalDependencyError(
                "ANTs registration requires the optional antspyx dependency; "
                "install 'HABIT[registration]' to use it."
            ) from exc

        moving_image_ants = ImageConverter.itk_2_ants(moving_image_sitk)

        moving_mask_ants = None
        if moving_mask_sitk is not None:
            moving_mask_ants = ImageConverter.itk_2_ants(moving_mask_sitk)
        
        fixed_mask_ants = None
        if fixed_mask_sitk is not None:
            fixed_mask_ants = ImageConverter.itk_2_ants(fixed_mask_sitk)

        reg_params: Dict[str, Any] = {
            "metric": self.metric,
            "optimizer": self.optimizer,
            **self.reg_params,
        }

        if fixed_mask_ants is not None:
            reg_params["mask"] = fixed_mask_ants
        if moving_mask_ants is not None:
            reg_params["moving_mask"] = moving_mask_ants

        if fixed_image_ants is None:
             fixed_image_ants = ImageConverter.itk_2_ants(fixed_image_sitk)

        reg_result: Dict[str, Any] = ants.registration(
            fixed=fixed_image_ants,
            moving=moving_image_ants,
            type_of_transform=self.type_of_transform,
            **reg_params,
        )

        registered_ants = reg_result["warpedmovout"]
        transform_files = reg_result["fwdtransforms"]

        registered_sitk = ImageConverter.ants_2_itk(registered_ants)
        # Preserve original spacing/origin
        registered_sitk.SetSpacing(fixed_image_sitk.GetSpacing())
        registered_sitk.SetOrigin(fixed_image_sitk.GetOrigin())
        registered_sitk.SetDirection(fixed_image_sitk.GetDirection())
        
        return registered_sitk, transform_files

    def apply_transform_mask(
        self,
        fixed_reference_sitk: sitk.Image,
        moving_mask_sitk: sitk.Image,
        transform_files: List[str],
        fixed_image_ants: Optional[Any] = None,
    ) -> sitk.Image:
        try:
            import ants
        except ImportError as exc:
            raise OptionalDependencyError(
                "ANTs transform application requires the optional antspyx "
                "dependency; install 'HABIT[registration]' to use it."
            ) from exc

        moving_mask_ants = ImageConverter.itk_2_ants(moving_mask_sitk)
        
        if fixed_image_ants is None:
            fixed_image_ants = ImageConverter.itk_2_ants(fixed_reference_sitk)

        transformed_mask = ants.apply_transforms(
            fixed=fixed_image_ants,
            moving=moving_mask_ants,
            transformlist=transform_files,
            interpolator="nearestNeighbor",
        )
        return ImageConverter.ants_2_itk(transformed_mask)
