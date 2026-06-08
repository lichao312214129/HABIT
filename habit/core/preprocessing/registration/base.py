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

class BaseRegistrationBackend:
    """Base class for all registration backends."""

    def __init__(
        self,
        fixed_image_key: str,
        type_of_transform: str,
        metric: str,
        optimizer: Optional[str],
        reg_params: Dict[str, Any],
        sitk_reg_params: Dict[str, Any],
    ) -> None:
        self.fixed_image_key = fixed_image_key
        self.type_of_transform = type_of_transform
        self.metric = metric
        self.optimizer = optimizer
        self.reg_params = reg_params
        self.sitk_reg_params = sitk_reg_params

    def register_image(
        self,
        fixed_image_sitk: sitk.Image,
        moving_image_sitk: sitk.Image,
        fixed_mask_sitk: Optional[sitk.Image] = None,
        moving_mask_sitk: Optional[sitk.Image] = None,
        fixed_image_ants: Optional[Any] = None,
    ) -> Tuple[sitk.Image, List[str]]:
        """Register a moving image to a fixed image.

        Returns:
            Tuple[sitk.Image, List[str]]: Registered sitk.Image and list of forward transform paths.
        """
        raise NotImplementedError

    def apply_transform_mask(
        self,
        fixed_reference_sitk: sitk.Image,
        moving_mask_sitk: sitk.Image,
        transform_files: List[str],
        fixed_image_ants: Optional[Any] = None,
    ) -> sitk.Image:
        """Apply transform(s) to a mask image.

        Returns:
            sitk.Image: Mask resampled onto the fixed grid.
        """
        raise NotImplementedError
