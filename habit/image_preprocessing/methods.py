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
"""Built-in image-volume preprocessors (``preprocessor`` domain)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from habit.contracts.image import ArrayImageRef
from habit.contracts.subject import Subject
from habit.image_preprocessing._subject import (
    mask_array,
    rebuild_subject,
    replace_from_sitk,
    select_modalities,
)
from habit.image_preprocessing.registry import PreprocessorRegistry
from habit.exceptions import HABITAPIError
from habit.kernels.image_clahe import adaptive_histogram_equalize_sitk_image
from habit.kernels.image_histogram import nyul_standardize_volume
from habit.kernels.image_n4 import n4_correct_sitk_image
from habit.kernels.image_reorient import reorient_sitk_image
from habit.kernels.image_resample import resample_sitk_image
from habit.kernels.image_zscore import zscore_normalize_volume
from habit.spec.specs import Spec

__all__ = [
    "AdaptiveHistogramEqualization",
    "HistogramStandardization",
    "N4Correction",
    "Registration",
    "Reorientation",
    "Resample",
    "ZScoreNormalization",
]


def _copy_maps(subject: Subject) -> Tuple[Dict[str, ArrayImageRef], Dict[str, ArrayImageRef]]:
    """Shallow-copy the subject's image and mask ref mappings."""
    return dict(subject.images), dict(subject.masks)


@PreprocessorRegistry.register("zscore_normalization")
class ZScoreNormalization:
    """Z-score intensity normalization (float32; signed values preserved)."""

    def __init__(
        self,
        only_inmask: bool = False,
        mask_key: Optional[str] = None,
        clip_values: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.only_inmask = bool(only_inmask)
        self.mask_key = mask_key
        self.clip_values = clip_values

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="zscore_normalization",
            params={
                "only_inmask": self.only_inmask,
                "mask_key": self.mask_key,
                "clip_values": self.clip_values,
            },
        )

    def __call__(
        self,
        subject: Subject,
        *,
        images: Optional[Sequence[str]] = None,
        mask_roi: Optional[str] = None,
    ) -> Subject:
        """Z-score each selected modality; geometry is unchanged."""
        roi = self.mask_key if self.only_inmask and self.mask_key else (
            mask_roi if self.only_inmask else None
        )
        mask = mask_array(subject, roi) if self.only_inmask else None
        new_images, new_masks = _copy_maps(subject)
        for modality in select_modalities(subject, images):
            volume = subject.image(modality)
            out = zscore_normalize_volume(
                np.asarray(volume.data),
                mask,
                clip_values=self.clip_values,
            )
            new_images[modality] = ArrayImageRef(array=out, geometry=volume.geometry)
        return rebuild_subject(subject, new_images, new_masks)


@PreprocessorRegistry.register("resample")
class Resample:
    """Resample images (and matching masks) to a target voxel spacing."""

    def __init__(
        self,
        target_spacing: Sequence[float] = (1.0, 1.0, 1.0),
        img_mode: str = "bilinear",
        padding_mode: str = "border",
        align_corners: bool = False,
    ) -> None:
        self.target_spacing = tuple(float(v) for v in target_spacing)
        self.img_mode = str(img_mode)
        self.padding_mode = str(padding_mode)
        self.align_corners = bool(align_corners)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="resample",
            params={
                "target_spacing": list(self.target_spacing),
                "img_mode": self.img_mode,
                "padding_mode": self.padding_mode,
                "align_corners": self.align_corners,
            },
        )

    def __call__(
        self,
        subject: Subject,
        *,
        images: Optional[Sequence[str]] = None,
        mask_roi: Optional[str] = None,
    ) -> Subject:
        """Resample selected images; resample every mask onto the new grid."""
        del mask_roi
        new_images, new_masks = _copy_maps(subject)
        modalities = select_modalities(subject, images)
        for modality in modalities:
            sitk_out = resample_sitk_image(
                subject.image(modality).to_sitk(),
                self.target_spacing,
                interpolator=self.img_mode,
            )
            new_images[modality] = replace_from_sitk(
                subject, modality=modality, sitk_image=sitk_out
            )
        for roi_name in list(subject.masks):
            sitk_out = resample_sitk_image(
                subject.mask(roi_name).to_sitk(),
                self.target_spacing,
                interpolator="nearest",
            )
            new_masks[roi_name] = replace_from_sitk(
                subject, modality=roi_name, sitk_image=sitk_out, is_mask=True
            )
        return rebuild_subject(subject, new_images, new_masks)


@PreprocessorRegistry.register("reorientation")
class Reorientation:
    """Reorient images (and masks) to a canonical DICOM orientation."""

    def __init__(
        self,
        target_orientation: str = "LPS",
        mode: str = "closest",
    ) -> None:
        self.target_orientation = str(target_orientation).upper()
        self.mode = str(mode).lower()
        if self.mode not in {"closest", "strict"}:
            raise HABITAPIError("reorientation mode must be 'closest' or 'strict'")

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="reorientation",
            params={
                "target_orientation": self.target_orientation,
                "mode": self.mode,
            },
        )

    def __call__(
        self,
        subject: Subject,
        *,
        images: Optional[Sequence[str]] = None,
        mask_roi: Optional[str] = None,
    ) -> Subject:
        """Reorient selected images and every mask with the same mode."""
        del mask_roi
        new_images, new_masks = _copy_maps(subject)
        for modality in select_modalities(subject, images):
            sitk_out = reorient_sitk_image(
                subject.image(modality).to_sitk(),
                target_orientation=self.target_orientation,
                mode=self.mode,
                is_mask=False,
            )
            new_images[modality] = replace_from_sitk(
                subject, modality=modality, sitk_image=sitk_out
            )
        for roi_name in list(subject.masks):
            sitk_out = reorient_sitk_image(
                subject.mask(roi_name).to_sitk(),
                target_orientation=self.target_orientation,
                mode=self.mode,
                is_mask=True,
            )
            new_masks[roi_name] = replace_from_sitk(
                subject, modality=roi_name, sitk_image=sitk_out, is_mask=True
            )
        return rebuild_subject(subject, new_images, new_masks)


@PreprocessorRegistry.register("n4_correction")
class N4Correction:
    """N4 bias-field correction (SimpleITK)."""

    def __init__(
        self,
        num_fitting_levels: int = 4,
        num_iterations: Optional[List[int]] = None,
        convergence_threshold: float = 0.001,
        shrink_factor: int = 4,
        mask_name: Optional[str] = None,
    ) -> None:
        self.num_fitting_levels = int(num_fitting_levels)
        self.num_iterations = (
            list(num_iterations)
            if num_iterations is not None
            else [50] * self.num_fitting_levels
        )
        self.convergence_threshold = float(convergence_threshold)
        self.shrink_factor = int(shrink_factor)
        # v0.1 batch preprocessing never selected a mask for N4 unless a
        # dedicated mask parameter was supplied. Keep that no-mask default so
        # the v1 atomic operator and the migrated recipe share one explicit,
        # reproducible definition instead of inferring an arbitrary ROI.
        self.mask_name = mask_name

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="n4_correction",
            params={
                "num_fitting_levels": self.num_fitting_levels,
                "num_iterations": list(self.num_iterations),
                "convergence_threshold": self.convergence_threshold,
                "shrink_factor": self.shrink_factor,
                "mask_name": self.mask_name,
            },
        )

    def __call__(
        self,
        subject: Subject,
        *,
        images: Optional[Sequence[str]] = None,
        mask_roi: Optional[str] = None,
    ) -> Subject:
        """Correct each selected modality; only an explicit mask restricts the fit."""
        # ``mask_roi`` is the generic pipeline routing hint used by other
        # preprocessors. N4 deliberately does not consume it: choosing a mask
        # changes the estimated bias field and must therefore be an explicit
        # algorithm parameter recorded in the component Spec.
        del mask_roi
        sitk_mask = (
            subject.mask(self.mask_name).to_sitk()
            if self.mask_name is not None
            else None
        )
        new_images, new_masks = _copy_maps(subject)
        for modality in select_modalities(subject, images):
            sitk_out = n4_correct_sitk_image(
                subject.image(modality).to_sitk(),
                sitk_mask,
                num_fitting_levels=self.num_fitting_levels,
                num_iterations=self.num_iterations,
                convergence_threshold=self.convergence_threshold,
                shrink_factor=self.shrink_factor,
            )
            new_images[modality] = replace_from_sitk(
                subject, modality=modality, sitk_image=sitk_out
            )
        return rebuild_subject(subject, new_images, new_masks)


@PreprocessorRegistry.register("histogram_standardization")
class HistogramStandardization:
    """Nyúl histogram standardization onto a standard intensity scale."""

    def __init__(
        self,
        percentiles: Optional[List[float]] = None,
        target_min: float = 0.0,
        target_max: float = 100.0,
        mask_key: Optional[str] = None,
    ) -> None:
        self.percentiles = (
            [float(p) for p in percentiles]
            if percentiles is not None
            else [1.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 99.0]
        )
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        self.mask_key = mask_key

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="histogram_standardization",
            params={
                "percentiles": list(self.percentiles),
                "target_min": self.target_min,
                "target_max": self.target_max,
                "mask_key": self.mask_key,
            },
        )

    def __call__(
        self,
        subject: Subject,
        *,
        images: Optional[Sequence[str]] = None,
        mask_roi: Optional[str] = None,
    ) -> Subject:
        """Standardize each selected modality; geometry is unchanged."""
        roi = self.mask_key or mask_roi
        mask = mask_array(subject, roi)
        new_images, new_masks = _copy_maps(subject)
        for modality in select_modalities(subject, images):
            volume = subject.image(modality)
            out = nyul_standardize_volume(
                np.asarray(volume.data),
                mask,
                percentiles=self.percentiles,
                target_min=self.target_min,
                target_max=self.target_max,
            )
            new_images[modality] = ArrayImageRef(array=out, geometry=volume.geometry)
        return rebuild_subject(subject, new_images, new_masks)


@PreprocessorRegistry.register("adaptive_histogram_equalization")
class AdaptiveHistogramEqualization:
    """Contrast-limited adaptive histogram equalization (SimpleITK)."""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.3,
        radius: Union[int, Tuple[int, int, int]] = 5,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.radius = radius

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="adaptive_histogram_equalization",
            params={
                "alpha": self.alpha,
                "beta": self.beta,
                "radius": self.radius,
            },
        )

    def __call__(
        self,
        subject: Subject,
        *,
        images: Optional[Sequence[str]] = None,
        mask_roi: Optional[str] = None,
    ) -> Subject:
        """Enhance local contrast of each selected modality."""
        del mask_roi
        new_images, new_masks = _copy_maps(subject)
        for modality in select_modalities(subject, images):
            sitk_out = adaptive_histogram_equalize_sitk_image(
                subject.image(modality).to_sitk(),
                alpha=self.alpha,
                beta=self.beta,
                radius=self.radius,
            )
            new_images[modality] = replace_from_sitk(
                subject, modality=modality, sitk_image=sitk_out
            )
        return rebuild_subject(subject, new_images, new_masks)


@PreprocessorRegistry.register("registration")
class Registration:
    """Register moving modalities onto a fixed reference (ANTs / SitK / elastix)."""

    def __init__(
        self,
        fixed_image: str,
        backend: str = "ants",
        type_of_transform: str = "SyN",
        metric: str = "MI",
        optimizer: Optional[str] = None,
        use_mask: bool = False,
        replace_by_fixed_image_mask: bool = True,
        mask_key: str = "",
        elastix_parameter_files: Optional[str] = None,
        elastix_path: Optional[str] = None,
        transformix_path: Optional[str] = None,
        elastix_threads: int = 0,
        elastix_parameter_overrides: Optional[Dict[str, Any]] = None,
        number_of_histogram_bins: int = 50,
        metric_sampling_percentage: float = 0.01,
        shrink_factors_per_level: Optional[List[int]] = None,
        smoothing_sigmas_per_level: Optional[List[float]] = None,
        learning_rate: float = 1.0,
        number_of_iterations: int = 100,
        bspline_mesh_size: int = 8,
        bspline_order: int = 3,
    ) -> None:
        self.fixed_image = str(fixed_image)
        self.backend = str(backend).strip().lower()
        if self.backend == "elastic":
            self.backend = "elastix"
        self.type_of_transform = str(type_of_transform)
        self.metric = str(metric)
        self.optimizer = optimizer
        self.use_mask = bool(use_mask)
        self.replace_by_fixed_image_mask = bool(replace_by_fixed_image_mask)
        self.mask_key = mask_key
        self.elastix_parameter_files = elastix_parameter_files
        self.elastix_path = elastix_path
        self.transformix_path = transformix_path
        self.elastix_threads = int(elastix_threads)
        self.elastix_parameter_overrides = elastix_parameter_overrides
        self.number_of_histogram_bins = int(number_of_histogram_bins)
        self.metric_sampling_percentage = float(metric_sampling_percentage)
        self.shrink_factors_per_level = (
            list(shrink_factors_per_level)
            if shrink_factors_per_level is not None
            else [4, 2, 1]
        )
        self.smoothing_sigmas_per_level = (
            list(smoothing_sigmas_per_level)
            if smoothing_sigmas_per_level is not None
            else [2.1, 1.0, 0.0]
        )
        self.learning_rate = float(learning_rate)
        self.number_of_iterations = int(number_of_iterations)
        self.bspline_mesh_size = int(bspline_mesh_size)
        self.bspline_order = int(bspline_order)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="registration",
            params={
                "fixed_image": self.fixed_image,
                "backend": self.backend,
                "type_of_transform": self.type_of_transform,
                "metric": self.metric,
                "optimizer": self.optimizer,
                "use_mask": self.use_mask,
                "replace_by_fixed_image_mask": self.replace_by_fixed_image_mask,
            },
        )

    def __call__(
        self,
        subject: Subject,
        *,
        images: Optional[Sequence[str]] = None,
        mask_roi: Optional[str] = None,
    ) -> Subject:
        """Reject retired external registration backends explicitly."""
        del subject, images, mask_roi
        raise HABITAPIError(
            "Image registration is not part of the focused HABIT v2 "
            "preprocessing surface. Use an external registration workflow "
            "before loading images into HABIT."
        )


