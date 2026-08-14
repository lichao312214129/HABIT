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
"""Apply image registration to a Subject (SitK kernel or ANTs/elastix backends)."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Mapping, Optional, Sequence

import SimpleITK as sitk

from habit.contracts.image import ArrayImageRef, ImageVolume, MaskVolume
from habit.contracts.subject import Subject
from habit.exceptions import HABITAPIError
from habit.kernels.image_registration import register_sitk_image, warp_sitk_mask

__all__ = ["apply_registration_to_subject"]


def _ref_from_sitk(
    subject: Subject, sitk_image: sitk.Image, *, name: str, is_mask: bool
) -> ArrayImageRef:
    """Wrap a SimpleITK result as an in-memory image/mask reference."""
    if is_mask:
        volume = MaskVolume.from_sitk(
            sitk_image, modality=name, subject_id=subject.subject_id
        )
    else:
        volume = ImageVolume.from_sitk(
            sitk_image, modality=name, subject_id=subject.subject_id
        )
    return ArrayImageRef(array=volume.data, geometry=volume.geometry)


def _apply_simpleitk(
    subject: Subject,
    *,
    images: Sequence[str],
    mask_roi: Optional[str],
    fixed_image: str,
    type_of_transform: str,
    metric: str,
    optimizer: Optional[str],
    use_mask: bool,
    replace_by_fixed_image_mask: bool,
    sitk_params: Mapping[str, Any],
) -> Subject:
    """Register moving modalities onto ``fixed_image`` with the SitK kernel."""
    if fixed_image not in subject.images:
        raise HABITAPIError(
            f"registration fixed_image={fixed_image!r} is not on subject "
            f"{subject.subject_id!r}. Available: {sorted(subject.images)}."
        )
    fixed = sitk.Cast(subject.image(fixed_image).to_sitk(), sitk.sitkFloat32)
    fixed_mask = None
    if use_mask and mask_roi and mask_roi in subject.masks:
        fixed_mask = sitk.Cast(subject.mask(mask_roi).to_sitk(), sitk.sitkUInt8)

    new_images: Dict[str, ArrayImageRef] = dict(subject.images)
    new_masks: Dict[str, ArrayImageRef] = dict(subject.masks)
    transforms: Dict[str, List[str]] = {}
    for modality in images:
        if modality == fixed_image:
            continue
        moving = sitk.Cast(subject.image(modality).to_sitk(), sitk.sitkFloat32)
        moving_mask = None
        if use_mask and mask_roi and mask_roi in subject.masks:
            moving_mask = sitk.Cast(subject.mask(mask_roi).to_sitk(), sitk.sitkUInt8)
        registered, tfm = register_sitk_image(
            fixed,
            moving,
            type_of_transform=type_of_transform,
            metric=metric,
            optimizer=optimizer,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            sitk_params=dict(sitk_params),
        )
        new_images[modality] = _ref_from_sitk(
            subject, registered, name=modality, is_mask=False
        )
        transforms[modality] = tfm

    for roi_name in list(subject.masks):
        if replace_by_fixed_image_mask and mask_roi and mask_roi in subject.masks:
            new_masks[roi_name] = new_masks[mask_roi]
            continue
        # Warp each mask with the first moving transform when present.
        moving_key = next((m for m in images if m != fixed_image), None)
        if moving_key is None or moving_key not in transforms:
            continue
        warped = warp_sitk_mask(
            fixed,
            sitk.Cast(subject.mask(roi_name).to_sitk(), sitk.sitkUInt8),
            transforms[moving_key],
        )
        new_masks[roi_name] = _ref_from_sitk(
            subject, warped, name=roi_name, is_mask=True
        )
    return dataclasses.replace(subject, images=new_images, masks=new_masks)


def apply_registration_to_subject(
    subject: Subject,
    *,
    images: Sequence[str],
    mask_roi: Optional[str],
    fixed_image: str,
    backend: str,
    type_of_transform: str,
    metric: str,
    optimizer: Optional[str],
    use_mask: bool,
    replace_by_fixed_image_mask: bool,
    elastix_parameter_files: Optional[str],
    elastix_path: Optional[str],
    transformix_path: Optional[str],
    elastix_threads: int,
    elastix_parameter_overrides: Optional[Dict[str, Any]],
    sitk_params: Mapping[str, Any],
) -> Subject:
    """
    Register moving modalities of ``subject`` onto ``fixed_image``.

    The SimpleITK backend is a kernel. ANTs / elastix reuse the v0.1
    backends so those optional stacks stay numerically identical.

    Args:
        subject: One imaging subject.
        images: Modality keys to treat as moving (plus the fixed key).
        mask_roi: Optional ROI used when ``use_mask`` is True.
        fixed_image: Reference modality key.
        backend: ``simpleitk``, ``ants``, or ``elastix``.
        type_of_transform: Transform / registration model name.
        metric: Similarity metric.
        optimizer: Optional optimizer hint.
        use_mask: Restrict the metric with ``mask_roi``.
        replace_by_fixed_image_mask: Copy the fixed mask onto moving ROIs.
        elastix_parameter_files: Optional elastix parameter file.
        elastix_path: Optional elastix executable.
        transformix_path: Optional transformix executable.
        elastix_threads: Elastix thread count (0 = default).
        elastix_parameter_overrides: Optional elastix parameter dict.
        sitk_params: SimpleITK tuning parameters.

    Returns:
        A new subject with registered volumes.
    """
    name = (backend or "simpleitk").strip().lower()
    if name == "simpleitk":
        return _apply_simpleitk(
            subject,
            images=images,
            mask_roi=mask_roi,
            fixed_image=fixed_image,
            type_of_transform=type_of_transform,
            metric=metric,
            optimizer=optimizer,
            use_mask=use_mask,
            replace_by_fixed_image_mask=replace_by_fixed_image_mask,
            sitk_params=sitk_params,
        )

    # ANTs / elastix: same backends the batch YAML pipeline uses.
    from habit.compat.engines.preprocessing.registration.registration_preprocessor import (
        RegistrationPreprocessor,
    )

    data: Dict[str, Any] = {"subj": subject.subject_id}
    for modality in images:
        data[modality] = subject.image(modality).to_sitk()
    if mask_roi and mask_roi in subject.masks:
        sitk_mask = subject.mask(mask_roi).to_sitk()
        for modality in images:
            data[f"mask_{modality}"] = sitk_mask

    extra: Dict[str, Any] = dict(sitk_params)
    if elastix_parameter_files is not None:
        extra["elastix_parameter_files"] = elastix_parameter_files
    if elastix_path is not None:
        extra["elastix_path"] = elastix_path
    if transformix_path is not None:
        extra["transformix_path"] = transformix_path
    extra["elastix_threads"] = elastix_threads
    if elastix_parameter_overrides is not None:
        extra["elastix_parameter_overrides"] = elastix_parameter_overrides

    processor = RegistrationPreprocessor(
        keys=list(images),
        fixed_image=fixed_image,
        type_of_transform=type_of_transform,
        metric=metric,
        optimizer=optimizer,
        use_mask=use_mask,
        replace_by_fixed_image_mask=replace_by_fixed_image_mask,
        backend=name,
        **extra,
    )
    processor(data)

    new_images: Dict[str, ArrayImageRef] = dict(subject.images)
    new_masks: Dict[str, ArrayImageRef] = dict(subject.masks)
    for modality in images:
        new_images[modality] = _ref_from_sitk(
            subject, data[modality], name=modality, is_mask=False
        )
    if mask_roi and f"mask_{images[0]}" in data:
        new_masks[mask_roi] = _ref_from_sitk(
            subject, data[f"mask_{images[0]}"], name=mask_roi, is_mask=True
        )
    return dataclasses.replace(subject, images=new_images, masks=new_masks)
