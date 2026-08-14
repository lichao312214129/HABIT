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
"""Public preprocessing API.

Two complementary entry points:

* :func:`run_preprocess` / recipe twin ``preprocess_images`` — **batch**
  directory pipeline (YAML / ``data_dir`` / ``out_dir``).
* :func:`preprocess_subject` / :func:`preprocess_image` — **atomic**
  in-memory operators that take a :class:`~habit.contracts.Subject` or a
  single :class:`~habit.api.image.ImageVolume` and return a new object. No
  filesystem, YAML, or cohort is required. This is the embedding-ecosystem
  surface: a third-party notebook can call ``preprocess_subject(subject,
  steps)`` on one case without accepting HABIT's directory conventions.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping as MappingABC
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Union

from habit.api.contracts import WorkflowResult, coerce_config
from habit.api.provenance import create_run_manifest, write_run_manifest
from habit.exceptions import HABITAPIError

if TYPE_CHECKING:
    from habit.api.image import ImageVolume, MaskVolume
    from habit.contracts.subject import Subject
    from habit.schemas.workflows.preprocessing import PreprocessingConfig

__all__ = [
    "PreprocessingConfig",
    "run_preprocess",
    "preprocess_subject",
    "preprocess_image",
]

#: Default modality key used by :func:`preprocess_image` when the caller
#: does not supply one. Kept short so logs and mask keys stay readable.
_DEFAULT_SINGLE_MODALITY: str = "image"


def __getattr__(name: str) -> Any:
    if name == "PreprocessingConfig":
        from habit.schemas.workflows.preprocessing import PreprocessingConfig

        return PreprocessingConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_preprocess(
    config: Union["PreprocessingConfig", Mapping[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> WorkflowResult[None]:
    """
    Run the preprocessing batch pipeline from a validated config object.

    Args:
        config: Validated config or dictionary accepted by
            :class:`~habit.schemas.workflows.preprocessing.PreprocessingConfig`.
        logger: Optional logger; core runner creates one when omitted.

    Returns:
        A result with the workflow output directory in ``artifacts``.
    """
    from habit.compat.legacy_core import run_preprocess_from_config
    from habit.schemas.workflows.preprocessing import PreprocessingConfig

    validated_config = coerce_config(config, PreprocessingConfig)
    run_preprocess_from_config(validated_config, logger=logger)
    manifest = create_run_manifest("preprocess", validated_config)
    manifest_path = write_run_manifest(manifest, validated_config.out_dir)
    return WorkflowResult(
        output_dir=validated_config.out_dir,
        metadata={
            "config_hash": manifest.config_hash,
            "habit_version": manifest.habit_version,
        },
        run_id=manifest.run_id,
        manifest_path=manifest_path,
    )


def preprocess_subject(
    subject: "Subject",
    steps: Mapping[str, Mapping[str, Any]],
    *,
    mask_roi: Optional[str] = None,
    broadcast_mask: bool = True,
) -> "Subject":
    """
    Apply an ordered image-preprocessing chain to one subject in memory.

    This is the subject-level atomic twin of the batch directory pipeline.
    Callers pass a :class:`~habit.contracts.Subject` and an ordered mapping
    of step name → parameters (same shape as the ``preprocessing:`` block
    in a v0.1/v1 YAML). The function returns a **new** Subject whose images
    and masks are in-memory :class:`~habit.contracts.ArrayImageRef` values;
    the input subject is never mutated.

    No ``data_dir``, ``out_dir``, YAML file, cohort, or execution backend is
    required — a notebook can debug one failing case with::

        processed = preprocess_subject(cohort[0], {"resample": {...}})

    Args:
        subject: One imaging subject (lazy file refs or in-memory arrays).
        steps: Ordered mapping ``{step_name: params}``. ``params`` may omit
            ``images``; when omitted, every modality on ``subject`` is used.
            Step names match the ``preprocessor`` registry
            (``resample``, ``zscore_normalization``, ``n4_correction``, …).
        mask_roi: ROI key in ``subject.masks`` to attach as
            ``mask_<modality>`` for steps that need a mask. When ``None``
            and the subject has exactly one mask, that ROI is used.
        broadcast_mask: When ``True`` (default), the chosen ROI mask is
            attached under ``mask_<modality>`` for every processed modality
            (the usual multi-modal / single-ROI layout). When ``False``,
            only ``mask_<roi_name>`` is attached when ``roi_name`` equals a
            modality name.

    Returns:
        A new Subject with processed in-memory volumes.

    Raises:
        HABITAPIError: If ``steps`` is empty, a step name is unknown, or
            mask resolution is ambiguous.
        KeyError: If a requested modality or ROI is absent on ``subject``.
    """
    if not isinstance(steps, MappingABC) or not steps:
        raise HABITAPIError(
            "preprocess_subject requires a non-empty ordered mapping of "
            "step_name -> parameter dict (same shape as the YAML "
            "'preprocessing:' block)."
        )

    # Import registers every built-in v1 image preprocessor.
    from habit.domain.image_preprocessing import PreprocessorRegistry

    modalities: list[str] = list(subject.images.keys())
    if not modalities:
        raise HABITAPIError(
            f"Subject {subject.subject_id!r} has no images to preprocess."
        )

    roi_name = _resolve_mask_roi(subject, mask_roi=mask_roi)
    current = subject
    for step_name, raw_params in steps.items():
        if not isinstance(step_name, str) or not step_name.strip():
            raise HABITAPIError(f"Invalid preprocessing step name: {step_name!r}.")
        params = dict(raw_params or {})
        step_modalities = list(params.pop("images", modalities))
        params.pop("keys", None)
        params.pop("allow_missing_keys", None)
        params_model = PreprocessorRegistry.params_model(step_name)
        if params_model is not None:
            allowed = set(params_model.model_fields)
            params = {key: value for key, value in params.items() if key in allowed}
        if not step_modalities:
            raise HABITAPIError(f"Step {step_name!r} has an empty 'images' list.")
        missing = [m for m in step_modalities if m not in current.images]
        if missing:
            raise HABITAPIError(
                f"Step {step_name!r} references modalities absent from "
                f"subject {subject.subject_id!r}: {missing}. "
                f"Available: {sorted(current.images)}."
            )
        available = set(PreprocessorRegistry.available())
        if step_name not in available:
            raise HABITAPIError(
                f"Unknown image preprocessor {step_name!r}. "
                f"Available: {sorted(available)}."
            )
        processor = PreprocessorRegistry.create(name=step_name, **params)
        current = processor(
            current,
            images=step_modalities,
            mask_roi=roi_name,
        )

    # broadcast_mask is retained for API compatibility; v1 steps already
    # operate on Subject.masks rather than duplicated mask_<modality> keys.
    del broadcast_mask
    return current


def preprocess_image(
    image: "ImageVolume",
    steps: Mapping[str, Mapping[str, Any]],
    *,
    mask: Optional["MaskVolume"] = None,
    modality: str = _DEFAULT_SINGLE_MODALITY,
) -> "ImageVolume":
    """
    Apply an ordered image-preprocessing chain to one volume in memory.

    Thin wrapper around :func:`preprocess_subject` for callers who hold a
    single :class:`~habit.api.image.ImageVolume` (and optional mask) rather
    than a full Subject. Useful when embedding HABIT next to MONAI / SimpleITK
    code that already works with individual volumes.

    Args:
        image: Intensity volume to process.
        steps: Ordered mapping ``{step_name: params}``. ``images`` inside
            each step is rewritten to ``[modality]`` automatically.
        mask: Optional ROI mask sharing geometry with ``image``.
        modality: Synthetic modality key used inside the subject wrapper
            (default ``\"image\"``).

    Returns:
        The processed intensity volume (new object; input is not mutated).

    Raises:
        HABITAPIError: Propagated from :func:`preprocess_subject`.
    """
    from habit.contracts.image import ArrayImageRef
    from habit.contracts.geometry import Geometry
    from habit.contracts.subject import Subject

    if not modality or not str(modality).strip():
        raise HABITAPIError("modality must be a non-empty string.")

    geometry = Geometry(
        shape=tuple(int(v) for v in image.data.shape),
        spacing=tuple(float(v) for v in image.spacing),
        origin=tuple(float(v) for v in image.origin),
        direction=tuple(float(v) for v in image.direction),
    )
    images = {modality: ArrayImageRef(array=image.data, geometry=geometry)}
    masks: Dict[str, ArrayImageRef] = {}
    if mask is not None:
        mask_geometry = Geometry(
            shape=tuple(int(v) for v in mask.data.shape),
            spacing=tuple(float(v) for v in mask.spacing),
            origin=tuple(float(v) for v in mask.origin),
            direction=tuple(float(v) for v in mask.direction),
        )
        masks[modality] = ArrayImageRef(array=mask.data, geometry=mask_geometry)

    subject = Subject(
        subject_id=str(image.subject_id or "image"),
        images=images,
        masks=masks,
    )
    # Force every step onto the single synthetic modality key.
    normalized_steps: Dict[str, Dict[str, Any]] = {}
    for step_name, raw_params in steps.items():
        params = dict(raw_params or {})
        params["images"] = [modality]
        normalized_steps[step_name] = params

    processed = preprocess_subject(
        subject,
        normalized_steps,
        mask_roi=modality if mask is not None else None,
        broadcast_mask=True,
    )
    return processed.image(modality)


# ---------------------------------------------------------------------------
# Internal helpers (Subject <-> SimpleITK subject_data dict)
# ---------------------------------------------------------------------------


def _resolve_mask_roi(
    subject: "Subject",
    *,
    mask_roi: Optional[str],
) -> Optional[str]:
    """
    Resolve which ROI key to feed to mask-aware preprocessors.

    Args:
        subject: Source subject.
        mask_roi: Explicit ROI name, or ``None`` to auto-select.

    Returns:
        ROI name, or ``None`` when the subject has no masks.

    Raises:
        HABITAPIError: If ``mask_roi`` is missing, or auto-select is
            ambiguous (more than one ROI and none named).
    """
    if not subject.masks:
        if mask_roi is not None:
            raise HABITAPIError(
                f"mask_roi={mask_roi!r} was requested but subject "
                f"{subject.subject_id!r} has no masks."
            )
        return None
    if mask_roi is not None:
        if mask_roi not in subject.masks:
            raise HABITAPIError(
                f"Subject {subject.subject_id!r} has no ROI {mask_roi!r}. "
                f"Available: {sorted(subject.masks)}."
            )
        return mask_roi
    if len(subject.masks) == 1:
        return next(iter(subject.masks))
    raise HABITAPIError(
        f"Subject {subject.subject_id!r} has {len(subject.masks)} masks "
        f"({sorted(subject.masks)}); pass mask_roi explicitly."
    )


def _subject_to_sitk_dict(
    subject: "Subject",
    *,
    modalities: Sequence[str],
    roi_name: Optional[str],
    broadcast_mask: bool,
) -> Dict[str, Any]:
    """
    Materialise a Subject into the SimpleITK ``subject_data`` dict that the
    legacy preprocessor chain expects.

    Args:
        subject: Source subject.
        modalities: Modality keys to load.
        roi_name: Resolved ROI name, or ``None``.
        broadcast_mask: Attach the ROI under every ``mask_<modality>``.

    Returns:
        Mutable dict with ``subj``, modality → ``sitk.Image``, and
        ``mask_<modality>`` entries.
    """
    data: Dict[str, Any] = {"subj": subject.subject_id, "output_dirs": {}}
    for modality in modalities:
        volume = subject.image(modality)
        data[modality] = volume.to_sitk()

    if roi_name is None:
        return data

    mask_volume = subject.mask(roi_name)
    sitk_mask = mask_volume.to_sitk()
    if broadcast_mask:
        for modality in modalities:
            data[f"mask_{modality}"] = sitk_mask
    else:
        data[f"mask_{roi_name}"] = sitk_mask
    return data


def _sitk_to_image_volume(
    sitk_image: Any,
    *,
    modality: str,
    subject_id: str,
) -> "ImageVolume":
    """Convert a SimpleITK image (or ndarray) back to ImageVolume."""
    from habit.api.image import ImageVolume
    import SimpleITK as sitk
    import numpy as np

    if isinstance(sitk_image, sitk.Image):
        return ImageVolume.from_sitk(
            sitk_image, modality=modality, subject_id=subject_id
        )
    if isinstance(sitk_image, np.ndarray):
        return ImageVolume.from_array(
            sitk_image, modality=modality, subject_id=subject_id
        )
    raise HABITAPIError(
        f"Preprocessor left modality {modality!r} as unsupported type "
        f"{type(sitk_image)!r}; expected SimpleITK.Image or ndarray."
    )


def _sitk_to_mask_volume(
    sitk_image: Any,
    *,
    roi_name: str,
    subject_id: str,
) -> "MaskVolume":
    """Convert a SimpleITK mask (or ndarray) back to MaskVolume."""
    from habit.api.image import MaskVolume
    import SimpleITK as sitk
    import numpy as np

    if isinstance(sitk_image, sitk.Image):
        return MaskVolume.from_sitk(
            sitk_image, modality=roi_name, subject_id=subject_id
        )
    if isinstance(sitk_image, np.ndarray):
        return MaskVolume.from_array(
            sitk_image, modality=roi_name, subject_id=subject_id
        )
    raise HABITAPIError(
        f"Preprocessor left mask {roi_name!r} as unsupported type "
        f"{type(sitk_image)!r}; expected SimpleITK.Image or ndarray."
    )


def _first_mask_key(
    subject_data: Mapping[str, Any],
    modalities: Sequence[str],
) -> Optional[str]:
    """Return the first ``mask_<modality>`` key present in ``subject_data``."""
    for modality in modalities:
        key = f"mask_{modality}"
        if key in subject_data:
            return key
    for key in subject_data:
        if isinstance(key, str) and key.startswith("mask_"):
            return key
    return None
