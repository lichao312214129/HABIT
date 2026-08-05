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
"""L4 image-preprocessing recipes (thin assembly).

Two surfaces, both free of direct ``habit.compat.engines`` imports (architecture
gate): they delegate to :mod:`habit.api.preprocessing`.

* :func:`preprocess_images` — batch directory pipeline (CLI twin).
* :func:`preprocess_subject` / :func:`preprocess_image` — atomic in-memory
  operators for embedding HABIT in a third-party notebook or pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:
    from habit.api.image import ImageVolume, MaskVolume
    from habit.contracts.subject import Subject

__all__ = ["preprocess_images", "preprocess_subject", "preprocess_image"]


def preprocess_images(
    config: Any,
    *,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """
    Run the batch image-preprocessing pipeline (``habit preprocess`` recipe).

    Args:
        config: Validated preprocessing configuration (v0.1 schema object or
            mapping accepted by
            :class:`~habit.api.preprocessing.PreprocessingConfig`).
        logger: Optional run logger forwarded to the workflow helper.

    Returns:
        :class:`~habit.api.contracts.WorkflowResult` with output directory
        metadata and a run manifest path.
    """
    from habit.api.preprocessing import run_preprocess

    return run_preprocess(config, logger=logger)


def preprocess_subject(
    subject: "Subject",
    steps: Mapping[str, Mapping[str, Any]],
    *,
    mask_roi: Optional[str] = None,
    broadcast_mask: bool = True,
) -> "Subject":
    """
    Apply an ordered image-preprocessing chain to one subject in memory.

    Recipe twin of :func:`habit.api.preprocessing.preprocess_subject`. See
    that function for full argument documentation.

    Args:
        subject: One imaging subject.
        steps: Ordered ``{step_name: params}`` mapping (YAML ``preprocessing``
            block shape).
        mask_roi: Optional ROI key; auto-selected when the subject has exactly
            one mask.
        broadcast_mask: Attach the ROI under every ``mask_<modality>``.

    Returns:
        A new Subject with processed in-memory volumes.
    """
    from habit.api.preprocessing import preprocess_subject as _api_preprocess_subject

    return _api_preprocess_subject(
        subject,
        steps,
        mask_roi=mask_roi,
        broadcast_mask=broadcast_mask,
    )


def preprocess_image(
    image: "ImageVolume",
    steps: Mapping[str, Mapping[str, Any]],
    *,
    mask: Optional["MaskVolume"] = None,
    modality: str = "image",
) -> "ImageVolume":
    """
    Apply an ordered image-preprocessing chain to one volume in memory.

    Recipe twin of :func:`habit.api.preprocessing.preprocess_image`.

    Args:
        image: Intensity volume to process.
        steps: Ordered ``{step_name: params}`` mapping.
        mask: Optional ROI mask.
        modality: Synthetic modality key used internally.

    Returns:
        The processed intensity volume.
    """
    from habit.api.preprocessing import preprocess_image as _api_preprocess_image

    return _api_preprocess_image(
        image, steps, mask=mask, modality=modality
    )
