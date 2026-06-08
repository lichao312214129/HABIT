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
"""
Helpers to shrink fitted :class:`HabitatPipeline` artefacts before persistence.

Predict mode only needs fitted group-level parameters (cluster centres,
preprocessing state, etc.). Training-only payloads such as per-sample cluster
labels and duplicate SimpleITK mask volumes are stripped at save time.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - optional at import time in some test envs
    sitk = None  # type: ignore[assignment]


def apply_mask_metadata_to_sitk_image(
    image: Any,
    mask_info: Dict[str, Any],
) -> None:
    """
    Copy physical space metadata from ``mask_info`` onto a SimpleITK image.

    Supports both legacy entries (with a ``mask`` SimpleITK image) and slim
    entries persisted by :func:`slim_mask_info_for_storage`.

    Args:
        image: SimpleITK image to annotate (modified in place).
        mask_info: Mask metadata dictionary from the pipeline cache.
    """
    mask_img = mask_info.get("mask")
    if mask_img is not None:
        image.CopyInformation(mask_img)
        return

    if "spacing" in mask_info:
        image.SetSpacing(mask_info["spacing"])
    if "origin" in mask_info:
        image.SetOrigin(mask_info["origin"])
    if "direction" in mask_info:
        image.SetDirection(mask_info["direction"])


def slim_mask_info_for_storage(mask_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replace a mask cache entry with a pickle-friendly minimal form.

    Drops the redundant SimpleITK ``mask`` object and keeps ``mask_array`` plus
    the physical-space fields required to write NRRD outputs at predict time.

    Args:
        mask_info: Original mask metadata (``mask`` + ``mask_array``).

    Returns:
        Slim dict with ``mask_array``, ``spacing``, ``origin``, and ``direction``.
    """
    if not isinstance(mask_info, dict):
        return mask_info

    mask_array = mask_info.get("mask_array")
    mask_img = mask_info.get("mask")

    if mask_array is None and mask_img is not None and sitk is not None:
        mask_array = sitk.GetArrayFromImage(mask_img)

    if mask_array is None:
        return dict(mask_info)

    slim: Dict[str, Any] = {"mask_array": mask_array}

    if mask_img is not None:
        slim["spacing"] = mask_img.GetSpacing()
        slim["origin"] = mask_img.GetOrigin()
        slim["direction"] = mask_img.GetDirection()
    else:
        for key in ("spacing", "origin", "direction"):
            if key in mask_info:
                slim[key] = mask_info[key]

    return slim


def strip_clustering_training_artifacts(clusterer: Any) -> None:
    """
    Remove training-only arrays from a fitted clustering wrapper.

    ``predict()`` only needs the underlying sklearn model parameters (e.g.
    cluster centres). Per-sample ``labels_`` arrays can be huge for direct
    pooling and are not serialized intentionally.

    Args:
        clusterer: ``BaseClustering`` instance or compatible wrapper.
    """
    if clusterer is None:
        return

    if hasattr(clusterer, "labels_"):
        clusterer.labels_ = None

    model = getattr(clusterer, "model", None)
    if model is not None and hasattr(model, "labels_"):
        model.labels_ = None


def prepare_pipeline_for_save(pipeline: Any) -> None:
    """
    Strip non-essential in-memory payloads before ``joblib.dump``.

    Mutates ``pipeline`` in place so the on-disk artefact stays small. Mask
    volumes are not persisted; predict-time NRRD export reloads them from
    ``data_dir`` via :meth:`FeatureService.load_mask_info`.

    Args:
        pipeline: Fitted :class:`HabitatPipeline` instance.
    """
    pipeline._train_checkpoint = None
    pipeline.mask_info_cache = {}

    for _, step in getattr(pipeline, "steps", []):
        _prepare_step_for_save(step)


def _prepare_step_for_save(step: Any) -> None:
    clustering_model = getattr(step, "clustering_model", None)
    strip_clustering_training_artifacts(clustering_model)

    clustering_service = getattr(step, "clustering_service", None)
    if clustering_service is not None:
        strip_clustering_training_artifacts(
            getattr(clustering_service, "voxel2supervoxel_clustering", None)
        )
        strip_clustering_training_artifacts(
            getattr(clustering_service, "supervoxel2habitat_clustering", None)
        )

    habitat_image_writer = getattr(step, "habitat_image_writer", None)
    if habitat_image_writer is not None:
        habitat_image_writer.results_df = None
        habitat_image_writer.mask_info_cache = {}
        if hasattr(habitat_image_writer, "_log_queue"):
            habitat_image_writer._log_queue = None

    feature_service = getattr(step, "feature_service", None)
    if feature_service is not None and hasattr(feature_service, "_log_queue"):
        feature_service._log_queue = None
