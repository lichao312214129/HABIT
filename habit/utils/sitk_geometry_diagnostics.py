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
SimpleITK physical-space diagnostics (spacing, origin, direction).

Used by the preprocessing batch pipeline to log **which step** first produces
invalid spacing (non-positive or non-finite values), which often precedes
ITK/ANTs warnings about negative spacing at registration time.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Sequence

import SimpleITK as sitk

# Dictionary keys that never hold voxel volumes for geometry checks
_SKIP_EXACT: frozenset[str] = frozenset({"subj", "output_dirs"})


def collect_sitk_geometry_issues(image: sitk.Image) -> List[str]:
    """
    Inspect one volume for invalid physical-space metadata.

    Args:
        image (sitk.Image): SimpleITK image loaded in memory.

    Returns:
        List[str]: Human-readable issue descriptions; empty list if none.
    """
    issues: List[str] = []
    spacing: Sequence[float] = image.GetSpacing()
    for axis_index, spacing_value in enumerate(spacing):
        if math.isnan(spacing_value) or math.isinf(spacing_value):
            issues.append(f"spacing[{axis_index}]={spacing_value} (non-finite)")
        elif spacing_value <= 0:
            issues.append(f"spacing[{axis_index}]={spacing_value} (non-positive)")
    origin: Sequence[float] = image.GetOrigin()
    for axis_index, origin_value in enumerate(origin):
        if math.isnan(origin_value) or math.isinf(origin_value):
            issues.append(f"origin[{axis_index}]={origin_value} (non-finite)")
    direction: Sequence[float] = image.GetDirection()
    for index, direction_value in enumerate(direction):
        if math.isnan(direction_value) or math.isinf(direction_value):
            issues.append(f"direction[{index}]={direction_value} (non-finite)")
    return issues


def log_sitk_geometry_for_subject_data(
    data: Dict[str, Any],
    *,
    checkpoint_label: str,
    subject_id: str,
    logger: logging.Logger,
    log_level: int = logging.WARNING,
) -> None:
    """
    Scan subject ``data`` for ``sitk.Image`` values and log any geometry problems.

    Call this after load and after each preprocessing step. The **first** checkpoint
    where a key shows issues pinpoints the step that introduced bad metadata (check logs
    for the preceding ``checkpoint_label``).

    Args:
        data (Dict[str, Any]): Subject dictionary (modalities, masks, metadata keys).
        checkpoint_label (str): Where this scan runs, e.g. ``after_load_image`` or
            ``after_resample``.
        subject_id (str): Subject identifier for log context.
        logger (logging.Logger): Logger to write to (batch processor logger).
        log_level (int): Level used when issues are found (default WARNING).
    """
    for key, value in data.items():
        if key in _SKIP_EXACT:
            continue
        if key.endswith("_meta_dict") or key.endswith("_transform_files"):
            continue
        if not isinstance(value, sitk.Image):
            continue
        issues: List[str] = collect_sitk_geometry_issues(value)
        if not issues:
            continue
        logger.log(
            log_level,
            "[%s] sitk_geometry checkpoint=%s key=%s dim=%d size=%s spacing=%s | %s",
            subject_id,
            checkpoint_label,
            key,
            value.GetDimension(),
            tuple(value.GetSize()),
            tuple(value.GetSpacing()),
            "; ".join(issues),
        )
