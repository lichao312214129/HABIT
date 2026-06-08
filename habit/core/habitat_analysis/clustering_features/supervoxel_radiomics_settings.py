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
"""Habit supervoxel_radiomics settings merged from habitat YAML into PyRadiomics settings."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

# Keys under FeatureConstruction.supervoxel_level.params forwarded into settings dict.
SUPERVOXEL_SETTING_KEYS: Tuple[str, ...] = (
    "supervoxelUnionBboxCrop",
    "supervoxelPadDistance",
    "useSupervoxelCext",
)


def merge_supervoxel_settings(
    extractor_settings: Mapping[str, object],
    kwargs: Mapping[str, object],
) -> Dict[str, object]:
    """
    Merge habit supervoxel extraction keys from YAML kwargs into settings.

    Args:
        extractor_settings: Settings loaded from ``params_file``.
        kwargs: Resolved ``supervoxel_level.params`` forwarded by FeatureService.

    Returns:
        Dict[str, object]: Settings passed to batched supervoxel radiomics helpers.
    """
    settings = dict(extractor_settings)
    for key in SUPERVOXEL_SETTING_KEYS:
        if key in kwargs:
            settings[key] = kwargs[key]
    return settings
