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
