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
"""Habit supervoxel_radiomics settings merged from habitat YAML into PyRadiomics settings."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

# Keys under feature_construction.supervoxel_level.params forwarded into settings dict.
SUPERVOXEL_SETTING_KEYS: Tuple[str, ...] = (
    "supervoxel_union_bbox_crop",
    "supervoxel_pad_distance",
    "use_supervoxel_cext",
    "union_bin",
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
