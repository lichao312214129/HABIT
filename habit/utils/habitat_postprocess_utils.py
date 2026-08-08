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
"""
Post-processing utilities for habitat/supervoxel label maps.

Thin facade over :func:`habit.kernels.label_postprocess.remove_small_connected_components`
that preserves the v0 settings dict contract (including ``enabled``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from habit.kernels.label_postprocess import (
    remove_small_connected_components as _remove_small_connected_components,
)
from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)

__all__ = ["remove_small_connected_components"]


def remove_small_connected_components(
    label_map: np.ndarray,
    roi_mask: np.ndarray,
    settings: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Remove tiny connected components by label-wise reassignment in ROI.

    Args:
        label_map: 3D integer label map where 0 means background.
        roi_mask: 3D boolean mask indicating valid ROI.
        settings: Post-process settings dictionary with keys:
            - enabled (bool)
            - min_component_size (int)
            - connectivity (int)

    Returns:
        Cleaned label map with reduced tiny fragments, or the input map when
        cleanup is disabled / unset.
    """
    if settings is None or not bool(settings.get("enabled", False)):
        return label_map

    logger.info(
        "Postprocess start: min_component_size=%s, connectivity=%s",
        settings.get("min_component_size", 30),
        settings.get("connectivity", 1),
    )
    return _remove_small_connected_components(
        label_map=label_map,
        roi_mask=roi_mask,
        settings=settings,
    )
