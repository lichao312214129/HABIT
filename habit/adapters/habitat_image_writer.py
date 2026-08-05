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
"""Habitat label-map writer helpers (L1 adapter).

Provides supervoxel-to-habitat NRRD export without importing ``habit.core``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk

from habit.schemas.workflows.habitat import ResultColumns
from habit.utils.habitat_postprocess_utils import remove_small_connected_components

__all__ = ["save_habitat_from_supervoxel_mapping"]


def save_habitat_from_supervoxel_mapping(
    subject: str,
    habitats_df: pd.DataFrame,
    supervoxel_path: str,
    destination: str,
    postprocess_settings: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save one habitat NRRD by mapping supervoxel labels to habitat labels.

    Used by two_step publish-time batch export when ``*_supervoxel.nrrd`` files
    already exist on disk.

    Args:
        subject: Subject identifier.
        habitats_df: Per-subject rows containing ``supervoxel`` and ``habitats``.
        supervoxel_path: Path to the subject supervoxel NRRD file.
        destination: Output directory for ``<subject>_habitats.nrrd``.
        postprocess_settings: Optional post-process settings dictionary.

    Returns:
        str: Path to the saved habitat NRRD file.
    """
    supervoxel = sitk.ReadImage(supervoxel_path)
    supervoxel_array = sitk.GetArrayFromImage(supervoxel)

    habitats_array = np.zeros_like(supervoxel_array)
    habitats_subj = habitats_df.loc[subject]
    n_clusters_supervoxel = habitats_subj.shape[0]
    for cluster_idx in range(n_clusters_supervoxel):
        supervoxel_id = cluster_idx + 1
        if (supervoxel_array == supervoxel_id).sum() > 0:
            habitat_rows = habitats_subj[
                habitats_subj[ResultColumns.SUPERVOXEL] == supervoxel_id
            ]
            habitats_array[supervoxel_array == supervoxel_id] = habitat_rows[
                ResultColumns.HABITATS
            ].values[0]

    roi_mask = supervoxel_array > 0
    if postprocess_settings and postprocess_settings.get("enabled", False):
        habitats_array = remove_small_connected_components(
            label_map=habitats_array.astype(np.int32, copy=False),
            roi_mask=roi_mask,
            settings=postprocess_settings,
        )

    habitats_img = sitk.GetImageFromArray(habitats_array)
    habitats_img.CopyInformation(supervoxel)

    output_path = os.path.join(destination, f"{subject}_habitats.nrrd")
    sitk.WriteImage(habitats_img, output_path)
    return output_path
