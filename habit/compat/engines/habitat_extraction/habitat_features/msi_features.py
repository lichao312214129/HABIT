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
MSI (Multiregional Spatial Interaction) Features Extraction.

Computation is delegated to the vectorised L0 kernels in
:mod:`habit.kernels.habitat_metrics` (same formulas as the historical
pure-Python triple loop, ~40x faster on typical habitat volumes). The
public class API is unchanged so CLI / plugin callers keep working.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np
import SimpleITK as sitk

from habit.kernels.habitat_metrics import (
    msi_features_from_matrix,
    spatial_interaction_matrix,
)
from habit.utils.log_utils import get_module_logger

logger = get_module_logger(__name__)


class MSIFeatureExtractor:
    """Extractor class for MSI features (compat facade over L0 kernels)."""

    def __init__(self, voxel_cutoff: int = 10) -> None:
        """
        Initialize MSI feature extractor.

        Args:
            voxel_cutoff: Historical constructor argument retained for API
                compatibility. Small-region filtering is not applied by the
                L0 matrix definition (matches the previous live path, which
                left the cutoff unused in the hot loop).
        """
        self.voxel_cutoff = int(voxel_cutoff)

    def calculate_MSI_matrix(
        self, habitat_array: np.ndarray, unique_class: int
    ) -> np.ndarray:
        """
        Calculate the MSI matrix from a habitat label array.

        Args:
            habitat_array: Integer habitat map (0 = background).
            unique_class: Number of classes including background; sets the
                matrix shape to ``(unique_class, unique_class)``.

        Returns:
            Int64 co-occurrence matrix of face-connected neighbour pairs.
        """
        labels = np.asarray(habitat_array)
        if labels.size == 0 or not np.any(labels != 0):
            logger.warning("No non-zero elements found in habitat array")
            return np.zeros((unique_class, unique_class), dtype=np.int64)
        return spatial_interaction_matrix(labels, int(unique_class))

    def calculate_MSI_features(
        self, msi_matrix: np.ndarray, name: str
    ) -> Dict[str, float]:
        """
        Derive MSI features from an interaction matrix.

        Args:
            msi_matrix: Square non-negative MSI matrix.
            name: Subject / dataset tag used only in error messages.

        Returns:
            Feature name → value mapping (v0.1 key scheme).
        """
        try:
            return msi_features_from_matrix(msi_matrix)
        except ValueError as exc:
            raise AssertionError(f"msi_matrix of {name}: {exc}") from exc

    def extract_MSI_features(
        self, habitat_path: str, n_habitats: int, subj: str
    ) -> Dict[str, Union[float, str]]:
        """
        Extract MSI features from a single habitat map on disk.

        Args:
            habitat_path: Path to the habitat map file.
            n_habitats: Number of habitats (background adds +1 class).
            subj: Subject ID (used in error logs / feature naming).

        Returns:
            Feature dict, or ``{"error": ...}`` on failure.
        """
        try:
            img = sitk.ReadImage(habitat_path)
            array = sitk.GetArrayFromImage(img)
            unique_class = int(n_habitats) + 1
            msi_matrix = self.calculate_MSI_matrix(array, unique_class)
            return self.calculate_MSI_features(msi_matrix, subj)
        except Exception as exc:  # noqa: BLE001 — keep CLI batch resilient
            logger.error("Error extracting MSI features for subject %s: %s", subj, exc)
            return {"error": str(exc)}
