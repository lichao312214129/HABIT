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
"""Raw-intensity voxel features: the reference VoxelFeatureExtractor."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from habit.exceptions import HABITAPIError
from habit.contracts.habitat import VoxelFeatureField
from habit.contracts.subject import Subject
from habit.voxel_features._base import (
    aligned_image,
    build_voxel_field,
    resolve_source_modalities,
    roi_voxels,
)
from habit.voxel_features.registry import VoxelFeatureExtractorRegistry
from habit.spec.specs import Spec

__all__ = ["RawVoxelFeatures"]


@VoxelFeatureExtractorRegistry.register("raw")
class RawVoxelFeatures:
    """
    Per-voxel raw intensity of every requested modality inside the ROI.

    This is the simplest possible :class:`VoxelFeatureExtractor` and serves
    as the reference implementation for the protocol: one subject in, one
    :class:`VoxelFeatureField` out, with geometry validated before any
    computation.

    Args:
        modality: Single modality key -- the explicit form used inside
            feature trees (``raw("T1")``). Mutually exclusive with
            ``modalities``.
        modalities: Modality keys to read from the subject, in feature
            order -- the historical convenience that stacks several
            modalities into one node without a ``concat`` combiner.
        as_: Optional output-column alias. Valid only with exactly one
            resolved modality; the column is then named after the alias
            instead of the modality.
        roi: Mask key defining the region of interest; ``None`` uses the
            subject's single mask.
    """

    def __init__(
        self,
        modalities: Sequence[str] = (),
        roi: Optional[str] = None,
        modality: Optional[str] = None,
        as_: Optional[str] = None,
    ) -> None:
        resolved, labels = resolve_source_modalities(
            modality, modalities, as_, owner="raw"
        )
        if not resolved:
            raise HABITAPIError("RawVoxelFeatures requires at least one modality.")
        self.modalities: Tuple[str, ...] = resolved
        self.source_labels: Tuple[str, ...] = labels
        self.modality = str(modality) if modality is not None else None
        self.as_ = str(as_) if as_ is not None else None
        self.roi = roi

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        params: Dict[str, Any] = {
            "modalities": list(self.modalities),
            "roi": self.roi,
        }
        # Fold the singular/alias forms in only when set so the historical
        # ``modalities=[...]`` fingerprint stays byte-identical.
        if self.modality is not None:
            params["modality"] = self.modality
        if self.as_ is not None:
            params["as_"] = self.as_
        return Spec(name="raw", params=params)

    def __call__(self, subject: Subject) -> VoxelFeatureField:
        """
        Compute per-voxel intensities for one subject.

        Args:
            subject: Subject providing the requested modalities and mask.

        Returns:
            One row per ROI voxel, one column per modality (named after the
            source label: the ``as_`` alias when given, else the modality).

        Raises:
            KeyError: If a modality or the ROI is absent on the subject.
            GeometryError: If a modality and the mask do not share a grid.
        """
        mask, inside, voxel_index = roi_voxels(subject, self.roi)
        arrays: List[np.ndarray] = [
            aligned_image(subject, modality, mask, owner="raw")
            for modality in self.modalities
        ]
        values = np.stack([array[inside] for array in arrays], axis=1)
        return build_voxel_field(
            subject, mask, voxel_index, self.source_labels, values, self.spec
        )

