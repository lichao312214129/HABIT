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

from typing import List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field

from habit.api.exceptions import GeometryError, HABITAPIError
from habit.contracts.habitat import VoxelFeatureField
from habit.contracts.provenance import Provenance
from habit.contracts.subject import Subject
from habit.domain.voxel_features.registry import VoxelFeatureExtractorRegistry
from habit.spec.specs import Spec

__all__ = ["RawVoxelFeatures", "RawVoxelFeaturesParams"]


class RawVoxelFeaturesParams(BaseModel):
    """Constructor parameters for :class:`RawVoxelFeatures`."""

    modalities: List[str] = Field(min_length=1)
    roi: Optional[str] = None


@VoxelFeatureExtractorRegistry.register("raw")
class RawVoxelFeatures:
    """
    Per-voxel raw intensity of every requested modality inside the ROI.

    This is the simplest possible :class:`VoxelFeatureExtractor` and serves
    as the reference implementation for the protocol: one subject in, one
    :class:`VoxelFeatureField` out, with geometry validated before any
    computation.

    Args:
        modalities: Modality keys to read from the subject, in feature order.
        roi: Mask key defining the region of interest; ``None`` uses the
            subject's single mask.
    """

    def __init__(self, modalities: Sequence[str], roi: Optional[str] = None) -> None:
        if not modalities:
            raise HABITAPIError("RawVoxelFeatures requires at least one modality.")
        self.modalities: Tuple[str, ...] = tuple(modalities)
        self.roi = roi

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="raw",
            params={"modalities": list(self.modalities), "roi": self.roi},
        )

    def __call__(self, subject: Subject) -> VoxelFeatureField:
        """
        Compute per-voxel intensities for one subject.

        Args:
            subject: Subject providing the requested modalities and mask.

        Returns:
            One row per ROI voxel, one column per modality.

        Raises:
            KeyError: If a modality or the ROI is absent on the subject.
            GeometryError: If a modality and the mask do not share a grid.
        """
        mask = subject.mask(self.roi)
        arrays: List[np.ndarray] = []
        for modality in self.modalities:
            image = subject.image(modality)
            if not image.geometry.is_compatible_with(mask.geometry):
                raise GeometryError(
                    f"Subject {subject.subject_id!r}: modality {modality!r} "
                    "and the ROI mask do not share a compatible voxel grid."
                )
            arrays.append(np.asarray(image.data))
        inside = np.asarray(mask.data) > 0
        voxel_index = np.argwhere(inside)
        values = np.stack([array[inside] for array in arrays], axis=1)
        provenance = Provenance.source("subject_images").derive(
            produced_by=f"voxel_feature_extractor.{self.spec.name}",
            spec_fingerprint=self.spec.fingerprint(),
        )
        return VoxelFeatureField(
            subject_id=subject.subject_id,
            feature_names=tuple(self.modalities),
            values=values.astype(np.float64, copy=False),
            voxel_index=voxel_index,
            geometry=mask.geometry,
            provenance=provenance,
        )


VoxelFeatureExtractorRegistry.register_params_model("raw", RawVoxelFeaturesParams)
