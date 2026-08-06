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
"""Local-entropy voxel features: neighbourhood heterogeneity per voxel."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from habit.contracts.habitat import VoxelFeatureField
from habit.contracts.subject import Subject
from habit.domain.voxel_features._base import (
    aligned_image,
    build_voxel_field,
    resolve_voxel_modalities,
    roi_voxels,
)
from habit.domain.voxel_features.registry import VoxelFeatureExtractorRegistry
from habit.kernels.voxel_texture import local_entropy_map
from habit.spec.specs import Spec

__all__ = ["LocalEntropyVoxelFeatures", "LocalEntropyVoxelFeaturesParams"]


class LocalEntropyVoxelFeaturesParams(BaseModel):
    """Constructor parameters for :class:`LocalEntropyVoxelFeatures`."""

    model_config = ConfigDict(extra="forbid")
    modalities: Sequence[str] = ()
    roi: Optional[str] = None
    kernel_size: int = Field(default=3, gt=0)
    bins: int = Field(default=32, gt=1)


@VoxelFeatureExtractorRegistry.register("local_entropy")
class LocalEntropyVoxelFeatures:
    """
    Shannon entropy of each voxel's intensity neighbourhood.

    Where raw intensity says how bright a voxel is, local entropy says how
    disordered its surroundings are, which separates homogeneous tissue from
    structurally mixed tissue at the same intensity. The entropy map is
    computed over the whole image and then restricted to the ROI, exactly as
    in v0.1, so voxels at the ROI border still see their true neighbourhood.

    Args:
        modalities: Modality keys to describe, in feature order; empty selects
            every image the subject carries.
        roi: Mask key defining the region of interest; ``None`` uses the
            subject's single mask.
        kernel_size: Neighbourhood edge length in voxels. Even values are
            incremented to keep the neighbourhood centred, as in v0.1.
        bins: Histogram bins used to discretise intensities.
    """

    def __init__(
        self,
        modalities: Sequence[str] = (),
        roi: Optional[str] = None,
        kernel_size: int = 3,
        bins: int = 32,
    ) -> None:
        self.modalities = tuple(modalities)
        self.roi = roi
        self.kernel_size = int(kernel_size)
        self.bins = int(bins)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        return Spec(
            name="local_entropy",
            params={
                "modalities": list(self.modalities),
                "roi": self.roi,
                "kernel_size": self.kernel_size,
                "bins": self.bins,
            },
        )

    def __call__(self, subject: Subject) -> VoxelFeatureField:
        """
        Compute per-voxel local entropy for one subject.

        Args:
            subject: Subject providing the requested modalities and mask.

        Returns:
            One row per ROI voxel, one ``local_entropy-{modality}`` column per
            modality.

        Raises:
            GeometryError: If a modality and the mask are on different grids.
            HABITAPIError: If a requested modality is absent.
        """
        modalities = resolve_voxel_modalities(
            subject, self.modalities, owner="local_entropy"
        )
        mask, inside, voxel_index = roi_voxels(subject, self.roi)

        names: List[str] = []
        columns: List[np.ndarray] = []
        for modality in modalities:
            array = aligned_image(subject, modality, mask, owner="local_entropy")
            entropy = local_entropy_map(
                array, kernel_size=self.kernel_size, bins=self.bins
            )
            names.append(f"local_entropy-{modality}")
            columns.append(entropy[inside])

        values = np.stack(columns, axis=1)
        return build_voxel_field(
            subject, mask, voxel_index, names, values, self.spec
        )


VoxelFeatureExtractorRegistry.register_params_model(
    "local_entropy", LocalEntropyVoxelFeaturesParams
)
