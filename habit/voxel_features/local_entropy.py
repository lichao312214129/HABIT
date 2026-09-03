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

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from habit.contracts.habitat import VoxelFeatureField
from habit.contracts.subject import Subject
from habit.voxel_features._base import (
    aligned_image,
    build_voxel_field,
    resolve_source_modalities,
    resolve_voxel_modalities,
    roi_voxels,
)
from habit.voxel_features.registry import VoxelFeatureExtractorRegistry
from habit.kernels.voxel_texture import local_entropy_map
from habit.spec.specs import Spec

__all__ = ["LocalEntropyVoxelFeatures"]


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
        modality: Single modality key -- the explicit form used inside
            feature trees. Mutually exclusive with ``modalities``.
        modalities: Modality keys to describe, in feature order; empty selects
            every image the subject carries.
        as_: Optional output-column alias. Valid only with exactly one
            resolved modality; the column suffix then uses the alias.
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
        modality: Optional[str] = None,
        as_: Optional[str] = None,
    ) -> None:
        if isinstance(kernel_size, bool) or not isinstance(kernel_size, int) or kernel_size < 1:
            raise ValueError(
                f"kernel_size must be a positive integer; got {kernel_size!r}."
            )
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
            raise ValueError(f"bins must be an integer of at least 2; got {bins!r}.")
        resolved, labels = resolve_source_modalities(
            modality, modalities, as_, owner="local_entropy"
        )
        self.modalities = resolved
        self.source_labels = labels
        self.modality = str(modality) if modality is not None else None
        self.as_ = str(as_) if as_ is not None else None
        self.roi = roi
        self.kernel_size = int(kernel_size)
        self.bins = int(bins)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        params: Dict[str, Any] = {
            "modalities": list(self.modalities),
            "roi": self.roi,
            "kernel_size": self.kernel_size,
            "bins": self.bins,
        }
        # Fold the singular/alias forms in only when set so the historical
        # ``modalities=[...]`` fingerprint stays byte-identical.
        if self.modality is not None:
            params["modality"] = self.modality
        if self.as_ is not None:
            params["as_"] = self.as_
        return Spec(name="local_entropy", params=params)

    def __call__(self, subject: Subject) -> VoxelFeatureField:
        """
        Compute per-voxel local entropy for one subject.

        Args:
            subject: Subject providing the requested modalities and mask.

        Returns:
            One row per ROI voxel, one ``local_entropy-{source}`` column per
            modality, where ``source`` is the ``as_`` alias when given, else
            the modality name.

        Raises:
            GeometryError: If a modality and the mask are on different grids.
            HABITAPIError: If a requested modality is absent.
        """
        modalities = resolve_voxel_modalities(
            subject, self.modalities, owner="local_entropy"
        )
        # ``resolve_voxel_modalities`` may expand an empty request to every
        # subject image; labels track that expansion one-to-one.
        labels = (
            self.source_labels
            if len(self.source_labels) == len(modalities)
            else modalities
        )
        mask, inside, voxel_index = roi_voxels(subject, self.roi)

        names: List[str] = []
        columns: List[np.ndarray] = []
        for modality, label in zip(modalities, labels):
            array = aligned_image(subject, modality, mask, owner="local_entropy")
            entropy = local_entropy_map(
                array, kernel_size=self.kernel_size, bins=self.bins
            )
            names.append(f"local_entropy-{label}")
            columns.append(entropy[inside])

        values = np.stack(columns, axis=1)
        return build_voxel_field(
            subject, mask, voxel_index, names, values, self.spec
        )

