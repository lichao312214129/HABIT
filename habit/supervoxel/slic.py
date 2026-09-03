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
"""SLIC supervoxels: the reference Supervoxelizer."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from habit.exceptions import HABITAPIError
from habit.contracts.habitat import Supervoxelization, VoxelFeatureField
from habit.supervoxel.registry import SupervoxelizerRegistry
from habit.supervoxel.features_base import aggregate_voxel_means
from habit.spec.specs import Spec
from habit.utils.estimator_utils import (
    check_passthrough_accepted,
    validate_estimator_params,
)

__all__ = ["SlicSupervoxelizer"]


@SupervoxelizerRegistry.register("slic")
class SlicSupervoxelizer:
    """
    Partition the ROI into SLIC supervoxels and average features within each.

    SLIC (Simple Linear Iterative Clustering) groups spatially coherent,
    feature-similar voxels; the v0.1 pipeline exposed the same algorithm
    through its clustering factory. Here it is one ordinary subject-level
    operator: field in, partition out.

    Implements :class:`~habit._protocols.Seedable` so every
    supervoxelizer shares the same seeding surface as kmeans/gmm and so
    ``HabitatSpec.random_seed`` reaches this stage during assembly. The
    current ``skimage.segmentation.slic`` backend has no RNG parameter;
    ``set_random_state`` therefore records the seed for API uniformity and
    future backends without changing today's deterministic partitions.

    Args:
        n_supervoxels: Requested number of supervoxels. Clamped to the
            number of ROI voxels (a partition cannot have more non-empty
            regions than voxels).
        compactness: Balance between colour similarity and spatial proximity
            (``skimage.segmentation.slic`` semantics).
        enforce_connectivity: When ``True``, disconnected segments are
            relabelled so every supervoxel is connected.
        estimator_params: Extra keyword arguments forwarded verbatim to
            ``skimage.segmentation.slic`` (e.g. ``{"sigma": 1.0}``), for
            vendor parameters HABIT does not declare. Keys colliding with a
            declared parameter or with a call argument HABIT controls
            (``n_segments``, ``mask``, ``channel_axis``, ``start_label``) are
            rejected, and every key is validated against the vendor signature
            at call time: a key recorded in the spec fingerprint must reach
            the vendor function, never be silently dropped.
    """

    def __init__(
        self,
        n_supervoxels: int = 100,
        compactness: float = 10.0,
        enforce_connectivity: bool = True,
        estimator_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if isinstance(n_supervoxels, bool) or not isinstance(n_supervoxels, int) or n_supervoxels < 1:
            raise HABITAPIError(
                f"n_supervoxels must be positive; got {n_supervoxels}."
            )
        if isinstance(compactness, bool) or not isinstance(compactness, (int, float)) or compactness <= 0:
            raise HABITAPIError(f"compactness must be positive; got {compactness!r}.")
        self.n_supervoxels = int(n_supervoxels)
        self.compactness = float(compactness)
        self.enforce_connectivity = bool(enforce_connectivity)
        self.estimator_params: Dict[str, Any] = validate_estimator_params(
            estimator_params,
            declared=("n_supervoxels", "compactness", "enforce_connectivity"),
            fixed=("n_segments", "mask", "channel_axis", "start_label"),
            owner="supervoxelizer.slic",
        )
        # Default matches other Seedable supervoxelizers (fixed seed 0).
        self._seed = 0

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        params: Dict[str, Any] = {
            "n_supervoxels": self.n_supervoxels,
            "compactness": self.compactness,
            "enforce_connectivity": self.enforce_connectivity,
        }
        # Fold the passthrough in only when non-empty so the default
        # configuration keeps its historical fingerprint.
        if self.estimator_params:
            params["estimator_params"] = dict(self.estimator_params)
        return Spec(name="slic", params=params)

    def set_random_state(self, seed: int) -> None:
        """
        Record the study seed for this supervoxelizer.

        Args:
            seed: Non-negative study seed from ``HabitatSpec.random_seed`` or
                an explicit caller. Stored for Seedable uniformity; the
                current skimage SLIC call does not consume it.
        """
        self._seed = int(seed)

    def __call__(self, field: VoxelFeatureField) -> Supervoxelization:
        """
        Group voxels into supervoxels and aggregate their features.

        Args:
            field: Per-voxel features for one subject.

        Returns:
            The supervoxel partition (``0`` = outside ROI, ``1..K`` =
            supervoxels) together with per-supervoxel mean features. Pass a
            :class:`~habit._protocols.SupervoxelFeatureExtractor` to
            the pipeline to describe the same regions differently.
        """
        # scikit-image is an OPTIONAL dependency (habitat-analysis[slic]): the
        # kernel contract is the Supervoxelizer PROTOCOL, and the default
        # backends (kmeans / gmm feature clustering) need nothing beyond the
        # required set. Only this SLIC implementation pulls scikit-image, and
        # it does so here rather than at module scope so the registry entry
        # stays importable (and discoverable via ``habit list``) without the
        # extra installed.
        from habit.utils.optional_deps import require

        slic = require(
            "skimage.segmentation",
            extra="slic",
            purpose="SLIC supervoxel segmentation (supervoxelizer 'slic')",
        ).slic

        check_passthrough_accepted(
            slic, self.estimator_params, owner="supervoxelizer.slic"
        )

        shape = tuple(int(v) for v in field.geometry.shape)
        n_voxels, n_features = field.values.shape
        dense = np.zeros((*shape, n_features), dtype=np.float64)
        dense[tuple(field.voxel_index.T)] = field.values
        mask = np.zeros(shape, dtype=bool)
        mask[tuple(field.voxel_index.T)] = True

        n_segments = max(1, min(self.n_supervoxels, n_voxels))
        labels = slic(
            dense,
            n_segments=n_segments,
            compactness=self.compactness,
            mask=mask,
            channel_axis=-1,
            start_label=1,
            enforce_connectivity=self.enforce_connectivity,
            **self.estimator_params,
        )
        labels = np.where(mask, labels, 0).astype(np.int32)

        features = aggregate_voxel_means(field, labels)

        provenance = field.provenance.derive(
            produced_by=f"supervoxelizer.{self.spec.name}",
            spec_fingerprint=self.spec.fingerprint(),
        )
        return Supervoxelization(
            subject_id=field.subject_id,
            label_array=labels,
            features=features,
            geometry=field.geometry,
            provenance=provenance,
        )

