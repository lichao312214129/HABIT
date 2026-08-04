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
"""Feature-space supervoxelizers: k-means and GMM.

Unlike SLIC, these group voxels by feature similarity ALONE, with no spatial
term, so a supervoxel may be spatially disconnected. That is the v0.1
``habitat_segmentation.supervoxel.algorithm: kmeans|gmm`` behaviour and it is
deliberate: within-subject clustering is used there as a denoising and
data-reduction step ahead of the population clustering, not as a
segmentation. Choose ``slic`` when spatial coherence matters.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from habit.exceptions import HABITAPIError
from habit.contracts.habitat import Supervoxelization, VoxelFeatureField
from habit.domain.supervoxel._base import partition_from_voxel_labels
from habit.domain.supervoxel.registry import SupervoxelizerRegistry
from habit.spec.specs import Spec

__all__ = [
    "KMeansSupervoxelizer",
    "KMeansSupervoxelizerParams",
    "GmmSupervoxelizer",
    "GmmSupervoxelizerParams",
]


def _effective_clusters(requested: int, n_voxels: int) -> int:
    """
    Clamp the requested cluster count to what the ROI can support.

    Args:
        requested: Configured number of supervoxels.
        n_voxels: ROI voxel count of this subject.

    Returns:
        The count actually used.

    Raises:
        HABITAPIError: If the ROI is empty.
    """
    if n_voxels < 1:
        raise HABITAPIError(
            "Cannot build supervoxels: the ROI contains no voxel."
        )
    return max(1, min(int(requested), int(n_voxels)))


class KMeansSupervoxelizerParams(BaseModel):
    """Constructor parameters for :class:`KMeansSupervoxelizer`."""

    n_supervoxels: int = Field(default=50, gt=0)
    max_iter: int = Field(default=300, gt=0)
    n_init: int = Field(default=10, gt=0)


@SupervoxelizerRegistry.register("kmeans")
class KMeansSupervoxelizer:
    """
    Partition the ROI by k-means over voxel features.

    The v0.1 default supervoxel algorithm for the two-step design.

    Args:
        n_supervoxels: Requested number of supervoxels, clamped to the ROI
            voxel count.
        max_iter: Maximum k-means iterations per restart.
        n_init: Number of k-means restarts.
    """

    def __init__(
        self,
        n_supervoxels: int = 50,
        max_iter: int = 300,
        n_init: int = 10,
    ) -> None:
        if n_supervoxels < 1:
            raise HABITAPIError(
                f"n_supervoxels must be positive; got {n_supervoxels}."
            )
        self.n_supervoxels = int(n_supervoxels)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self._seed = 0

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name="kmeans",
            params={
                "n_supervoxels": self.n_supervoxels,
                "max_iter": self.max_iter,
                "n_init": self.n_init,
            },
        )

    def set_random_state(self, seed: int) -> None:
        """Set the seed applied to k-means initialisation."""
        self._seed = int(seed)

    def __call__(self, field: VoxelFeatureField) -> Supervoxelization:
        """
        Cluster the subject's voxels into supervoxels.

        Args:
            field: Per-voxel features for one subject.

        Returns:
            The supervoxel partition summarised by feature means.
        """
        from sklearn.cluster import KMeans

        n_clusters = _effective_clusters(self.n_supervoxels, field.values.shape[0])
        model = KMeans(
            n_clusters=n_clusters,
            random_state=self._seed,
            max_iter=self.max_iter,
            n_init=self.n_init,
        )
        # Labels are 1-based: 0 is reserved for "outside the ROI".
        voxel_labels = model.fit_predict(field.values) + 1
        return partition_from_voxel_labels(field, voxel_labels, self.spec)


class GmmSupervoxelizerParams(BaseModel):
    """Constructor parameters for :class:`GmmSupervoxelizer`."""

    n_supervoxels: int = Field(default=50, gt=0)
    max_iter: int = Field(default=300, gt=0)
    n_init: int = Field(default=10, gt=0)
    covariance_type: str = "full"


@SupervoxelizerRegistry.register("gmm")
class GmmSupervoxelizer:
    """
    Partition the ROI by a Gaussian mixture over voxel features.

    Soft-assignment counterpart of :class:`KMeansSupervoxelizer`; each voxel
    takes the component of highest posterior probability.

    Args:
        n_supervoxels: Requested number of supervoxels, clamped to the ROI
            voxel count.
        max_iter: Maximum EM iterations.
        n_init: Number of EM restarts.
        covariance_type: scikit-learn covariance parameterisation
            (``"full"``, ``"tied"``, ``"diag"``, ``"spherical"``).
    """

    def __init__(
        self,
        n_supervoxels: int = 50,
        max_iter: int = 300,
        n_init: int = 10,
        covariance_type: str = "full",
    ) -> None:
        if n_supervoxels < 1:
            raise HABITAPIError(
                f"n_supervoxels must be positive; got {n_supervoxels}."
            )
        self.n_supervoxels = int(n_supervoxels)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.covariance_type = str(covariance_type)
        self._seed = 0

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name="gmm",
            params={
                "n_supervoxels": self.n_supervoxels,
                "max_iter": self.max_iter,
                "n_init": self.n_init,
                "covariance_type": self.covariance_type,
            },
        )

    def set_random_state(self, seed: int) -> None:
        """Set the seed applied to mixture initialisation."""
        self._seed = int(seed)

    def __call__(self, field: VoxelFeatureField) -> Supervoxelization:
        """
        Cluster the subject's voxels into supervoxels.

        Args:
            field: Per-voxel features for one subject.

        Returns:
            The supervoxel partition summarised by feature means.
        """
        from sklearn.mixture import GaussianMixture

        n_components = _effective_clusters(self.n_supervoxels, field.values.shape[0])
        model = GaussianMixture(
            n_components=n_components,
            random_state=self._seed,
            max_iter=self.max_iter,
            n_init=self.n_init,
            covariance_type=self.covariance_type,
        )
        voxel_labels = np.asarray(model.fit_predict(field.values)) + 1
        return partition_from_voxel_labels(field, voxel_labels, self.spec)


SupervoxelizerRegistry.register_params_model("kmeans", KMeansSupervoxelizerParams)
SupervoxelizerRegistry.register_params_model("gmm", GmmSupervoxelizerParams)
