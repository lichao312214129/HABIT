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
"""Consensus-clustering habitat model fitter.

The numerical consensus step is not Apache-2.0. ``fit`` calls
:class:`habit.third_party.inmoose.consensus_clustering.consensusClustering`,
which is GPL-3.0-or-later (Sajovic 2019-2021; Weill and Appé 2023, via
InMoose). Importing or calling this fitter uses that GPL component. The GPL
text ships at ``habit/third_party/inmoose/LICENSE``. HABIT's Apache-2.0
terms do not relicense it. See the repository ``NOTICE``.

What this fitter returns is still a HABIT :class:`~habit.contracts.habitat.HabitatModel`:
consensus labels exist only for the items that were clustered, so each
consensus group is summarised by the mean of its feature rows. Those means
are the centroids. Unseen subjects are labelled by nearest centroid, the
same assignment path as k-means and the Gaussian mixture.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Type

import numpy as np

from habit.contracts.habitat import HabitatModel, Supervoxelization
from habit.contracts.subject import Cohort
from habit.exceptions import HABITAPIError
from habit.habitat_model._base import build_habitat_model, pool_supervoxel_features
from habit.habitat_model.registry import HabitatModelFitterRegistry
from habit.spec.specs import Spec

__all__ = ["ConsensusHabitatModelFitter"]

#: Above this item count the ``n x n`` consensus matrices are not copied onto
#: the fitted model. The run itself still refuses to start above ``max_items``.
_MATRIX_STORE_LIMIT = 800

_IMPLEMENTATION = (
    "inmoose.consensus_clustering.consensusClustering "
    "(GPL-3.0-or-later; Sajovic / Weill / Appe). "
    "See habit/third_party/inmoose/LICENSE."
)


def _kmeans_cluster_class(random_state: int, n_init: int, max_iter: int) -> Type:
    """
    Build a cluster class the vendored algorithm can construct with ``n_clusters``.

    InMoose calls ``cluster(n_clusters=k)`` and then ``fit_predict``. Extra
    constructor arguments are not forwarded, so the seed and the k-means
    restart count are closed over as class attributes.

    Args:
        random_state: Seed passed to scikit-learn ``KMeans``.
        n_init: Independent k-means restarts inside each resample.
        max_iter: Iteration cap for each k-means restart.

    Returns:
        A class with ``__init__(n_clusters)`` and ``fit_predict(data)``.
    """

    class KMeansInner:
        """K-means inner clusterer for one consensus resample or one final cut."""

        _random_state = int(random_state)
        _n_init = int(n_init)
        _max_iter = int(max_iter)

        def __init__(self, n_clusters: int) -> None:
            self.n_clusters = int(n_clusters)

        def fit_predict(self, data: np.ndarray) -> np.ndarray:
            """
            Cluster rows of ``data``.

            Args:
                data: Array of shape ``(n_samples, n_features)``.

            Returns:
                Integer labels of shape ``(n_samples,)``, starting at 0.
            """
            from sklearn.cluster import KMeans

            model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self._random_state,
                n_init=self._n_init,
                max_iter=self._max_iter,
            )
            return np.asarray(model.fit_predict(np.asarray(data)), dtype=int)

    return KMeansInner


def _agglomerative_cluster_class() -> Type:
    """
    Build the scikit-learn agglomerative class used by the InMoose example.

    ``AgglomerativeClustering(n_clusters=k)`` is ward linkage on Euclidean
    distance. The same class is applied to feature rows during resampling
    and to rows of ``1 - consensus_matrix`` for the final cut.

    Returns:
        A class with ``__init__(n_clusters)`` and ``fit_predict(data)``.
    """

    class AgglomerativeInner:
        """Ward agglomerative inner clusterer."""

        def __init__(self, n_clusters: int) -> None:
            self.n_clusters = int(n_clusters)

        def fit_predict(self, data: np.ndarray) -> np.ndarray:
            """
            Cluster rows of ``data``.

            Args:
                data: Array of shape ``(n_samples, n_features)``.

            Returns:
                Integer labels of shape ``(n_samples,)``, starting at 0.
            """
            from sklearn.cluster import AgglomerativeClustering

            model = AgglomerativeClustering(n_clusters=self.n_clusters)
            return np.asarray(model.fit_predict(np.asarray(data)), dtype=int)

    return AgglomerativeInner


@HabitatModelFitterRegistry.register("consensus")
class ConsensusHabitatModelFitter:
    """
    Learn population habitats by Monti consensus clustering of pooled items.

    Items are the rows of the pooled supervoxel feature matrix (one row per
    supervoxel). The vendored InMoose / Sajovic algorithm draws
    ``n_resamples`` subsets of those rows (fraction ``resample_proportion``,
    default ``0.5``, items only — features are not resampled), clusters each
    subset, and records how often each pair co-clusters. The habitat count
    is their ``bestK`` (relative change in area under the consensus-index
    CDF). Their delta curve is indexed on ``k = min_habitats .. max_habitats - 1``;
    ``max_habitats`` itself is never chosen by that rule.

    The consensus cut labels only the fitted items. Each occupied group is
    replaced by its mean feature vector so the result is a centroid model
    that :class:`~habit.habitat_model.assignment.nearest_centroid.NearestCentroidAssigner`
    can apply to a new subject.

    The consensus step is GPL-3.0-or-later. See the module docstring.

    Args:
        n_habitats: Fixed cluster count. ``None`` selects ``bestK`` between
            ``min_habitats`` and ``max_habitats``.
        min_habitats: Smallest candidate count. Must be at least 2, matching
            the vendored implementation.
        max_habitats: Largest candidate count. Not selectable by ``bestK``;
            see the class docstring.
        n_resamples: Resamples per candidate count. The vendored default is
            50.
        resample_proportion: Fraction of items drawn without replacement on
            each resample. The vendored default is 0.5.
        inner: ``"kmeans"`` (default) or ``"agglomerative"`` (scikit-learn
            ward). This is the class passed into the vendored algorithm for
            both the resamples and the final cut of ``1 - consensus_matrix``.
        n_init: K-means restarts when ``inner`` is ``"kmeans"``. Ignored by
            agglomerative clustering.
        max_iter: K-means iteration cap when ``inner`` is ``"kmeans"``.
        max_items: Refuse to run when the pooled matrix has more rows than
            this. Consensus matrices are ``n x n`` per candidate ``k``.
    """

    def __init__(
        self,
        n_habitats: Optional[int] = None,
        min_habitats: int = 2,
        max_habitats: int = 10,
        n_resamples: int = 50,
        resample_proportion: float = 0.5,
        inner: str = "kmeans",
        n_init: int = 10,
        max_iter: int = 300,
        max_items: int = 2000,
    ) -> None:
        inner_name = str(inner).strip().lower()
        if inner_name not in {"kmeans", "agglomerative"}:
            raise HABITAPIError(
                "consensus inner must be 'kmeans' or 'agglomerative', "
                f"got {inner!r}."
            )
        self.n_habitats = None if n_habitats is None else int(n_habitats)
        self.min_habitats = int(min_habitats)
        self.max_habitats = int(max_habitats)
        self.n_resamples = int(n_resamples)
        self.resample_proportion = float(resample_proportion)
        self.inner = inner_name
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.max_items = int(max_items)
        self._seed = 0
        self._validate_settings()

    def _validate_settings(self) -> None:
        """Reject settings the vendored algorithm cannot execute."""
        if self.min_habitats < 2:
            raise HABITAPIError(
                "consensus min_habitats must be at least 2 "
                f"(got {self.min_habitats})."
            )
        if self.n_habitats is not None and self.n_habitats < 2:
            raise HABITAPIError(
                "consensus n_habitats must be at least 2 "
                f"(got {self.n_habitats})."
            )
        if self.n_habitats is None and self.max_habitats <= self.min_habitats:
            raise HABITAPIError(
                "max_habitats must be greater than min_habitats when "
                "n_habitats is selected automatically."
            )
        if self.n_resamples < 1:
            raise HABITAPIError(
                f"consensus n_resamples must be at least 1 (got {self.n_resamples})."
            )
        if not 0.0 < self.resample_proportion <= 1.0:
            raise HABITAPIError(
                "consensus resample_proportion must be in (0, 1], "
                f"got {self.resample_proportion}."
            )
        if self.max_items < 2:
            raise HABITAPIError(
                f"consensus max_items must be at least 2 (got {self.max_items})."
            )

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification, including the GPL inner step."""
        return Spec(
            name="consensus",
            params={
                "n_habitats": self.n_habitats,
                "min_habitats": self.min_habitats,
                "max_habitats": self.max_habitats,
                "n_resamples": self.n_resamples,
                "resample_proportion": self.resample_proportion,
                "inner": self.inner,
                "n_init": self.n_init,
                "max_iter": self.max_iter,
                "max_items": self.max_items,
            },
        )

    def set_random_state(self, seed: int) -> None:
        """Set the seed for item resampling and, when used, k-means restarts."""
        self._seed = int(seed)

    def _inner_class(self) -> Type:
        """Return the inner cluster class closed over this fitter's seed."""
        if self.inner == "agglomerative":
            return _agglomerative_cluster_class()
        return _kmeans_cluster_class(self._seed, self.n_init, self.max_iter)

    def _candidate_bounds(self, n_items: int) -> tuple:
        """
        Return ``(mink, maxk)`` actually passed to the vendored algorithm.

        A fixed ``n_habitats`` collapses the search to that single count.
        Automatic search clamps ``max_habitats`` to ``n_items - 1``.

        Args:
            n_items: Number of pooled feature rows.

        Returns:
            Inclusive ``(mink, maxk)`` pair.
        """
        if self.n_habitats is not None:
            if n_items <= self.n_habitats:
                raise HABITAPIError(
                    f"Cannot fit {self.n_habitats} habitats on {n_items} "
                    "pooled items."
                )
            return int(self.n_habitats), int(self.n_habitats)
        upper = min(self.max_habitats, n_items - 1)
        if upper < self.min_habitats:
            raise HABITAPIError(
                f"Cannot search habitats in [{self.min_habitats}, "
                f"{self.max_habitats}] with only {n_items} items."
            )
        return self.min_habitats, upper

    def fit(
        self,
        units: Sequence[Supervoxelization],
        *,
        cohort: Optional[Cohort] = None,
    ) -> HabitatModel:
        """
        Learn shared habitat centroids from consensus groups of all subjects.

        Args:
            units: Supervoxelizations in a defined, reproducible order.
            cohort: Cohort the units came from, fingerprinted into the model.

        Returns:
            A centroid habitat model. ``preprocessing_state["consensus"]``
            holds the CDF areas, delta curve, per-cluster consensus, the
            training-item labels, and (when the item count is small) the
            consensus matrices.

        Raises:
            HABITAPIError: If there are too many items for an ``n x n``
                matrix, or a resample would be smaller than the largest
                candidate ``k``.
        """
        from habit.third_party.inmoose.consensus_clustering import (
            consensusClustering,
        )

        matrix, feature_names = pool_supervoxel_features(units)
        n_items = int(matrix.shape[0])
        if n_items > self.max_items:
            raise HABITAPIError(
                f"Consensus clustering builds an {n_items} x {n_items} matrix "
                f"per candidate k, above max_items={self.max_items}. It is "
                "meant for supervoxels or subjects, not a voxel-level matrix. "
                "Lower the item count, or raise max_items if you accept the "
                "memory cost."
            )
        mink, maxk = self._candidate_bounds(n_items)
        resample_n = int(n_items * self.resample_proportion)
        if resample_n < maxk:
            raise HABITAPIError(
                f"Each resample keeps {resample_n} of {n_items} items, fewer "
                f"than the largest candidate k={maxk}. Raise "
                "resample_proportion or lower max_habitats / n_habitats."
            )

        # GPL-3.0-or-later implementation. Do not reimplement this call.
        cc = consensusClustering(
            self._inner_class(),
            mink=mink,
            maxk=maxk,
            nb_resampling_iteration=self.n_resamples,
            resample_proportion=self.resample_proportion,
        )
        cc.compute_consensus_clustering(matrix, random_state=self._seed, verbose=False)
        selected = int(self.n_habitats) if self.n_habitats is not None else int(cc.bestK)
        labels = np.asarray(cc.predict(selected), dtype=int)
        present = [i for i in range(selected) if np.any(labels == i)]
        if not present:
            raise HABITAPIError(
                f"Consensus clustering returned no occupied cluster at k={selected}."
            )
        centroids = np.vstack(
            [matrix[labels == i].mean(axis=0) for i in present]
        ).astype(np.float64)

        report = _consensus_report(
            cc,
            mink=mink,
            maxk=maxk,
            selected=selected,
            labels=labels,
            present=present,
            n_items=n_items,
            inner=self.inner,
            n_resamples=self.n_resamples,
            resample_proportion=self.resample_proportion,
        )
        return build_habitat_model(
            fitter_name="consensus",
            spec=self.spec,
            centroids=centroids,
            feature_names=feature_names,
            units=units,
            cohort=cohort,
            random_seed=self._seed,
            preprocessing_state={"consensus": report},
        )


def _consensus_report(
    cc: Any,
    *,
    mink: int,
    maxk: int,
    selected: int,
    labels: np.ndarray,
    present: Sequence[int],
    n_items: int,
    inner: str,
    n_resamples: int,
    resample_proportion: float,
) -> Dict[str, object]:
    """
    Copy the vendored numeric outputs into a JSON-friendly report.

    Args:
        cc: Fitted ``consensusClustering`` instance.
        mink: Smallest ``k`` that was evaluated.
        maxk: Largest ``k`` that was evaluated.
        selected: ``k`` used for the final cut.
        labels: Final consensus labels for the training items, 0-based.
        present: Consensus label ids that received at least one item, in
            the same order as the centroid rows.
        n_items: Number of training items.
        inner: Inner clusterer name.
        n_resamples: Resamples per candidate ``k``.
        resample_proportion: Item fraction per resample.

    Returns:
        A dict safe to store on ``HabitatModel.preprocessing_state``.
    """
    candidates = list(range(int(mink), int(maxk) + 1))
    areas = np.asarray(getattr(cc, "Ak"), dtype=np.float64).ravel()
    delta = np.asarray(getattr(cc, "deltaK"), dtype=np.float64).ravel()
    # InMoose plots delta against range(min_k, max_k), which has the same
    # length as deltaK and stops one short of max_k.
    delta_x = list(range(int(mink), int(mink) + int(delta.size)))
    cluster_consensus: Dict[str, List[float]] = {}
    for k in candidates:
        prediction = np.asarray(cc.predict(k), dtype=int)
        scores = np.asarray(
            cc.compute_clusters_consensus(prediction, k),
            dtype=np.float64,
        )
        cluster_consensus[str(k)] = [float(v) for v in scores.tolist()]
    item_prediction, _, item_consensus = cc.compute_summary_statistics(selected)
    report: Dict[str, object] = {
        "implementation": _IMPLEMENTATION,
        "license": "GPL-3.0-or-later",
        "inner": inner,
        "n_resamples": int(n_resamples),
        "resample_proportion": float(resample_proportion),
        "n_items": int(n_items),
        "candidates": candidates,
        "areas": areas,
        "delta_k": delta,
        "delta_k_x": delta_x,
        "selected_k": int(selected),
        "consensus_label_ids": [int(i) for i in present],
        "training_labels": np.asarray(labels, dtype=int),
        "cluster_consensus": cluster_consensus,
        "item_consensus_k": int(selected),
        "item_consensus": np.asarray(item_consensus, dtype=np.float64),
        "item_labels": np.asarray(item_prediction, dtype=int),
    }
    matrices = getattr(cc, "consensus_matrices")
    if matrices is not None and n_items <= _MATRIX_STORE_LIMIT:
        report["matrices"] = np.asarray(matrices, dtype=np.float64)
    return report
