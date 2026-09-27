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
"""Rebuild a fitted habitat estimator and call its own predict.

K-means and a Gaussian mixture keep enough parameters on the
``HabitatModel`` to reconstruct the scikit-learn object and call
``predict`` / ``predict_proba``. Algorithms without an out-of-sample
``predict`` (agglomerative clustering) store class centres and a distance
name; new rows are labelled by the nearest centre under that distance.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from habit.contracts.habitat import HabitatModel
from habit.exceptions import HABITAPIError

__all__ = [
    "CENTROID_METRICS",
    "assignment_rule",
    "assign_to_centers",
    "class_centers",
    "predict_labels",
    "predict_proba_frame",
]

#: Distances supported for class-centre assignment. Each one uses the centre
#: that actually represents the class under that distance.
CENTROID_METRICS: Tuple[str, ...] = ("euclidean", "manhattan", "cosine")


def fitter_name(model: HabitatModel) -> str:
    """
    Return the fitter name recorded on a habitat model.

    Args:
        model: Fitted habitat definition.

    Returns:
        Registered fitter name, or the prefix of ``model_id`` when the
        model card has no fitter block.
    """
    payload = model.spec_payload.get("habitat_model_fitter")
    if isinstance(payload, dict) and payload.get("name"):
        return str(payload["name"])
    return model.model_id.split("-", 1)[0]


def assignment_rule(model: HabitatModel) -> str:
    """
    Return the decision rule this model must use at assignment time.

    Args:
        model: Fitted habitat definition.

    Returns:
        ``"kmeans"``, ``"gmm"``, or ``"nearest_centroid"``.

    Raises:
        HABITAPIError: If the model is a GMM but the mixture parameters
            were never stored (a version-1 archive).
    """
    rule = model.estimator_state.get("rule")
    if rule:
        return str(rule)
    if fitter_name(model) == "gmm":
        raise HABITAPIError(
            f"Habitat model {model.model_id!r} was fit by a Gaussian mixture "
            "but does not store mixture weights and covariances, so "
            "sklearn.mixture.GaussianMixture.predict cannot be rebuilt. "
            "Refit the study with the current HABIT and save the new model."
        )
    # Version-1 k-means and consensus archives stored only centroids.
    # KMeans.predict on those centres is the Euclidean assignment they used.
    return "kmeans"


def class_centers(
    matrix: np.ndarray,
    labels: np.ndarray,
    metric: str,
) -> np.ndarray:
    """
    Summarise each class by the centre that matches ``metric``.

    Args:
        matrix: Training feature rows, shape ``(n_samples, n_features)``.
        labels: Integer class id per row, expected to be ``0 .. k-1`` with
            every id occupied.
        metric: ``"euclidean"`` (mean), ``"manhattan"`` (per-feature
            median), or ``"cosine"`` (mean of L2-normalised rows).

    Returns:
        Array of shape ``(k, n_features)`` in class-id order.

    Raises:
        HABITAPIError: On an unknown metric, an empty class, or a zero
            vector under cosine.
    """
    _check_metric(metric)
    values = np.asarray(matrix, dtype=np.float64)
    ids = np.asarray(labels, dtype=np.int64)
    if ids.size == 0:
        raise HABITAPIError("class_centers requires at least one labelled row.")
    k = int(ids.max()) + 1
    present = set(int(v) for v in np.unique(ids))
    if present != set(range(k)):
        missing = sorted(set(range(k)) - present)
        raise HABITAPIError(
            f"class_centers: labels are missing classes {missing}; every "
            "class needs a centre."
        )
    centers = np.empty((k, values.shape[1]), dtype=np.float64)
    for class_id in range(k):
        block = values[ids == class_id]
        if metric == "euclidean":
            centers[class_id] = block.mean(axis=0)
        elif metric == "manhattan":
            centers[class_id] = np.median(block, axis=0)
        else:
            centers[class_id] = _mean_direction(block)
    return centers


def assign_to_centers(
    matrix: np.ndarray,
    centers: np.ndarray,
    metric: str,
) -> np.ndarray:
    """
    Label each row by the nearest class centre.

    Args:
        matrix: Feature rows, shape ``(n_samples, n_features)``.
        centers: Class centres, shape ``(k, n_features)``, row ``i`` is
            habitat ``i``.
        metric: One of :data:`CENTROID_METRICS`. Cosine distance is
            scikit-learn's ``pairwise_distances(..., metric="cosine")``
            against centres that are means of L2-normalised training rows.

    Returns:
        Integer labels of shape ``(n_samples,)``, starting at 0.

    Raises:
        HABITAPIError: On an unknown metric or a zero vector under cosine.
    """
    from sklearn.metrics import pairwise_distances

    _check_metric(metric)
    values = np.asarray(matrix, dtype=np.float64)
    prototypes = np.asarray(centers, dtype=np.float64)
    if metric == "cosine":
        _reject_zero_rows(values, what="feature row")
        _reject_zero_rows(prototypes, what="class centre")
    distances = pairwise_distances(values, prototypes, metric=metric)
    return np.argmin(distances, axis=1).astype(np.int64, copy=False)


def predict_labels(model: HabitatModel, matrix: np.ndarray) -> np.ndarray:
    """
    Label feature rows with the model's own decision rule.

    Args:
        model: Fitted habitat definition.
        matrix: Feature rows in ``model.feature_names`` order.

    Returns:
        Integer labels of shape ``(n_samples,)``, starting at 0. Habitat
        ids on a map are these values plus one.

    Raises:
        HABITAPIError: If the stored rule cannot be executed.
    """
    rule = assignment_rule(model)
    values = np.asarray(matrix, dtype=np.float64)
    if rule == "kmeans":
        return _kmeans_predict(model.centroids, values)
    if rule == "gmm":
        labels, _proba = _gmm_predict(model, values)
        return labels
    if rule == "nearest_centroid":
        metric = str(model.estimator_state.get("metric", "euclidean"))
        return assign_to_centers(values, model.centroids, metric)
    raise HABITAPIError(
        f"Habitat model {model.model_id!r} has unknown assignment rule {rule!r}."
    )


def predict_proba_frame(
    model: HabitatModel,
    matrix: np.ndarray,
    index: pd.Index,
) -> pd.DataFrame:
    """
    Return per-row habitat probabilities when the estimator provides them.

    Args:
        model: Fitted habitat definition.
        matrix: Feature rows in ``model.feature_names`` order.
        index: Row index, typically the supervoxel ids.

    Returns:
        A frame with one column ``habitat_{id}`` per component, in the
        estimator's component order. Rows sum to 1.

    Raises:
        HABITAPIError: If the decision rule has no ``predict_proba``.
    """
    rule = assignment_rule(model)
    if rule != "gmm":
        raise HABITAPIError(
            f"Assignment rule {rule!r} has no predict_proba. K-means calls "
            "sklearn.cluster.KMeans.predict, which returns hard labels only. "
            "Class-centre assignment (euclidean, manhattan, cosine) is also "
            "a hard label."
        )
    _labels, proba = _gmm_predict(model, np.asarray(matrix, dtype=np.float64))
    columns = [f"habitat_{class_id + 1}" for class_id in range(proba.shape[1])]
    frame = pd.DataFrame(proba, index=index, columns=columns)
    frame.index.name = index.name
    return frame


def _check_metric(metric: str) -> None:
    """Raise if ``metric`` is not one of the supported centre distances."""
    if metric not in CENTROID_METRICS:
        raise HABITAPIError(
            f"Class-centre metric must be one of {CENTROID_METRICS}; "
            f"got {metric!r}."
        )


def _mean_direction(block: np.ndarray) -> np.ndarray:
    """Return the mean of L2-normalised rows, the cosine class centre."""
    _reject_zero_rows(block, what="training row")
    norms = np.linalg.norm(block, axis=1, keepdims=True)
    centre = (block / norms).mean(axis=0)
    if not np.isfinite(centre).all() or float(np.linalg.norm(centre)) == 0.0:
        raise HABITAPIError(
            "Cosine class centre is undefined: the mean of the L2-normalised "
            "rows is the zero vector."
        )
    return centre


def _reject_zero_rows(matrix: np.ndarray, *, what: str) -> None:
    """Raise when a row has zero Euclidean length (cosine is undefined)."""
    norms = np.linalg.norm(np.asarray(matrix, dtype=np.float64), axis=1)
    if np.any(norms == 0.0):
        raise HABITAPIError(
            f"Cosine distance is undefined for a {what} with zero length."
        )


def _kmeans_predict(centroids: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Call ``KMeans.predict`` on a reconstructed estimator.

    Only ``cluster_centers_`` is required for prediction. The object is
    rebuilt from the stored centres so the label is scikit-learn's nearest
    code in the codebook, not a second implementation of Euclidean distance.
    """
    from sklearn.cluster import KMeans

    centers = np.asarray(centroids, dtype=np.float64)
    estimator = KMeans(n_clusters=int(centers.shape[0]), n_init=1)
    estimator.cluster_centers_ = centers
    estimator.n_features_in_ = int(centers.shape[1])
    # ``predict`` reads this attribute; a reconstructed estimator never ran
    # ``fit``, so the thread count has to be set explicitly.
    estimator._n_threads = 1
    labels = estimator.predict(np.asarray(matrix, dtype=np.float64))
    return np.asarray(labels, dtype=np.int64)


def _gmm_predict(
    model: HabitatModel,
    matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Call ``GaussianMixture.predict`` and ``predict_proba``.

    Weights, means and covariances are the fitted mixture. The Cholesky
    factor of the precision is recomputed with scikit-learn's own helper
    so ``predict`` matches a model that was never serialised as a pickle.
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

    state = model.estimator_state
    missing = [
        key for key in ("weights", "covariances", "covariance_type") if key not in state
    ]
    if missing:
        raise HABITAPIError(
            f"Habitat model {model.model_id!r} is missing GMM parameters "
            f"{missing}. Refit so weights and covariances are stored."
        )
    covariance_type = str(state["covariance_type"])
    means = np.asarray(model.centroids, dtype=np.float64)
    weights = np.asarray(state["weights"], dtype=np.float64)
    covariances = np.asarray(state["covariances"], dtype=np.float64)
    estimator = GaussianMixture(
        n_components=int(means.shape[0]),
        covariance_type=covariance_type,
    )
    estimator.weights_ = weights
    estimator.means_ = means
    estimator.covariances_ = covariances
    estimator.precisions_cholesky_ = _compute_precision_cholesky(
        covariances, covariance_type
    )
    estimator.n_features_in_ = int(means.shape[1])
    values = np.asarray(matrix, dtype=np.float64)
    labels = np.asarray(estimator.predict(values), dtype=np.int64)
    proba = np.asarray(estimator.predict_proba(values), dtype=np.float64)
    return labels, proba
