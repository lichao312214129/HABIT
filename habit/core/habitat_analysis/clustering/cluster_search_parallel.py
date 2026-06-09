# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
"""
Parallel cluster-count search for group-level habitat clustering.

Each candidate ``k`` is evaluated in an isolated worker process: one model fit
per ``k``, then all requested validation metrics are derived from that fit.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from habit.core.habitat_analysis.clustering.cluster_validation_methods import (
    is_valid_method_for_algorithm,
)

logger = logging.getLogger(__name__)

# Label-based metrics computed from cluster assignments after a single fit.
_LABEL_BASED_METHODS = frozenset(
    {"silhouette", "calinski_harabasz", "davies_bouldin", "gap"}
)

# Inertia-based metrics share the same scalar per k for KMeans.
_INERTIA_BASED_METHODS = frozenset({"inertia", "kneedle", "elbow"})

# GMM information criteria computed directly from the fitted mixture model.
_GMM_CRITERIA_METHODS = frozenset({"aic", "bic"})


def resolve_cluster_search_workers(cluster_search_workers: Optional[int]) -> int:
    """
    Resolve worker count for parallel cluster-count search.

    Args:
        cluster_search_workers: Explicit worker count from YAML, or ``None`` for
            ``max(1, cpu_count - 4)``.

    Returns:
        int: Number of worker processes (at least 1).
    """
    if cluster_search_workers is not None:
        return max(1, int(cluster_search_workers))
    from habit.utils.parallel_utils import default_cluster_search_workers

    return default_cluster_search_workers()


def _limit_blas_threads_in_worker() -> None:
    """Avoid nested BLAS threading when multiple cluster fits run in parallel."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _merged_constructor_kwargs(
    model_params: Mapping[str, Any],
    *,
    top_level_keys: Sequence[str],
) -> Dict[str, Any]:
    """
    Merge top-level clusterer attrs with the nested ``kwargs`` bag.

    Habit clusterers such as :class:`KMeansClustering` store ``max_iter`` in
    ``self.kwargs`` because it is not a dedicated constructor attribute. Top-level
    values win when both places define the same key.

    Args:
        model_params: Serialized parameters from ``_cluster_search_model_params``.
        top_level_keys: Constructor keys to lift from ``model_params``.

    Returns:
        Dict[str, Any]: De-duplicated kwargs safe to pass to sklearn once.
    """
    merged: Dict[str, Any] = dict(model_params.get("kwargs", {}))
    for key in top_level_keys:
        if key in model_params and key != "kwargs":
            merged[key] = model_params[key]
    return merged


def _fit_clustering_model(
    X: np.ndarray,
    n_clusters: int,
    algorithm: str,
    model_params: Mapping[str, Any],
) -> Tuple[np.ndarray, Optional[float], Optional[float], Optional[float]]:
    """
    Fit one clustering model for a single candidate ``k``.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        n_clusters: Candidate cluster count.
        algorithm: ``kmeans`` or ``gmm``.
        model_params: Constructor kwargs excluding ``n_clusters`` / ``n_components``.

    Returns:
        Tuple containing:
            - labels: Cluster assignment per row.
            - inertia: KMeans inertia, or ``None`` for GMM.
            - aic: GMM AIC, or ``None`` for KMeans.
            - bic: GMM BIC, or ``None`` for KMeans.
    """
    algo = algorithm.lower()
    if algo == "kmeans":
        from sklearn.cluster import KMeans

        km_kwargs = _merged_constructor_kwargs(
            model_params,
            top_level_keys=("random_state", "init", "n_init", "max_iter"),
        )
        km_kwargs.setdefault("random_state", 0)
        km_kwargs.setdefault("init", "k-means++")
        km_kwargs.setdefault("n_init", 10)
        km_kwargs.setdefault("max_iter", 300)
        km = KMeans(n_clusters=n_clusters, **km_kwargs)
        labels = km.fit_predict(X)
        return labels, float(km.inertia_), None, None

    if algo == "gmm":
        from sklearn.mixture import GaussianMixture

        gmm_kwargs = _merged_constructor_kwargs(
            model_params,
            top_level_keys=("random_state", "covariance_type", "n_init", "max_iter"),
        )
        gmm_kwargs.setdefault("random_state", 0)
        gmm_kwargs.setdefault("covariance_type", "full")
        gmm_kwargs.setdefault("n_init", 10)
        gmm_kwargs.setdefault("max_iter", 300)
        gmm = GaussianMixture(n_components=n_clusters, **gmm_kwargs)
        gmm.fit(X)
        labels = gmm.predict(X)
        return labels, None, float(gmm.aic(X)), float(gmm.bic(X))

    raise ValueError(
        f"Parallel cluster search supports 'kmeans' and 'gmm'; got {algorithm!r}."
    )


def compute_validation_scores_for_k(
    X: np.ndarray,
    n_clusters: int,
    algorithm: str,
    model_params: Mapping[str, Any],
    methods: Sequence[str],
) -> Dict[str, float]:
    """
    Fit once for candidate ``k`` and compute all applicable validation metrics.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        n_clusters: Candidate cluster count.
        algorithm: Clustering algorithm name (``kmeans`` or ``gmm``).
        model_params: Serialized constructor parameters for the clusterer.
        methods: Validation metric names requested by the caller.

    Returns:
        Dict[str, float]: Metric name to scalar score for this ``k``.
    """
    labels, inertia, aic, bic = _fit_clustering_model(
        X, n_clusters, algorithm, model_params
    )
    unique_labels = np.unique(labels)
    has_multiple_clusters: bool = unique_labels.size > 1

    scores: Dict[str, float] = {}
    for method in methods:
        if not is_valid_method_for_algorithm(algorithm, method):
            continue

        if method in _INERTIA_BASED_METHODS:
            if inertia is not None:
                scores[method] = inertia
            continue

        if method == "aic" and aic is not None:
            scores[method] = aic
            continue

        if method == "bic" and bic is not None:
            scores[method] = bic
            continue

        if method in _LABEL_BASED_METHODS:
            if not has_multiple_clusters:
                scores[method] = 0.0
                continue
            if method == "silhouette":
                scores[method] = float(silhouette_score(X, labels))
            elif method == "calinski_harabasz":
                scores[method] = float(calinski_harabasz_score(X, labels))
            elif method == "davies_bouldin":
                scores[method] = float(davies_bouldin_score(X, labels))
            elif method == "gap":
                # Gap is only exposed for one_step; serial path is also undefined here.
                raise NotImplementedError(
                    "Gap statistic is not supported in parallel cluster search."
                )

    return scores


def _init_cluster_search_worker() -> None:
    """
    Configure logging and BLAS limits inside cluster-search worker processes.

    Windows spawn workers start without the parent's logging handlers, so this
    initializer restores habit logging when possible and falls back to stderr.
    """
    _limit_blas_threads_in_worker()
    try:
        from habit.utils.log_utils import restore_logging_in_subprocess

        restore_logging_in_subprocess()
    except Exception:
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
            )


def _format_score_snapshot(
    per_k_scores: Mapping[str, float],
    methods: Sequence[str],
) -> str:
    """
    Build a compact score summary for log lines.

    Args:
        per_k_scores: Metric values for one candidate k.
        methods: Metrics requested by the caller.

    Returns:
        str: Comma-separated ``metric=value`` text.
    """
    parts: List[str] = []
    for method in methods:
        if method not in per_k_scores:
            continue
        value = per_k_scores[method]
        parts.append(f"{method}={value:.6g}")
    return ", ".join(parts) if parts else "no-scores"


def _evaluate_cluster_count_worker(
    args: Tuple[int, np.ndarray, str, Dict[str, Any], Tuple[str, ...]],
) -> Tuple[int, Dict[str, float], float]:
    """
    Process-pool entry point for evaluating one candidate cluster count.

    Args:
        args: Tuple of
            (n_clusters, X, algorithm, model_params, methods).

    Returns:
        Tuple[int, Dict[str, float], float]:
            Candidate k, per-metric scores, and worker elapsed seconds.
    """
    n_clusters, X, algorithm, model_params, methods = args
    started_at = time.monotonic()
    from habit.utils.log_utils import get_module_logger

    worker_logger = get_module_logger(__name__)
    worker_logger.info(
        "Parallel cluster search worker started: algorithm=%s k=%d",
        algorithm,
        n_clusters,
    )
    scores = compute_validation_scores_for_k(
        X=X,
        n_clusters=n_clusters,
        algorithm=algorithm,
        model_params=model_params,
        methods=methods,
    )
    elapsed_sec = time.monotonic() - started_at
    worker_logger.info(
        "Parallel cluster search worker finished: algorithm=%s k=%d elapsed_sec=%.2f scores=%s",
        algorithm,
        n_clusters,
        elapsed_sec,
        _format_score_snapshot(scores, methods),
    )
    return n_clusters, scores, elapsed_sec


def parallel_cluster_search_scores(
    X: np.ndarray,
    cluster_range: List[int],
    algorithm: str,
    model_params: Mapping[str, Any],
    methods: Sequence[str],
    *,
    n_workers: int,
    show_progress: bool = True,
    log: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, List[float]], List[str]]:
    """
    Evaluate all candidate cluster counts in parallel.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        cluster_range: Ordered candidate cluster counts.
        algorithm: Clustering algorithm name (``kmeans`` or ``gmm``).
        model_params: Serialized constructor parameters shared across all k.
        methods: Requested validation metric names.
        n_workers: Maximum concurrent worker processes.
        show_progress: Whether to emit per-k completion logs.
        log: Optional logger for progress messages.

    Returns:
        Tuple containing:
            - scores_dict: metric -> score list aligned with ``cluster_range``.
            - valid_methods: Metrics that produced at least one score.
    """
    if not cluster_range:
        raise ValueError("cluster_range must not be empty.")

    X_array = np.asarray(X, dtype=np.float64)
    algo = algorithm.lower()
    valid_methods: List[str] = [
        method
        for method in methods
        if is_valid_method_for_algorithm(algo, method)
    ]
    if not valid_methods:
        raise ValueError(
            f"No valid validation methods for algorithm {algorithm!r}: {list(methods)}"
        )

    worker_count = max(1, min(int(n_workers), len(cluster_range)))
    methods_tuple = tuple(valid_methods)
    model_params_dict = dict(model_params)

    if log is not None and show_progress:
        log.info(
            "Parallel cluster search: algorithm=%s workers=%d candidates=%s methods=%s",
            algo,
            worker_count,
            cluster_range,
            ", ".join(valid_methods),
        )

    scores_by_k: Dict[int, Dict[str, float]] = {}
    job_args = [
        (k, X_array, algo, model_params_dict, methods_tuple) for k in cluster_range
    ]
    total_jobs = len(job_args)

    if log is not None and show_progress:
        for job_index, args in enumerate(job_args, start=1):
            log.info(
                "Parallel cluster search queued k=%d [%d/%d]",
                args[0],
                job_index,
                total_jobs,
            )

    search_started_at = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_cluster_search_worker,
    ) as executor:
        futures = {
            executor.submit(_evaluate_cluster_count_worker, args): args[0]
            for args in job_args
        }
        for future in as_completed(futures):
            n_clusters, per_k_scores, worker_elapsed_sec = future.result()
            scores_by_k[n_clusters] = per_k_scores
            if log is not None and show_progress:
                log.info(
                    "Parallel cluster search finished k=%d [%d/%d] worker_elapsed_sec=%.2f scores=%s",
                    n_clusters,
                    len(scores_by_k),
                    total_jobs,
                    worker_elapsed_sec,
                    _format_score_snapshot(per_k_scores, valid_methods),
                )

    if log is not None and show_progress:
        log.info(
            "Parallel cluster search completed: candidates=%s total_elapsed_sec=%.2f",
            cluster_range,
            time.monotonic() - search_started_at,
        )

    scores_dict: Dict[str, List[float]] = {}
    for method in valid_methods:
        scores_dict[method] = [
            float(scores_by_k[k].get(method, float("nan"))) for k in cluster_range
        ]

    return scores_dict, valid_methods
