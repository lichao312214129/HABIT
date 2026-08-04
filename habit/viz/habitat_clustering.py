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
"""Habitat-clustering figures.

Pure functions: feature matrix and habitat labels in, a matplotlib ``Figure``
out, no filesystem. This module is the v1 home for plots that v0.1 produced
from ``habit.utils.visualization.plot_cluster_results`` inside
``ClusteringService.visualize_habitat_clustering``.

Only the population-level 2D PCA scatter is migrated here (B3 phase 1).
Three-dimensional and interactive HTML exports remain in the legacy stack until
a follow-up change lands them behind the same pure-figure contract.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from habit.exceptions import HABITAPIError
from habit.viz.labels import sanitize_label

__all__ = ["plot_habitat_clustering_pca_2d"]


def _plt():
    """Return the pyplot module with the Agg canvas guaranteed headless."""
    import matplotlib

    if matplotlib.get_backend().lower() not in (
        "agg",
        "module://matplotlib_inline.backend_inline",
    ):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _as_feature_matrix(features: np.ndarray) -> np.ndarray:
    """Coerce ``features`` to a float64 matrix of shape ``(n_samples, n_features)``."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise HABITAPIError(
            "habit.viz.plot_habitat_clustering_pca_2d: features must be 2D; "
            f"received {matrix.ndim}D."
        )
    if matrix.shape[0] < 2:
        raise HABITAPIError(
            "habit.viz.plot_habitat_clustering_pca_2d: need at least two samples."
        )
    return matrix


def _as_labels(labels: np.ndarray, n_samples: int) -> np.ndarray:
    """Coerce ``labels`` to a 1D integer array aligned with ``features`` rows."""
    vector = np.asarray(labels)
    if vector.ndim != 1:
        raise HABITAPIError(
            "habit.viz.plot_habitat_clustering_pca_2d: labels must be 1D; "
            f"received {vector.ndim}D."
        )
    if vector.shape[0] != n_samples:
        raise HABITAPIError(
            "habit.viz.plot_habitat_clustering_pca_2d: labels length "
            f"{vector.shape[0]} does not match features rows {n_samples}."
        )
    return vector


def _reduce_pca_2d(
    features: np.ndarray,
    centers: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Project ``features`` (and optional ``centers``) to two PCA components.

    When the input already has at most two columns the values are used
    directly so low-dimensional synthetic tests stay deterministic without
    invoking sklearn.

    Returns:
        ``(coords, centers_2d, explained_variance_ratio)`` where the last
        entry is ``None`` when no PCA was applied.
    """
    if features.shape[1] == 1:
        # Single-feature cohorts cannot form a true 2D PCA plane; pad a zero
        # second axis so the scatter remains plottable on synthetic micro-cohorts.
        pad = np.zeros((features.shape[0], 1), dtype=np.float64)
        coords = np.hstack([features, pad])
        centers_2d = None
        if centers is not None:
            c = np.asarray(centers, dtype=np.float64)
            c_pad = np.zeros((c.shape[0], 1), dtype=np.float64)
            centers_2d = np.hstack([c[:, :1], c_pad])
        return coords, centers_2d, None
    if features.shape[1] <= 2:
        coords = features[:, :2]
        centers_2d = None if centers is None else np.asarray(centers, dtype=np.float64)[:, :2]
        return coords, centers_2d, None

    from sklearn.decomposition import PCA

    reducer = PCA(n_components=2)
    coords = reducer.fit_transform(features)
    centers_2d = None if centers is None else reducer.transform(np.asarray(centers, dtype=np.float64))
    return coords, centers_2d, reducer.explained_variance_ratio_


def _palette_colors(n_items: int, palette: Sequence[str]) -> list:
    """Cycle through ``palette`` until ``n_items`` colours are available."""
    if n_items <= 0:
        return []
    if not palette:
        raise HABITAPIError(
            "habit.viz.plot_habitat_clustering_pca_2d: palette must not be empty."
        )
    return [palette[index % len(palette)] for index in range(n_items)]


def plot_habitat_clustering_pca_2d(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    centers: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    n_clusters: Optional[int] = None,
    palette: Optional[Sequence[str]] = None,
    alpha: float = 0.7,
    marker_size: int = 20,
    center_marker: str = "X",
    center_size: int = 50,
    center_color: str = "#000000",
    show_grid: bool = True,
    grid_alpha: float = 0.3,
    max_legend_items: int = 10,
):
    """
    Two-dimensional PCA scatter of habitat clustering units.

    Each point is one clustering unit (supervoxel, voxel or pooled habitat
    row) coloured by its assigned habitat id. Optional ``centers`` are
    projected with the same PCA fitted on ``features``, matching the v0.1
    ``plot_cluster_results(..., plot_3d=False)`` behaviour for population-level
    habitat clustering.

    Args:
        features: Feature matrix, shape ``(n_samples, n_features)``.
        labels: Habitat assignment per row, shape ``(n_samples,)``.
        centers: Optional centroid matrix, shape ``(n_habitats, n_features)``.
        title: Figure title; defaults to a population-level English caption.
        n_clusters: Selected cluster count for the default title; inferred from
            ``labels`` when omitted.
        palette: Optional colour cycle; defaults to the active matplotlib
            cycle from :func:`~habit.viz.use_style`.
        alpha: Scatter-point transparency in ``[0, 1]``.
        marker_size: Scatter marker area in points squared.
        center_marker: Marker style for centroids.
        center_size: Centroid marker area in points squared.
        center_color: Centroid colour.
        show_grid: Draw a light dashed grid on both axes.
        grid_alpha: Grid-line transparency.
        max_legend_items: Hide the legend when the habitat count exceeds this.

    Returns:
        A matplotlib ``Figure``. The caller owns persistence and display.
    """
    matrix = _as_feature_matrix(features)
    habitat_labels = _as_labels(labels, matrix.shape[0])

    centers_array: Optional[np.ndarray] = None
    if centers is not None:
        centers_array = np.asarray(centers, dtype=np.float64)
        if centers_array.ndim != 2:
            raise HABITAPIError(
                "habit.viz.plot_habitat_clustering_pca_2d: centers must be 2D; "
                f"received {centers_array.ndim}D."
            )
        if centers_array.shape[1] != matrix.shape[1]:
            raise HABITAPIError(
                "habit.viz.plot_habitat_clustering_pca_2d: centers column "
                f"count {centers_array.shape[1]} does not match features "
                f"columns {matrix.shape[1]}."
            )

    coords, centers_2d, explained_var = _reduce_pca_2d(matrix, centers_array)

    unique_labels = np.unique(habitat_labels)
    n_habitats = len(unique_labels)
    cluster_count = n_clusters if n_clusters is not None else n_habitats

    plt = _plt()
    fig, ax = plt.subplots()

    if palette is None:
        cycle = plt.rcParams.get("axes.prop_cycle")
        palette = tuple(cycle.by_key()["color"])
    colors = _palette_colors(n_habitats, palette)

    for index, habitat_id in enumerate(unique_labels):
        mask = habitat_labels == habitat_id
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[colors[index]],
            label=f"Habitat {int(habitat_id)}",
            alpha=alpha,
            s=marker_size,
            zorder=1,
        )

    if centers_2d is not None:
        ax.scatter(
            centers_2d[:, 0],
            centers_2d[:, 1],
            c=center_color,
            marker=center_marker,
            s=center_size,
            label="Centroids",
            edgecolors="none",
            alpha=1.0,
            zorder=10,
        )

    if explained_var is not None:
        ax.set_xlabel(f"PC1 ({explained_var[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained_var[1] * 100:.1f}%)")
    elif matrix.shape[1] == 2:
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
    else:
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    display_title = title
    if display_title is None:
        display_title = (
            f"Habitat Clustering (Population Level)\n(n_clusters={cluster_count})"
        )
    ax.set_title(sanitize_label(display_title))

    if n_habitats <= max_legend_items:
        ax.legend(loc="best", fontsize=8)
    if show_grid:
        ax.grid(True, linestyle="--", alpha=grid_alpha)

    fig.tight_layout()
    return fig
