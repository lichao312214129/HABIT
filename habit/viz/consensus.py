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
"""Figures for a consensus-clustering habitat fit.

These helpers draw the numeric report stored at
``HabitatModel.preprocessing_state["consensus"]``. They do not import the
GPL consensus implementation. The report itself was produced by that
GPL-3.0-or-later code; see ``NOTICE``.

Every function returns a matplotlib ``Figure`` and does not write a file.
Axis text is English.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

import numpy as np

from habit.exceptions import HABITAPIError
from habit.utils.optional_deps import require
from habit.viz.labels import sanitize_label
from habit.viz.style import use_style

__all__ = [
    "plot_consensus_delta",
    "plot_consensus_cdf",
    "plot_consensus_matrix",
    "plot_consensus_matrices",
    "plot_consensus_cluster_stability",
    "plot_consensus_item_stability",
]

_VIZ_PURPOSE = "consensus clustering figures"


def _plt():
    """
    Return pyplot with a non-interactive Agg backend when possible.

    Returns:
        The ``matplotlib.pyplot`` module.

    Raises:
        OptionalDependencyError: When matplotlib is not installed.
    """
    matplotlib = require("matplotlib", extra="viz", purpose=_VIZ_PURPOSE)
    if matplotlib.get_backend().lower() not in (
        "agg",
        "module://matplotlib_inline.backend_inline",
    ):
        matplotlib.use("Agg")
    return require("matplotlib.pyplot", extra="viz", purpose=_VIZ_PURPOSE)


def consensus_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Return the consensus sub-report from a fitted model or from the sub-report.

    Args:
        report: Either ``preprocessing_state`` or the ``"consensus"`` dict
            written by :class:`~habit.habitat_model.ConsensusHabitatModelFitter`.

    Returns:
        The dict that holds ``candidates``, ``delta_k`` and related arrays.

    Raises:
        HABITAPIError: If the mapping is not a consensus report.
    """
    if "delta_k" in report or "candidates" in report:
        return report
    nested = report.get("consensus")
    if isinstance(nested, Mapping) and (
        "delta_k" in nested or "candidates" in nested
    ):
        return nested
    raise HABITAPIError(
        "Expected a consensus report (HabitatModel.preprocessing_state "
        "or its 'consensus' entry)."
    )


def _candidates(report: Mapping[str, Any]) -> List[int]:
    """Return the candidate habitat counts stored on the report."""
    values = consensus_report(report).get("candidates")
    if not values:
        raise HABITAPIError("Consensus report has no candidate k values.")
    return [int(v) for v in values]


def _matrix_stack(report: Mapping[str, Any]) -> np.ndarray:
    """
    Return consensus matrices stacked on axis 0, one per candidate ``k``.

    Args:
        report: Consensus report.

    Returns:
        Array of shape ``(n_k, n_items, n_items)``.

    Raises:
        HABITAPIError: If matrices were not stored on the model.
    """
    body = consensus_report(report)
    matrices = body.get("matrices")
    if matrices is None:
        raise HABITAPIError(
            "This consensus report has no matrices. They are stored only "
            "when the number of items is small enough to keep on the model."
        )
    stack = np.asarray(matrices, dtype=np.float64)
    if stack.ndim != 3:
        raise HABITAPIError(
            "Consensus matrices must have shape (n_k, n_items, n_items); "
            f"got {stack.shape}."
        )
    return stack


def _matrix_for_k(report: Mapping[str, Any], k: Optional[int]) -> Tuple[int, np.ndarray]:
    """
    Return ``(k, matrix)`` for one candidate, defaulting to the selected ``k``.

    Args:
        report: Consensus report.
        k: Candidate count. ``None`` uses ``selected_k``.

    Returns:
        The chosen ``k`` and its consensus matrix.
    """
    body = consensus_report(report)
    chosen = int(body["selected_k"] if k is None else k)
    candidates = _candidates(body)
    if chosen not in candidates:
        raise HABITAPIError(
            f"k={chosen} is not in the consensus candidates {candidates}."
        )
    stack = _matrix_stack(body)
    return chosen, stack[candidates.index(chosen)]


def _linkage_order(matrix: np.ndarray) -> np.ndarray:
    """
    Order items by average-linkage on ``1 - consensus``.

    Args:
        matrix: Square consensus matrix, values in ``[0, 1]``.

    Returns:
        Integer permutation of the rows. The identity when linkage cannot run.
    """
    similarity = np.asarray(matrix, dtype=np.float64)
    similarity = 0.5 * (similarity + similarity.T)
    np.fill_diagonal(similarity, 1.0)
    if similarity.shape[0] < 3:
        return np.arange(similarity.shape[0])
    distance = np.clip(1.0 - similarity, 0.0, None)
    np.fill_diagonal(distance, 0.0)
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    condensed = squareform(distance, checks=False)
    try:
        return np.asarray(leaves_list(linkage(condensed, method="average")), dtype=int)
    except ValueError:
        return np.arange(similarity.shape[0])


def plot_consensus_delta(
    report: Mapping[str, Any],
    *,
    title: Optional[str] = None,
):
    """
    Plot the relative change in area under the consensus-index CDF.

    The x coordinates are the ones stored by the InMoose / Sajovic rule
    (``delta_k_x``). The selected ``k`` is marked when it lies on that axis.
    ``max_habitats`` is not on the axis and is not selectable by this curve.

    Args:
        report: Consensus report, or a preprocessing state that contains it.
        title: Figure title. A default is used when omitted.

    Returns:
        A matplotlib ``Figure``.
    """
    body = consensus_report(report)
    delta = np.asarray(body.get("delta_k"), dtype=np.float64).ravel()
    if delta.size == 0:
        raise HABITAPIError(
            "No delta(k) curve: consensus was run at a single k."
        )
    x_values = [int(v) for v in body.get("delta_k_x", [])]
    if len(x_values) != delta.size:
        raise HABITAPIError(
            "delta_k and delta_k_x must have the same length."
        )
    selected = int(body["selected_k"])
    plt = _plt()
    with use_style("radiology") as style:
        fig, ax = plt.subplots(figsize=style.figsize(columns=1), constrained_layout=True)
        ax.plot(x_values, delta, marker="o", color="#0072B2", linewidth=style.line_width)
        if selected in x_values:
            y_mark = float(delta[x_values.index(selected)])
            ax.scatter(
                [selected],
                [y_mark],
                s=60,
                zorder=3,
                color="#D55E00",
                label=f"selected k={selected}",
            )
            ax.legend(
                frameon=False,
                fontsize=8,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )
        ax.set_xticks(x_values)
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Relative change in area under CDF")
        ax.set_title(
            sanitize_label(title or "Consensus clustering: delta(k)")
        )
    return fig


def plot_consensus_cdf(
    report: Mapping[str, Any],
    *,
    title: Optional[str] = None,
):
    """
    Plot the empirical CDF of each consensus matrix.

    Entries include the diagonal, which the vendored area calculation also
    includes (those diagonal entries are 1).

    Args:
        report: Consensus report, or a preprocessing state that contains it.
        title: Figure title.

    Returns:
        A matplotlib ``Figure``.
    """
    body = consensus_report(report)
    stack = _matrix_stack(body)
    candidates = _candidates(body)
    if stack.shape[0] != len(candidates):
        raise HABITAPIError(
            "Consensus matrix count does not match the candidate list."
        )
    selected = int(body["selected_k"])
    plt = _plt()
    with use_style("radiology") as style:
        fig, ax = plt.subplots(figsize=style.figsize(columns=1), constrained_layout=True)
        for index, k in enumerate(candidates):
            values = np.sort(stack[index].ravel())
            cdf = np.linspace(0.0, 1.0, values.size, endpoint=True)
            width = 2.0 if k == selected else style.line_width
            ax.plot(values, cdf, linewidth=width, label=f"k={k}")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Consensus index")
        ax.set_ylabel("Empirical CDF")
        ax.set_title(sanitize_label(title or "Consensus index CDF"))
        ax.legend(
            frameon=False,
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
    return fig


def plot_consensus_matrix(
    report: Mapping[str, Any],
    *,
    k: Optional[int] = None,
    title: Optional[str] = None,
):
    """
    Plot one consensus matrix with items ordered by average linkage.

    Args:
        report: Consensus report, or a preprocessing state that contains it.
        k: Candidate count. ``None`` uses the selected ``k``.
        title: Figure title.

    Returns:
        A matplotlib ``Figure``.
    """
    chosen, matrix = _matrix_for_k(report, k)
    order = _linkage_order(matrix)
    ordered = matrix[np.ix_(order, order)]
    plt = _plt()
    with use_style("radiology") as style:
        fig, ax = plt.subplots(figsize=style.figsize(columns=1), constrained_layout=True)
        image = ax.imshow(
            ordered,
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
            interpolation="nearest",
            aspect="equal",
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="consensus index")
        ax.set_xlabel("Item (average-linkage order)")
        ax.set_ylabel("Item (average-linkage order)")
        ax.set_title(
            sanitize_label(title or f"Consensus matrix, k={chosen}")
        )
        ax.set_xticks([])
        ax.set_yticks([])
    return fig


def plot_consensus_matrices(
    report: Mapping[str, Any],
    *,
    title: Optional[str] = None,
):
    """
    Plot every candidate consensus matrix in one figure.

    Args:
        report: Consensus report, or a preprocessing state that contains it.
        title: Figure title. Panel titles are ``k=<count>``.

    Returns:
        A matplotlib ``Figure``.
    """
    body = consensus_report(report)
    stack = _matrix_stack(body)
    candidates = _candidates(body)
    n_panels = len(candidates)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    plt = _plt()
    with use_style("radiology") as style:
        width, _ = style.figsize(columns=2)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(max(width, 2.6 * ncols), 2.6 * nrows),
            constrained_layout=True,
            squeeze=False,
        )
        last_image = None
        for index, k in enumerate(candidates):
            row, col = divmod(index, ncols)
            ax = axes[row][col]
            order = _linkage_order(stack[index])
            ordered = stack[index][np.ix_(order, order)]
            last_image = ax.imshow(
                ordered,
                vmin=0.0,
                vmax=1.0,
                cmap="Blues",
                interpolation="nearest",
                aspect="equal",
            )
            ax.set_title(f"k={k}")
            ax.set_xticks([])
            ax.set_yticks([])
        for index in range(n_panels, nrows * ncols):
            row, col = divmod(index, ncols)
            axes[row][col].axis("off")
        if last_image is not None:
            fig.colorbar(
                last_image,
                ax=axes.ravel().tolist(),
                fraction=0.03,
                pad=0.02,
                label="consensus index",
            )
        fig.suptitle(sanitize_label(title or "Consensus matrices"))
    return fig


def plot_consensus_cluster_stability(
    report: Mapping[str, Any],
    *,
    title: Optional[str] = None,
):
    """
    Plot per-cluster consensus (mean pairwise co-clustering inside a cluster).

    One group of bars per candidate ``k``. A value near 1 means the items
    placed in that cluster co-clustered in almost every resample that drew
    them. Empty or singleton clusters are stored as NaN and left as gaps.

    Args:
        report: Consensus report, or a preprocessing state that contains it.
        title: Figure title.

    Returns:
        A matplotlib ``Figure``.
    """
    body = consensus_report(report)
    by_k = body.get("cluster_consensus")
    if not isinstance(by_k, Mapping) or not by_k:
        raise HABITAPIError("Consensus report has no cluster_consensus table.")
    candidates = _candidates(body)
    plt = _plt()
    with use_style("radiology") as style:
        fig, ax = plt.subplots(figsize=style.figsize(columns=1), constrained_layout=True)
        max_clusters = max(len(by_k[str(k)]) for k in candidates if str(k) in by_k)
        width = 0.8 / max(max_clusters, 1)
        palette = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000")
        for cluster_index in range(max_clusters):
            heights: List[float] = []
            positions: List[float] = []
            for k_index, k in enumerate(candidates):
                values = by_k.get(str(k), [])
                if cluster_index >= len(values):
                    continue
                positions.append(k_index + cluster_index * width)
                heights.append(float(values[cluster_index]))
            ax.bar(
                positions,
                heights,
                width=width * 0.95,
                color=palette[cluster_index % len(palette)],
                label=f"cluster {cluster_index}",
            )
        centers = [
            i + (max_clusters - 1) * width / 2.0 for i in range(len(candidates))
        ]
        ax.set_xticks(centers)
        ax.set_xticklabels([str(k) for k in candidates])
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Cluster consensus")
        ax.set_title(
            sanitize_label(title or "Cluster consensus (stability)")
        )
        ax.legend(
            frameon=False,
            fontsize=8,
            title="Cluster",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
    return fig


def plot_consensus_item_stability(
    report: Mapping[str, Any],
    *,
    title: Optional[str] = None,
):
    """
    Plot item consensus for the selected ``k``.

    Each row is one training item. Each column is one cluster. The value is
    the mean consensus index between that item and the members of that
    cluster, as defined by the vendored implementation.

    Args:
        report: Consensus report, or a preprocessing state that contains it.
        title: Figure title.

    Returns:
        A matplotlib ``Figure``.
    """
    body = consensus_report(report)
    matrix = np.asarray(body.get("item_consensus"), dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise HABITAPIError("Consensus report has no item_consensus matrix.")
    labels = np.asarray(body.get("item_labels", body.get("training_labels")), dtype=int)
    if labels.shape[0] == matrix.shape[0]:
        order = np.argsort(labels, kind="mergesort")
        matrix = matrix[order]
    k = int(body.get("item_consensus_k", body.get("selected_k")))
    plt = _plt()
    with use_style("radiology") as style:
        fig, ax = plt.subplots(figsize=style.figsize(columns=1), constrained_layout=True)
        image = ax.imshow(
            matrix,
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
            interpolation="nearest",
            aspect="auto",
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="item consensus")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Item (sorted by consensus label)")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_title(sanitize_label(title or f"Item consensus, k={k}"))
    return fig
