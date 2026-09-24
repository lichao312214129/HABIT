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
"""Figures that show how habitat labels were matched.

Two figures, one per matcher in :mod:`habit.kernels.habitat_label_match`:

* :func:`plot_label_overlap_matrix` -- the voxel-overlap table of two maps
  of the same voxels, with the Hungarian pairs outlined (what
  :func:`~habit.precision.align_habitat_map` and
  :func:`~habit.precision.habitat_stability` decide on).
* :func:`plot_prototype_matching` -- every subject's habitat summaries in
  a 2-D feature plane, coloured by the shared prototype they were named
  after, with a link from each habitat to its prototype (what
  :func:`~habit.precision.align_habitat_maps_to_prototypes` and
  :func:`~habit.kernels.habitat_label_match.match_rows_to_prototypes`
  decide on).

Pure matplotlib: arrays / result objects in, ``Figure`` out, no IO.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from habit.exceptions import HABITAPIError
from habit.kernels.habitat_label_match import (
    PROTOTYPE_METRICS,
    match_labels_by_overlap,
    overlap_count_table,
)
from habit.viz.colorbar import ColorbarSpec, add_image_colorbar_from_spec
from habit.viz.habitat_core import _plt
from habit.viz.habitat_overlay import _habitat_color_lookup
from habit.viz.labels import sanitize_label
from habit.viz.orientation import array_from_display_input
from habit.viz.style import use_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_label_overlap_matrix", "plot_prototype_matching"]

#: Sequential map for overlap counts (greyscale-safe; no rainbow).
_OVERLAP_CMAP = "Blues"

#: Outline colour of the Hungarian-selected cells.
_PAIR_EDGE = "#D55E00"

#: Marker cycle that tells subjects apart in the prototype plane. Colour
#: is reserved for the prototype (= shared habitat id).
_BLOCK_MARKERS: Tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">")

#: Colour of habitats left unnamed (``max_distance`` or extra rows).
_UNMATCHED_FACE = "#9A9A9A"

#: Matrices larger than this are drawn without per-cell numbers.
_MAX_ANNOTATED_CELLS = 12


def _new_axes(ax: Optional["Axes"], height_mm: float) -> Tuple["Figure", "Axes"]:
    """
    Return ``(figure, axes)``: the caller's axes, or a new one-column figure.

    Args:
        ax: Optional existing matplotlib axes (for multi-panel pages).
        height_mm: Height of a new figure in millimetres.

    Returns:
        The owning figure and the axes to draw on.
    """
    if ax is not None:
        return ax.figure, ax
    plt = _plt()
    with use_style("radiology") as style:
        fig, new_ax = plt.subplots(
            1,
            1,
            figsize=style.figsize(columns=1, height_mm=height_mm),
            constrained_layout=True,
        )
    return fig, new_ax


def plot_label_overlap_matrix(
    reference: Any,
    moving: Any,
    *,
    mapping: Optional[Mapping[int, int]] = None,
    normalize: bool = False,
    reference_name: str = "reference",
    moving_name: str = "moving",
    title: Optional[str] = None,
    colorbar: ColorbarSpec = True,
    ax: Optional["Axes"] = None,
) -> "Figure":
    """
    Heatmap of the voxel-overlap table of two label maps of the same voxels.

    Cell ``(i, j)`` counts voxels labelled moving habitat ``i`` and reference
    habitat ``j`` (:func:`~habit.kernels.habitat_label_match.overlap_count_table`).
    The outlined cells are the one-to-one pairs chosen by Hungarian
    assignment on those counts
    (:func:`~habit.kernels.habitat_label_match.match_labels_by_overlap`),
    i.e. the pairing that keeps the largest total number of voxels. A
    moving habitat without an outlined cell has no reference partner (the
    moving map has more habitats) and receives a new id when remapped.

    Args:
        reference: Reference labels: ``np.ndarray`` or ``HabitatMap``.
        moving: Moving labels on the same voxel grid.
        mapping: ``{moving_id: reference_id}`` to outline. ``None`` (default)
            computes the overlap Hungarian pairing.
        normalize: If True, colour each row by the fraction of that moving
            habitat's labelled voxels (rows sum to at most 1), so small
            habitats are as readable as large ones. Numbers stay counts.
        reference_name: Axis name of the reference map.
        moving_name: Axis name of the moving map.
        title: Optional axes title.
        colorbar: ``True`` / ``False`` or colorbar style kwargs.
        ax: Optional axes to draw into (returns its figure).

    Returns:
        A matplotlib ``Figure``.

    Raises:
        HABITAPIError: If the maps differ in shape or share no labelled voxel.
    """
    # Any shape is fine (a volume, a slice, or raveled voxels pooled over
    # subjects); only voxel-by-voxel correspondence matters.
    ref_labels = np.asarray(array_from_display_input(reference), dtype=np.int64)
    mov_labels = np.asarray(array_from_display_input(moving), dtype=np.int64)
    try:
        ref_ids, mov_ids, overlap = overlap_count_table(ref_labels, mov_labels)
        pairs = (
            dict(mapping)
            if mapping is not None
            else match_labels_by_overlap(ref_labels, mov_labels)
        )
    except ValueError as exc:
        raise HABITAPIError(f"plot_label_overlap_matrix: {exc}") from exc
    if overlap.size == 0 or int(overlap.sum()) == 0:
        raise HABITAPIError(
            "plot_label_overlap_matrix: the two maps share no labelled voxel."
        )

    counts = overlap.astype(np.float64)
    if normalize:
        # Row share of each moving habitat's voxels inside the shared ROI.
        row_totals = counts.sum(axis=1, keepdims=True)
        display = np.divide(
            counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0
        )
        cbar_label = f"Fraction of {moving_name} habitat"
        vmax = 1.0
    else:
        display = counts
        cbar_label = "Overlapping voxels"
        vmax = float(counts.max())

    ref_index = {int(v): j for j, v in enumerate(ref_ids.tolist())}
    mov_index = {int(v): i for i, v in enumerate(mov_ids.tolist())}
    matched_rows = {int(m) for m in pairs if int(m) in mov_index}

    fig, axes = _new_axes(ax, height_mm=74.0)
    plt = _plt()
    with use_style("radiology") as style:
        image = axes.imshow(
            display,
            cmap=plt.get_cmap(_OVERLAP_CMAP),
            vmin=0.0,
            vmax=vmax if vmax > 0 else 1.0,
            interpolation="nearest",
            aspect="auto",
        )
        add_image_colorbar_from_spec(image, colorbar, ax=axes, label=cbar_label)
        for moving_id, reference_id in pairs.items():
            if int(moving_id) not in mov_index or int(reference_id) not in ref_index:
                continue
            row = mov_index[int(moving_id)]
            col = ref_index[int(reference_id)]
            axes.add_patch(
                plt.Rectangle(
                    (col - 0.5, row - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor=_PAIR_EDGE,
                    linewidth=2.0,
                )
            )
        if len(ref_ids) <= _MAX_ANNOTATED_CELLS and len(mov_ids) <= _MAX_ANNOTATED_CELLS:
            top = float(display.max()) if display.size else 0.0
            for i in range(counts.shape[0]):
                for j in range(counts.shape[1]):
                    dark = top > 0 and float(display[i, j]) / top >= 0.55
                    axes.text(
                        j,
                        i,
                        f"{int(counts[i, j])}",
                        ha="center",
                        va="center",
                        color="white" if dark else "#222222",
                        fontsize=max(style.font_size - 1.5, 5.0),
                    )
        axes.set_xticks(range(len(ref_ids)))
        axes.set_xticklabels([sanitize_label(f"H{int(v)}") for v in ref_ids])
        axes.set_yticks(range(len(mov_ids)))
        axes.set_yticklabels(
            [
                sanitize_label(
                    f"H{int(v)}" if int(v) in matched_rows else f"H{int(v)} (new)"
                )
                for v in mov_ids
            ]
        )
        axes.set_xlabel(sanitize_label(f"{reference_name} habitat"))
        axes.set_ylabel(sanitize_label(f"{moving_name} habitat"))
        axes.set_title(
            sanitize_label(
                title if title is not None else "Voxel overlap (outlined: Hungarian pairs)"
            )
        )
    return fig


def _blocks_as_arrays(blocks: Sequence[Any]) -> List[np.ndarray]:
    """Coerce every block to a finite 2-D float array (one row per habitat)."""
    arrays: List[np.ndarray] = []
    for index, block in enumerate(blocks):
        array = np.asarray(block, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise HABITAPIError(
                f"plot_prototype_matching: block {index} must be 2-D "
                f"(habitats x features); got shape {array.shape}."
            )
        arrays.append(array)
    if not arrays:
        raise HABITAPIError("plot_prototype_matching: blocks is empty.")
    return arrays


def _resolve_match(
    arrays: Sequence[np.ndarray],
    match: Any,
) -> Tuple[List[np.ndarray], np.ndarray, str, Optional[List[str]], Optional[Tuple[str, ...]]]:
    """
    Read assignments / prototypes from either supported result type.

    Args:
        arrays: Habitat summaries per subject, in the order that was matched.
        match: ``PrototypeMatch`` (kernel) or ``HabitatPrototypeAlignment``.

    Returns:
        ``(assignments, prototypes, metric, block_names, feature_names)``.
        Assignments are 0-based prototype indices with ``-1`` for unnamed
        rows; ``prototypes`` are in the units of ``arrays``.
    """
    table = getattr(match, "assignments", None)
    if hasattr(match, "habitat_maps") and hasattr(table, "columns"):
        # HabitatPrototypeAlignment: the table lists habitats block by block
        # in input order, so consecutive slices recover each block.
        prototype_ids = table["prototype_id"].to_numpy(dtype="float64", na_value=np.nan)
        if prototype_ids.size != sum(a.shape[0] for a in arrays):
            raise HABITAPIError(
                "plot_prototype_matching: blocks do not match the alignment; "
                f"{prototype_ids.size} habitats were aligned but blocks hold "
                f"{sum(a.shape[0] for a in arrays)} rows. Pass the same "
                "summaries (e.g. each model's centroids) in the same order."
            )
        assignments: List[np.ndarray] = []
        start = 0
        for array in arrays:
            chunk = prototype_ids[start : start + array.shape[0]]
            start += array.shape[0]
            assignments.append(
                np.where(np.isnan(chunk), -1, chunk - 1).astype(np.int64)
            )
        names = [str(m.subject_id) for m in match.habitat_maps]
        return (
            assignments,
            np.asarray(match.prototypes, dtype=np.float64),
            "raw",
            names if len(names) == len(arrays) else None,
            tuple(match.feature_names),
        )
    if hasattr(match, "assignments") and hasattr(match, "prototypes") and hasattr(match, "metric"):
        # PrototypeMatch from the kernel: prototypes live in the metric's
        # working space (unit rows for cosine / correlation).
        assignments = [np.asarray(a, dtype=np.int64) for a in match.assignments]
        if len(assignments) != len(arrays) or any(
            a.shape[0] != b.shape[0] for a, b in zip(assignments, arrays)
        ):
            raise HABITAPIError(
                "plot_prototype_matching: blocks must be the ones passed to "
                "match_rows_to_prototypes (same count, same row counts)."
            )
        metric = str(match.metric)
        if metric not in PROTOTYPE_METRICS:
            raise HABITAPIError(
                f"plot_prototype_matching: unknown metric {metric!r}."
            )
        return assignments, np.asarray(match.prototypes, dtype=np.float64), metric, None, None
    raise HABITAPIError(
        "plot_prototype_matching: match must be a PrototypeMatch "
        "(match_rows_to_prototypes) or a HabitatPrototypeAlignment "
        "(align_habitat_maps_to_prototypes)."
    )


def plot_prototype_matching(
    blocks: Sequence[Any],
    match: Any,
    *,
    block_names: Optional[Sequence[str]] = None,
    feature_names: Optional[Sequence[str]] = None,
    feature_axes: Tuple[int, int] = (0, 1),
    show_links: bool = True,
    title: Optional[str] = None,
    ax: Optional["Axes"] = None,
) -> "Figure":
    """
    Habitat summaries of every subject, coloured by their shared prototype.

    Each point is one habitat of one subject (marker = subject, colour =
    the prototype it was named after, i.e. the shared habitat id
    ``k + 1``). Grey hollow points were left unnamed (``max_distance`` or
    more habitats than frozen prototypes). Prototypes are drawn so the
    reader sees *why* a habitat got its name:

    * ``HabitatPrototypeAlignment`` -- the descriptive prototypes (mean or
      median of the input summaries), as black crosses with links;
    * ``PrototypeMatch`` with ``"sqeuclidean"`` / ``"manhattan"`` -- the
      fitted prototypes, as crosses with links (pass the same, possibly
      z-scored, blocks that were matched);
    * ``"cosine"`` -- prototypes are directions, drawn as rays from the
      origin: points on one ray are identical to cosine however far apart;
    * ``"correlation"`` -- prototypes live in row-centred space with no
      point in this plane, so only the colours are drawn.

    Args:
        blocks: Per-subject summaries, shape ``(n_habitats, n_features)``,
            in the order that was matched (e.g. ``[m.centroids for m in models]``).
        match: Result of :func:`~habit.kernels.habitat_label_match.match_rows_to_prototypes`
            or :func:`~habit.precision.align_habitat_maps_to_prototypes`.
        block_names: Legend name per block. Defaults to subject ids for an
            alignment, ``S1, S2, ...`` otherwise.
        feature_names: Axis names. Defaults to the alignment's feature names,
            ``feature 0, feature 1, ...`` otherwise.
        feature_axes: The two feature columns to plot (1-feature summaries
            are drawn against a zero y-axis).
        show_links: Draw a line from each habitat to its prototype.
        title: Optional axes title.
        ax: Optional axes to draw into (returns its figure).

    Returns:
        A matplotlib ``Figure``.

    Raises:
        HABITAPIError: On malformed blocks, mismatched results, or feature
            axes out of range.
    """
    arrays = _blocks_as_arrays(blocks)
    assignments, prototypes, metric, default_names, default_features = _resolve_match(
        arrays, match
    )
    n_features = arrays[0].shape[1]
    if any(a.shape[1] != n_features for a in arrays) or prototypes.shape[1] != n_features:
        raise HABITAPIError(
            "plot_prototype_matching: every block and the prototypes must have "
            "the same number of features."
        )
    x_col = int(feature_axes[0])
    y_col = int(feature_axes[1]) if n_features > 1 else None
    if not 0 <= x_col < n_features or (y_col is not None and not 0 <= y_col < n_features):
        raise HABITAPIError(
            f"plot_prototype_matching: feature_axes {feature_axes} out of range "
            f"for {n_features} features."
        )
    names = (
        [str(n) for n in block_names]
        if block_names is not None
        else default_names or [f"S{i + 1}" for i in range(len(arrays))]
    )
    if len(names) != len(arrays):
        raise HABITAPIError(
            "plot_prototype_matching: block_names must have one name per block."
        )
    axis_names = (
        [str(n) for n in feature_names]
        if feature_names is not None
        else list(default_features or [f"feature {i}" for i in range(n_features)])
    )

    def xy(rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project rows onto the plotted plane."""
        ys = rows[:, y_col] if y_col is not None else np.zeros(rows.shape[0])
        return rows[:, x_col], ys

    n_prototypes = int(prototypes.shape[0])
    colours = _habitat_color_lookup(list(range(1, n_prototypes + 1)))
    draw_points = metric in ("raw", "sqeuclidean", "manhattan")
    fig, axes = _new_axes(ax, height_mm=80.0)
    with use_style("radiology"):
        proto_x, proto_y = xy(prototypes)
        if metric == "cosine":
            # A ray per prototype direction, long enough to cross the data.
            reach = 1.1 * max(
                float(np.max(np.hypot(*xy(a)))) if a.size else 0.0 for a in arrays
            )
            for k in range(n_prototypes):
                norm = float(np.hypot(proto_x[k], proto_y[k]))
                if norm <= 0:
                    continue
                axes.plot(
                    [0.0, proto_x[k] / norm * reach],
                    [0.0, proto_y[k] / norm * reach],
                    color=colours[k + 1],
                    linewidth=1.2,
                    linestyle="--",
                    zorder=1,
                )
        for index, (array, assigned) in enumerate(zip(arrays, assignments)):
            marker = _BLOCK_MARKERS[index % len(_BLOCK_MARKERS)]
            xs, ys = xy(array)
            for row, k in enumerate(assigned.tolist()):
                if k < 0:
                    axes.scatter(
                        xs[row], ys[row], marker=marker, s=46, facecolors="none",
                        edgecolors=_UNMATCHED_FACE, linewidths=1.2, zorder=3,
                    )
                    continue
                if show_links and draw_points:
                    axes.plot(
                        [xs[row], proto_x[k]], [ys[row], proto_y[k]],
                        color=colours[k + 1], linewidth=0.8, alpha=0.7, zorder=1,
                    )
                axes.scatter(
                    xs[row], ys[row], marker=marker, s=46, color=colours[k + 1],
                    edgecolors="white", linewidths=0.5, zorder=3,
                )
            # Legend entry per subject: marker shape only.
            axes.scatter(
                [], [], marker=marker, s=40, facecolors="none",
                edgecolors="#333333", label=sanitize_label(names[index]),
            )
        if draw_points:
            # Below the habitat points (which are smaller), so a prototype
            # holding a single habitat does not hide that habitat.
            axes.scatter(
                proto_x, proto_y, marker="X", s=150, color="black",
                edgecolors="white", linewidths=0.8, zorder=2,
                label=sanitize_label("prototype"),
            )
            for k in range(n_prototypes):
                axes.annotate(
                    sanitize_label(f"P{k + 1}"),
                    (proto_x[k], proto_y[k]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontweight="bold",
                )
        if any(np.any(a < 0) for a in assignments):
            axes.scatter(
                [], [], marker="o", s=40, facecolors="none",
                edgecolors=_UNMATCHED_FACE, label=sanitize_label("unnamed"),
            )
        axes.set_xlabel(sanitize_label(axis_names[x_col]))
        axes.set_ylabel(
            sanitize_label(axis_names[y_col]) if y_col is not None else ""
        )
        axes.grid(True, alpha=0.25, linewidth=0.6)
        axes.set_axisbelow(True)
        axes.legend(loc="best", fontsize=7, frameon=False)
        default_title = {
            "raw": "Habitats named by shared prototypes",
            "sqeuclidean": "Prototype matching (sqeuclidean)",
            "manhattan": "Prototype matching (manhattan)",
            "cosine": "Prototype matching (cosine: rays = directions)",
            "correlation": "Prototype matching (correlation: colours only)",
        }[metric]
        axes.set_title(sanitize_label(title if title is not None else default_title))
    return fig
