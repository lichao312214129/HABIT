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
"""Core habitat-analysis figures: validation curves, map features, compares.

Pure matplotlib helpers for the habitat product spine (not general ML plots):

* auto-K / cluster-validation curves from a ``selection_report``
* volume fractions, MSI heatmaps, ITH summary
* train vs predict (or any two) label-map compare
* optional two-step supervoxel | habitat triptych

Arrays / mappings in → ``Figure`` out. No filesystem, no ``show``.
All axis text is ASCII via :func:`~habit.viz.labels.sanitize_label`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from habit.exceptions import HABITAPIError
from habit.utils.optional_deps import require
from habit.viz.habitat_overlay import (
    _as_volume,
    _direction_matrix,
    _habitat_color_list,
    _habitat_color_lookup,
    _imshow_physical_extent,
    _positive_habitat_ids,
    _prepare_overlay_slice,
    _slice_index,
    _spacing_xyz,
)
from habit.viz.colorbar import (
    ColorbarSpec,
    DEFAULT_HABITAT_CBAR_LABEL,
    add_discrete_habitat_colorbar,
    add_image_colorbar_from_spec,
)
from habit.viz.labels import sanitize_label
from habit.viz.orientation import (
    DEFAULT_DISPLAY_CONVENTION,
    DisplayConvention,
    normalize_display_convention,
    resolve_display_geometry,
)
from habit.viz.style import use_style

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "plot_cluster_validation_curves",
    "plot_cluster_validation_from_report",
    "plot_habitat_volume_fractions",
    "plot_msi_matrix",
    "plot_ith_summary",
    "plot_habitat_label_compare",
    "plot_partition_triptych",
]

_VIZ_PURPOSE = "habitat core analysis figures"

#: Perceptually uniform sequential map (greyscale-safe; no rainbow).
_MSI_CMAP = "cividis"

#: Light grey for masked MSI diagonal cells (connected-subregion size, not
#: an interface count). Must stay distinct from the sequential map.
_MSI_DIAGONAL_FACE = "#D0D0D0"

#: Accepted ``plot_msi_matrix(..., scale=)`` keys.
_MSI_SCALES = ("linear", "log1p", "normalized", "raw")

#: Default bar width as a fraction of the category slot (thinner than
#: matplotlib's 0.8 so a few habitats do not look like a solid block).
_BAR_WIDTH = 0.55

#: ITH summary bars: slimmer than ``_BAR_WIDTH`` / matplotlib 0.8. A single
#: fat column looks clumsy when only 1–3 habitats are present.
_ITH_BAR_WIDTH = 0.36


def _resolved_bar_width(
    n_categories: int,
    bar_width: Optional[float] = None,
    *,
    default: float = _ITH_BAR_WIDTH,
) -> float:
    """
    Return a bar width in category-slot units.

    An explicit ``bar_width`` wins. Otherwise the default is thinned further
    when few categories would otherwise produce oversized columns.

    Args:
        n_categories: Number of bars (treated as at least 1).
        bar_width: Optional override; must be finite and ``> 0``.
        default: Width used when many categories are present.

    Returns:
        Width in data units (one unit = one category slot).

    Raises:
        HABITAPIError: If ``bar_width`` is non-finite or ``<= 0``.
    """
    if bar_width is not None:
        width = float(bar_width)
        if not np.isfinite(width) or width <= 0.0:
            raise HABITAPIError(
                f"bar_width must be a finite value > 0; got {bar_width!r}."
            )
        return width
    n = max(int(n_categories), 1)
    # Keep 1–3 habitats as slim columns; approach ``default`` as n grows.
    if n <= 1:
        return min(default, 0.22)
    if n == 2:
        return min(default, 0.26)
    if n == 3:
        return min(default, 0.28)
    return float(default)


def _set_category_xlim(
    ax: Any,
    n_categories: int,
    bar_width: float,
) -> None:
    """
    Pad the x-axis so a handful of bars do not fill the panel.

    Matplotlib autoscales xlim to the bar patches, so a single bar of any
    width still looks fat. Add gutters (and extra empty slots when ``n`` is
    small) and centre the real categories at integer x = 0 .. n-1.

    Args:
        ax: Matplotlib axes that already hold the bars.
        n_categories: Number of bars.
        bar_width: Width used for those bars (data units).
    """
    n = max(int(n_categories), 1)
    # Half-bar plus a fixed gutter, then extra empty slots for 1–3 categories
    # so the ink occupies ~8–12% of the axes instead of ~90%.
    base_pad = 0.5 * float(bar_width) + 0.50
    extra_slots = max(0.0, 1.05 - 0.28 * float(n))
    pad = base_pad + extra_slots
    ax.set_xlim(-pad, float(n - 1) + pad)


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


def _palette_color(index: int, palette: Sequence[str]) -> str:
    """
    Return a colour from a style palette, cycling if needed.

    Args:
        index: Zero-based colour index.
        palette: Hex colour cycle (Okabe–Ito when using ``use_style``).

    Returns:
        str: A ``#RRGGBB`` colour string.
    """
    if not palette:
        return "#0072B2"
    return str(palette[int(index) % len(palette)])


def _msi_robust_limits(values: np.ndarray) -> Tuple[float, float]:
    """
    Return ``(vmin, vmax)`` from the 2nd–98th percentiles of finite cells.

    Falls back to the true min/max when there are too few cells or the
    percentiles collapse. A constant sample returns equal limits (honest:
    no fake stretch). Callers must put these same numbers on the colorbar.

    Args:
        values: Samples already filtered to the cells that should set the
            colour scale (typically finite off-diagonal entries).

    Returns:
        Tuple of ``(vmin, vmax)``. ``vmax >= vmin``; both finite.
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    if finite.size >= 4:
        vmin = float(np.percentile(finite, 2.0))
        vmax = float(np.percentile(finite, 98.0))
    else:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax < vmin:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    if vmax < vmin:
        vmax = vmin
    return vmin, vmax


def _msi_extend_mode(
    values: np.ndarray,
    vmin: float,
    vmax: float,
) -> str:
    """
    Return a colorbar ``extend`` mode for cells outside ``[vmin, vmax]``.

    Args:
        values: Finite samples that were coloured (NaN/masked excluded).
        vmin: Lower colour limit shown on the colorbar.
        vmax: Upper colour limit shown on the colorbar.

    Returns:
        ``"neither"``, ``"min"``, ``"max"``, or ``"both"``.
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return "neither"
    # Integer counts can sit exactly on a percentile; use a tiny absolute
    # tolerance so extend triangles appear only for true clipping.
    tol = 1e-12 * (1.0 + abs(vmax) + abs(vmin))
    below = bool(np.any(finite < (vmin - tol)))
    above = bool(np.any(finite > (vmax + tol)))
    if below and above:
        return "both"
    if below:
        return "min"
    if above:
        return "max"
    return "neither"


def _msi_normalized_probability(arr: np.ndarray) -> np.ndarray:
    """
    Return ``P = M / D`` with the same ``D`` as the MSI feature kernel.

    Args:
        arr: Square non-negative interaction matrix.

    Returns:
        Probability matrix, all zeros when ``D <= 0``.
    """
    denominator_mat = np.tril(arr, k=0).copy()
    if denominator_mat.shape[0] > 0:
        denominator_mat[0] = 0.0
    denominator = float(denominator_mat.sum())
    if denominator <= 0.0:
        return np.zeros_like(arr)
    return arr / denominator


def _msi_display_matrix(
    matrix: np.ndarray,
    scale: str,
    *,
    mask_diagonal: bool,
) -> Tuple[np.ndarray, str, float, float, str]:
    """
    Convert a raw MSI count matrix into a display array and colour limits.

    Diagonal entries of ``M`` are connected-subregion size (and ``M[0, 0]``
    is the background–background pair count inside the padded bbox). They
    are a different quantity from off-diagonal interface counts and typically
    dominate a linear or log scale, collapsing real but smaller border
    differences to one colour. When ``mask_diagonal`` is true those cells
    are set to NaN so the colour scale is fitted to finite off-diagonal
    values only; the plot still annotates the true diagonal numbers.

    ``log1p`` is retained as an opt-in. On an already-narrow off-diagonal
    range it compresses contrast further; the default is linear.

    Args:
        matrix: Square non-negative interaction matrix.
        scale: ``"linear"``, ``"log1p"``, ``"normalized"``, or ``"raw"``.
        mask_diagonal: If True, diagonal cells are NaN in the display array
            and excluded from ``vmin`` / ``vmax``.

    Returns:
        Tuple of ``(display, colorbar_label, vmin, vmax, extend)``.
        ``vmin`` / ``vmax`` are the limits that must appear on the colorbar.

    Raises:
        HABITAPIError: If ``scale`` is unknown.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    key = str(scale).strip().lower()
    if key == "linear":
        display = np.clip(arr, 0.0, None)
        cbar_label = "Neighbour-pair count"
    elif key == "log1p":
        display = np.log1p(np.clip(arr, 0.0, None))
        cbar_label = "log1p(neighbour-pair count)"
    elif key == "raw":
        display = np.clip(arr, 0.0, None)
        cbar_label = "Neighbour-pair count"
    elif key == "normalized":
        display = _msi_normalized_probability(np.clip(arr, 0.0, None))
        cbar_label = "Normalized interaction P"
    else:
        raise HABITAPIError(
            f"plot_msi_matrix: scale must be {list(_MSI_SCALES)!r}; "
            f"got {scale!r}."
        )

    display = np.array(display, dtype=np.float64, copy=True)
    if mask_diagonal and display.size:
        display[np.eye(display.shape[0], dtype=bool)] = np.nan
        cbar_label = f"{cbar_label} (diagonal masked)"

    finite = display[np.isfinite(display)]
    vmin, vmax = _msi_robust_limits(finite)
    extend = _msi_extend_mode(finite, vmin, vmax)
    return display, cbar_label, vmin, vmax, extend


def plot_cluster_validation_curves(
    scores: Mapping[str, Sequence[float]],
    cluster_range: Sequence[int],
    *,
    selected: Optional[Union[int, Mapping[str, int]]] = None,
    methods: Optional[Sequence[str]] = None,
    directions: Optional[Mapping[str, str]] = None,
    title: Optional[str] = None,
) -> "Figure":
    """
    Plot auto-K / cluster-validation score curves (one panel per method).

    Args:
        scores: Method name → score sequence aligned with ``cluster_range``.
        cluster_range: Candidate habitat counts (x-axis), ascending.
        selected: Global selected ``k``, or per-method best ``k`` to mark.
        methods: Subset / order of methods; default is ``scores`` key order.
        directions: Optional ``method -> {"maximize","minimize","knee"}``;
            used only when ``selected`` is a single int and a mark must be
            recomputed for a method missing from a mapping.
        title: Optional figure title (ASCII-sanitised).

    Returns:
        A matplotlib ``Figure``.

    Raises:
        HABITAPIError: On empty inputs or length mismatches.
    """
    cluster_vals = [int(v) for v in cluster_range]
    if not cluster_vals:
        raise HABITAPIError(
            "plot_cluster_validation_curves: cluster_range must be non-empty."
        )
    method_names = list(methods) if methods is not None else list(scores.keys())
    if not method_names:
        raise HABITAPIError(
            "plot_cluster_validation_curves: no methods to plot."
        )

    n = len(method_names)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    plt = _plt()
    with use_style("radiology") as style:
        width_in, _ = style.figsize(columns=2 if n > 1 else 1)
        height_in = (2.2 * nrows) + 0.6
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(max(width_in, 3.4 * ncols), height_in),
            constrained_layout=True,
            squeeze=False,
        )
        line_color = _palette_color(0, style.palette)
        mark_color = _palette_color(1, style.palette)

        for idx, method in enumerate(method_names):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            if method not in scores:
                raise HABITAPIError(
                    f"plot_cluster_validation_curves: missing scores for {method!r}."
                )
            values = [float(v) for v in scores[method]]
            if len(values) != len(cluster_vals):
                raise HABITAPIError(
                    "plot_cluster_validation_curves: scores length must match "
                    f"cluster_range for {method!r}."
                )
            ax.plot(
                cluster_vals,
                values,
                "o-",
                color=line_color,
                linewidth=style.line_width,
                markersize=4.5,
                markerfacecolor=line_color,
                markeredgecolor=line_color,
            )
            mark_k = _resolve_selected_k(
                method,
                cluster_vals,
                values,
                selected=selected,
                directions=directions,
            )
            if mark_k is not None and mark_k in cluster_vals:
                mark_idx = cluster_vals.index(mark_k)
                ax.plot(
                    mark_k,
                    values[mark_idx],
                    marker="x",
                    color=mark_color,
                    markersize=8,
                    markeredgewidth=1.4,
                    linestyle="none",
                )
                ax.set_title(
                    sanitize_label(f"{method} (selected k={mark_k})")
                )
            else:
                ax.set_title(sanitize_label(str(method)))
            ax.set_xlabel(sanitize_label("Number of habitats (k)"))
            ax.set_ylabel(sanitize_label("Score"))
            ax.set_xticks(cluster_vals)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)

        for idx in range(n, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].axis("off")

        if title is not None:
            fig.suptitle(sanitize_label(title))
        else:
            fig.suptitle(sanitize_label("Cluster validation curves"))
    return fig


def plot_cluster_validation_from_report(
    report: Mapping[str, Any],
    *,
    title: Optional[str] = None,
) -> "Figure":
    """
    Draw validation curves from a habitat-model ``selection_report``.

    Expected keys (see :func:`habit.domain.habitat_model._selection.build_selection_report`):
    ``candidates``, ``scores``, optional ``methods``, ``selected``, ``directions``.

    Args:
        report: Selection report mapping.
        title: Optional figure title.

    Returns:
        A matplotlib ``Figure``.

    Raises:
        HABITAPIError: When required keys are missing.
    """
    if "candidates" not in report or "scores" not in report:
        raise HABITAPIError(
            "plot_cluster_validation_from_report: report needs "
            "'candidates' and 'scores'."
        )
    methods = report.get("methods")
    return plot_cluster_validation_curves(
        report["scores"],
        report["candidates"],
        selected=report.get("selected"),
        methods=methods,
        directions=report.get("directions"),
        title=title,
    )


def _resolve_selected_k(
    method: str,
    cluster_vals: Sequence[int],
    values: Sequence[float],
    *,
    selected: Optional[Union[int, Mapping[str, int]]],
    directions: Optional[Mapping[str, str]],
) -> Optional[int]:
    """
    Pick the k to mark on one validation panel.

    ``directions`` / ``values`` are reserved for callers that later want
    per-method recomputation; today a global ``selected`` int is marked on
    every panel for product clarity.
    """
    _ = (cluster_vals, values, directions)
    if selected is None:
        return None
    if isinstance(selected, Mapping):
        if method in selected:
            return int(selected[method])
        return None
    return int(selected)


def plot_habitat_volume_fractions(
    fractions: Mapping[int, float],
    *,
    title: Optional[str] = None,
) -> "Figure":
    """
    Bar chart of per-habitat volume fractions (of non-background VOI).

    Args:
        fractions: Habitat id → fraction in ``[0, 1]``.
        title: Optional figure title.

    Returns:
        A matplotlib ``Figure``.
    """
    if not fractions:
        raise HABITAPIError(
            "plot_habitat_volume_fractions: fractions mapping is empty."
        )
    ids = sorted(int(k) for k in fractions.keys())
    vals = [float(fractions[k]) for k in ids]
    plt = _plt()
    with use_style("radiology") as style:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=style.figsize(columns=1, height_mm=62.0),
            constrained_layout=True,
        )
        x = np.arange(len(ids))
        ax.bar(
            x,
            vals,
            width=_BAR_WIDTH,
            color=_palette_color(0, style.palette),
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([sanitize_label(f"H{i}") for i in ids])
        ax.set_ylabel(sanitize_label("Volume fraction"))
        ax.set_ylim(0.0, max(1.0, max(vals) * 1.15 if vals else 1.0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(
            sanitize_label(
                title if title is not None else "Habitat volume fractions"
            )
        )
    return fig


def plot_msi_matrix(
    matrix: np.ndarray,
    *,
    habitat_ids: Optional[Sequence[int]] = None,
    title: Optional[str] = None,
    scale: str = "linear",
    mask_diagonal: Optional[bool] = None,
    colorbar: ColorbarSpec = True,
) -> "Figure":
    """
    Heatmap of a spatial interaction (MSI) matrix.

    Row/column 0 is background. Diagonal entries are connected-subregion
    size (``M[0, 0]`` is background–background pairs in the padded bbox and
    is not exported as an MSI feature). Off-diagonal entries are interface
    counts. A shared linear colour scale over the full matrix is dominated
    by the diagonal, so the default ``scale='linear'`` masks the diagonal
    and stretches colour over the 2nd–98th percentiles of finite
    off-diagonal cells. The colorbar shows those same numeric limits.
    ``log1p`` is opt-in: on a nearly-constant off-diagonal block it
    flattens contrast further.

    Args:
        matrix: Square 2D array (row/column 0 is typically background).
        habitat_ids: Optional labels for axes ``1..K`` (background stays ``BG``).
        title: Optional figure title.
        scale: ``"linear"`` (default), ``"normalized"``, ``"raw"``, or
            ``"log1p"``. ``"normalized"`` is :math:`P=M/D` with the same
            :math:`D` as :func:`habit.kernels.habitat_metrics.msi_features_from_matrix`.
        mask_diagonal: Mask the main diagonal in the colour scale. Default
            is True for ``linear`` / ``log1p`` / ``normalized``, False for
            ``raw`` (full-matrix linear stretch).
        colorbar: Draw a short vertical colorbar (default ``True``). Pass
            ``False`` to hide it, or a mapping of style kwargs
            (``shrink``, ``pad``, ``fraction``, ``aspect``, ``ticks``,
            ``label``, ...) to override the default.

    Returns:
        A matplotlib ``Figure``.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise HABITAPIError(
            "plot_msi_matrix: matrix must be square 2D; "
            f"got shape {arr.shape}."
        )
    n = int(arr.shape[0])
    if habitat_ids is None:
        tick_labels = ["BG"] + [f"H{i}" for i in range(1, n)]
    else:
        ids = [int(v) for v in habitat_ids]
        if len(ids) != n - 1:
            raise HABITAPIError(
                "plot_msi_matrix: habitat_ids length must be matrix size - 1 "
                f"(background row); got {len(ids)} for size {n}."
            )
        tick_labels = ["BG"] + [f"H{i}" for i in ids]

    scale_key = str(scale).strip().lower()
    if mask_diagonal is None:
        hide_diag = scale_key != "raw"
    else:
        hide_diag = bool(mask_diagonal)

    plt = _plt()
    display, cbar_label, vmin, vmax, extend = _msi_display_matrix(
        arr, scale, mask_diagonal=hide_diag
    )
    # imshow requires vmax > vmin; a one-count bump is only for the
    # degenerate constant-matrix case and is not shown as a fake range
    # when we override colorbar ticks below.
    plot_vmax = vmax if vmax > vmin else vmin + 1.0
    annot = (
        _msi_normalized_probability(np.clip(arr, 0.0, None))
        if scale_key == "normalized"
        else arr
    )
    with use_style("radiology") as style:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=style.figsize(columns=1, height_mm=78.0),
            constrained_layout=True,
        )
        cmap = plt.get_cmap(_MSI_CMAP).copy()
        cmap.set_bad(_MSI_DIAGONAL_FACE)
        im = ax.imshow(
            np.ma.masked_invalid(display),
            cmap=cmap,
            interpolation="nearest",
            vmin=vmin,
            vmax=plot_vmax,
        )
        cbar = add_image_colorbar_from_spec(
            im, colorbar, ax=ax, label=cbar_label, extend=extend
        )
        if cbar is not None and vmax <= vmin:
            cbar.set_ticks([vmin])
            cbar.set_ticklabels([f"{vmin:g}"])
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(
            [sanitize_label(t) for t in tick_labels], rotation=45, ha="right"
        )
        ax.set_yticklabels([sanitize_label(t) for t in tick_labels])
        ax.set_xlabel(sanitize_label("Habitat j"))
        ax.set_ylabel(sanitize_label("Habitat i"))
        ax.set_title(
            sanitize_label(
                title if title is not None else "Spatial interaction (MSI)"
            )
        )
        # Annotate small matrices so a reviewer can read the cells,
        # including masked diagonal counts.
        if n <= 8:
            span = plot_vmax - vmin
            for i in range(n):
                for j in range(n):
                    cell = float(display[i, j])
                    if np.isfinite(cell) and span > 0.0:
                        norm = (cell - vmin) / span
                        text_color = "white" if norm >= 0.55 else "#222222"
                    else:
                        text_color = "#222222"
                    value = float(annot[i, j])
                    if scale_key == "normalized":
                        cell_label = f"{value:.2f}"
                    else:
                        cell_label = f"{value:.0f}"
                    ax.text(
                        j,
                        i,
                        cell_label,
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=max(style.font_size - 1.5, 5.0),
                    )
    return fig


def plot_ith_summary(
    ith: float,
    *,
    per_habitat: Optional[Mapping[int, Tuple[int, int]]] = None,
    title: Optional[str] = None,
    bar_width: Optional[float] = None,
) -> "Figure":
    """
    Summarise ITH score, optionally with per-habitat region counts.

    Bars are thinner than matplotlib's default and the x-axis is padded when
    few habitats are present, so 1–3 columns stay slim instead of filling
    the panel.

    Args:
        ith: Scalar ITH in ``[0, 1)``.
        per_habitat: Optional habitat id → ``(num_regions, largest_size)``.
        title: Optional figure title.
        bar_width: Optional bar width in category-slot units. When omitted,
            a slim default is used and further reduced for 1–3 categories.

    Returns:
        A matplotlib ``Figure``.
    """
    score = float(ith)
    if not np.isfinite(score):
        raise HABITAPIError("plot_ith_summary: ith must be finite.")
    plt = _plt()
    with use_style("radiology") as style:
        bar_ith = _palette_color(3, style.palette)  # reddish purple
        bar_frag = _palette_color(2, style.palette)  # bluish green
        if per_habitat:
            ids = sorted(int(k) for k in per_habitat.keys())
            counts = [int(per_habitat[k][0]) for k in ids]
            fig, axes = plt.subplots(
                1,
                2,
                figsize=style.figsize(columns=2, height_mm=62.0),
                constrained_layout=True,
            )
            ith_width = _resolved_bar_width(1, bar_width)
            axes[0].bar(
                [0],
                [score],
                color=bar_ith,
                edgecolor="white",
                width=ith_width,
                linewidth=0.4,
            )
            _set_category_xlim(axes[0], 1, ith_width)
            axes[0].set_ylim(0.0, max(1.0, score * 1.15))
            axes[0].set_xticks([0])
            axes[0].set_xticklabels([sanitize_label("ITH")])
            axes[0].set_ylabel(sanitize_label("Score"))
            axes[0].set_title(sanitize_label(f"ITH = {score:.4f}"))
            axes[0].spines["top"].set_visible(False)
            axes[0].spines["right"].set_visible(False)
            axes[0].grid(True, axis="y", alpha=0.25, linewidth=0.6)
            axes[0].set_axisbelow(True)
            n_hab = len(ids)
            frag_width = _resolved_bar_width(n_hab, bar_width)
            x = np.arange(n_hab)
            axes[1].bar(
                x,
                counts,
                width=frag_width,
                color=bar_frag,
                edgecolor="white",
                linewidth=0.4,
            )
            _set_category_xlim(axes[1], n_hab, frag_width)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([sanitize_label(f"H{i}") for i in ids])
            axes[1].set_ylabel(sanitize_label("Connected regions"))
            axes[1].set_title(sanitize_label("Fragmentation by habitat"))
            axes[1].spines["top"].set_visible(False)
            axes[1].spines["right"].set_visible(False)
            axes[1].grid(True, axis="y", alpha=0.25, linewidth=0.6)
            axes[1].set_axisbelow(True)
        else:
            fig, ax = plt.subplots(
                1,
                1,
                figsize=style.figsize(columns=1, height_mm=58.0),
                constrained_layout=True,
            )
            ith_width = _resolved_bar_width(1, bar_width)
            ax.bar(
                [0],
                [score],
                color=bar_ith,
                edgecolor="white",
                width=ith_width,
                linewidth=0.4,
            )
            _set_category_xlim(ax, 1, ith_width)
            ax.set_ylim(0.0, max(1.0, score * 1.15))
            ax.set_xticks([0])
            ax.set_xticklabels([sanitize_label("ITH")])
            ax.set_ylabel(sanitize_label("Score"))
            ax.set_title(sanitize_label(f"ITH = {score:.4f}"))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
            ax.set_axisbelow(True)
        if title is not None:
            fig.suptitle(sanitize_label(title))
    return fig


def plot_habitat_label_compare(
    image: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    *,
    titles: Tuple[str, str] = ("Reference", "Predict"),
    alpha: float = 1.0,
    axis: int = 0,
    index: Optional[int] = None,
    direction: Optional[Sequence[float]] = None,
    spacing: Optional[Sequence[float]] = None,
    display_convention: DisplayConvention = DEFAULT_DISPLAY_CONVENTION,
    show_disagreement: bool = True,
    colorbar: ColorbarSpec = True,
    colorbar_label: str = DEFAULT_HABITAT_CBAR_LABEL,
) -> "Figure":
    """
    Side-by-side habitat overlays, optional disagreement mask.

    Args:
        image: Anatomy volume ``(z, y, x)`` or 2D.
        labels_a: Reference / train labels (same shape).
        labels_b: Compared / predicted labels (same shape).
        titles: Panel titles for A and B.
        alpha: Overlay opacity (default ``1.0`` = opaque habitat colours).
        axis: Slice axis for 3D volumes.
        index: Slice index; densest union of labels when omitted.
        direction: Optional SimpleITK direction cosines.
        spacing: Optional SimpleITK spacing ``(x, y[, z])``.
        display_convention: Radiological / neurological / native.
        show_disagreement: If True, add a third panel for label mismatch.
        colorbar: Discrete habitat-ID colorbar on the last habitat panel
            (default ``True``). The disagreement panel is not a habitat
            map and does not get this bar. Pass ``False`` to hide it.
        colorbar_label: Colorbar label (English default ``\"Habitat\"``).

    Returns:
        A matplotlib ``Figure``.
    """
    if not (0.0 < float(alpha) <= 1.0):
        raise HABITAPIError(
            f"plot_habitat_label_compare: alpha must be in (0, 1]; got {alpha!r}."
        )
    try:
        convention = normalize_display_convention(display_convention)
    except HABITAPIError as exc:
        raise HABITAPIError(f"plot_habitat_label_compare: {exc}") from exc

    image_vol = _as_volume(image, "image")
    a = np.asarray(_as_volume(labels_a, "labels_a"), dtype=np.int32)
    b = np.asarray(_as_volume(labels_b, "labels_b"), dtype=np.int32)
    if image_vol.shape != a.shape or image_vol.shape != b.shape:
        raise HABITAPIError(
            "plot_habitat_label_compare: image/labels shapes must match; "
            f"got {image_vol.shape}, {a.shape}, {b.shape}."
        )

    axis_id = 0 if image_vol.ndim == 2 else int(axis)
    union = np.where((a > 0) | (b > 0), 1, 0).astype(np.int32)
    slice_index = _slice_index(union, axis_id, index)
    resolved_direction, resolved_spacing = resolve_display_geometry(
        image, labels_a, labels_b, direction=direction, spacing=spacing
    )
    direction_matrix = _direction_matrix(resolved_direction, ndim=image_vol.ndim)
    spacing_xyz = _spacing_xyz(resolved_spacing, ndim=image_vol.ndim)
    # Union of A/B IDs so both habitat panels and the shared colorbar match.
    habitat_ids = _positive_habitat_ids(np.concatenate([a.ravel(), b.ravel()]))
    id_to_color = _habitat_color_lookup(habitat_ids)

    panels = 3 if show_disagreement else 2
    plt = _plt()
    with use_style("radiology") as style:
        width_in, height_in = style.figsize(columns=2, height_mm=72.0)
        fig, axes = plt.subplots(
            1,
            panels,
            figsize=(width_in, height_in),
            constrained_layout=True,
        )
        if panels == 2:
            axes = [axes[0], axes[1]]

        for ax, labs, panel_title in zip(
            axes[:2], (a, b), (titles[0], titles[1])
        ):
            rgb, _labs = _prepare_overlay_slice(
                image_vol,
                labs,
                axis_id=axis_id,
                slice_index=slice_index,
                alpha=float(alpha),
                direction=direction_matrix,
                convention=convention,
                id_to_color=id_to_color,
            )
            extent = _imshow_physical_extent(
                (int(rgb.shape[0]), int(rgb.shape[1])),
                spacing_xyz,
                slice_axis=axis_id,
                ndim=image_vol.ndim,
                direction=direction_matrix,
                convention=convention,
            )
            ax.imshow(
                rgb,
                interpolation="nearest",
                origin="upper",
                extent=extent,
                aspect="equal",
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(sanitize_label(panel_title))
            ax.axis("off")

        if show_disagreement:
            disagree = ((a != b) & ((a > 0) | (b > 0))).astype(np.int32)
            rgb, _labs = _prepare_overlay_slice(
                image_vol,
                disagree,
                axis_id=axis_id,
                slice_index=slice_index,
                alpha=float(alpha),
                direction=direction_matrix,
                convention=convention,
            )
            extent = _imshow_physical_extent(
                (int(rgb.shape[0]), int(rgb.shape[1])),
                spacing_xyz,
                slice_axis=axis_id,
                ndim=image_vol.ndim,
                direction=direction_matrix,
                convention=convention,
            )
            axes[2].imshow(
                rgb,
                interpolation="nearest",
                origin="upper",
                extent=extent,
                aspect="equal",
            )
            axes[2].set_aspect("equal", adjustable="box")
            axes[2].set_title(sanitize_label("Disagreement"))
            axes[2].axis("off")

        add_discrete_habitat_colorbar(
            axes[1],
            habitat_ids,
            _habitat_color_list(habitat_ids),
            colorbar=colorbar,
            label=colorbar_label,
        )
        fig.suptitle(sanitize_label("Habitat label compare"))
    return fig


def plot_partition_triptych(
    image: np.ndarray,
    supervoxel_labels: np.ndarray,
    habitat_labels: np.ndarray,
    *,
    titles: Tuple[str, str, str] = (
        "Anatomy",
        "Supervoxels",
        "Habitats",
    ),
    alpha: float = 1.0,
    axis: int = 0,
    index: Optional[int] = None,
    direction: Optional[Sequence[float]] = None,
    spacing: Optional[Sequence[float]] = None,
    display_convention: DisplayConvention = DEFAULT_DISPLAY_CONVENTION,
    colorbar: ColorbarSpec = True,
    colorbar_label: str = DEFAULT_HABITAT_CBAR_LABEL,
) -> "Figure":
    """
    Two-step partition view: greyscale | supervoxel overlay | habitat overlay.

    Args:
        image: Anatomy volume.
        supervoxel_labels: Integer supervoxel map (0 = background).
        habitat_labels: Integer habitat map (0 = background).
        titles: Three panel titles.
        alpha: Overlay opacity for label panels (default ``1.0`` = opaque).
        axis: Slice axis for 3D volumes.
        index: Slice index; densest habitat slice when omitted.
        direction: Optional SimpleITK direction cosines.
        spacing: Optional SimpleITK spacing.
        display_convention: Display convention for flips.
        colorbar: Discrete habitat-ID colorbar on the habitat panel only
            (default ``True``). Anatomy and supervoxel panels are not
            habitat maps. Pass ``False`` to hide it.
        colorbar_label: Colorbar label (English default ``\"Habitat\"``).

    Returns:
        A matplotlib ``Figure``.
    """
    from habit.viz.habitat_overlay import (
        _normalize_grey,
        _orient_slice_for_display,
        _take_slice,
    )

    if not (0.0 < float(alpha) <= 1.0):
        raise HABITAPIError(
            f"plot_partition_triptych: alpha must be in (0, 1]; got {alpha!r}."
        )
    try:
        convention = normalize_display_convention(display_convention)
    except HABITAPIError as exc:
        raise HABITAPIError(f"plot_partition_triptych: {exc}") from exc

    image_vol = _as_volume(image, "image")
    sv = np.asarray(_as_volume(supervoxel_labels, "supervoxel_labels"), dtype=np.int32)
    hab = np.asarray(_as_volume(habitat_labels, "habitat_labels"), dtype=np.int32)
    if image_vol.shape != sv.shape or image_vol.shape != hab.shape:
        raise HABITAPIError(
            "plot_partition_triptych: shapes must match; "
            f"got {image_vol.shape}, {sv.shape}, {hab.shape}."
        )

    axis_id = 0 if image_vol.ndim == 2 else int(axis)
    slice_index = _slice_index(hab, axis_id, index)
    resolved_direction, resolved_spacing = resolve_display_geometry(
        image,
        supervoxel_labels,
        habitat_labels,
        direction=direction,
        spacing=spacing,
    )
    direction_matrix = _direction_matrix(resolved_direction, ndim=image_vol.ndim)
    spacing_xyz = _spacing_xyz(resolved_spacing, ndim=image_vol.ndim)

    grey = _normalize_grey(_take_slice(image_vol, axis_id, slice_index))
    grey = _orient_slice_for_display(
        grey, slice_axis=axis_id, direction=direction_matrix, convention=convention
    )
    # Fake RGB greyscale for consistent imshow path.
    anatomy_rgb = np.stack([grey, grey, grey], axis=-1)

    habitat_ids = _positive_habitat_ids(hab)
    hab_id_to_color = _habitat_color_lookup(habitat_ids)
    sv_rgb, _sv_labs = _prepare_overlay_slice(
        image_vol,
        sv,
        axis_id=axis_id,
        slice_index=slice_index,
        alpha=float(alpha),
        direction=direction_matrix,
        convention=convention,
    )
    hab_rgb, _hab_labs = _prepare_overlay_slice(
        image_vol,
        hab,
        axis_id=axis_id,
        slice_index=slice_index,
        alpha=float(alpha),
        direction=direction_matrix,
        convention=convention,
        id_to_color=hab_id_to_color,
    )

    plt = _plt()
    with use_style("radiology") as style:
        fig, axes = plt.subplots(
            1,
            3,
            figsize=style.figsize(columns=2, height_mm=62.0),
            constrained_layout=True,
        )
        for ax, rgb, panel_title in zip(
            axes, (anatomy_rgb, sv_rgb, hab_rgb), titles
        ):
            extent = _imshow_physical_extent(
                (int(rgb.shape[0]), int(rgb.shape[1])),
                spacing_xyz,
                slice_axis=axis_id,
                ndim=image_vol.ndim,
                direction=direction_matrix,
                convention=convention,
            )
            ax.imshow(
                rgb,
                interpolation="nearest",
                origin="upper",
                extent=extent,
                aspect="equal",
            )
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(sanitize_label(panel_title))
            ax.axis("off")
        add_discrete_habitat_colorbar(
            axes[2],
            habitat_ids,
            _habitat_color_list(habitat_ids),
            colorbar=colorbar,
            label=colorbar_label,
        )
        fig.suptitle(sanitize_label("Two-step partitions"))
    return fig
