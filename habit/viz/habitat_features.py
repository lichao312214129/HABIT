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
"""Publication figures for between-habitat feature contrasts.

High-dimensional texture tables are *not* drawn as one violin per feature.
The default story is:

* heatmap -- habitats x features (z-scored), cohort mean or one subject;
* effect-size forest -- top-k paired Cliff's delta (or Cohen's d);
* violin / grouped bar -- only the selected (or top-k) features.

Bar panels are **faceted by feature** so incommensurable scales (Energy vs
``volume_fraction``) never share one linear y-axis. Arrays / panel objects
in, ``Figure`` out. No filesystem. Axis text is ASCII via
:func:`~habit.viz.labels.sanitize_label`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from habit.exceptions import HABITAPIError
from habit.utils.optional_deps import require
from habit.viz.labels import sanitize_label
from habit.viz.style import use_style

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from habit.domain.habitat_features.compare import (
        HabitatFeatureComparison,
        HabitatFeaturePanel,
    )

__all__ = [
    "plot_habitat_feature_heatmap",
    "plot_habitat_feature_effect",
    "plot_habitat_feature_violin",
    "plot_habitat_feature_bars",
]

_VIZ_PURPOSE = "habitat feature contrast figures"

#: Default cap so a 200-feature heatmap stays readable in one column.
_DEFAULT_HEATMAP_FEATURES = 40
#: Violins / bars are for a shortlist, not the full texture bank.
_DEFAULT_DETAIL_FEATURES = 6
_DEFAULT_EFFECT_TOP_K = 20
#: Violin KDE is misleading below this per-habitat point count.
_MIN_VIOLIN_N = 5

# GitHub Pages / gallery readability (same intent as the graph-viz pass).
_TITLE_FONTSIZE = 11.0
_LABEL_FONTSIZE = 10.0
_TICK_FONTSIZE = 9.0
_LEGEND_FONTSIZE = 9.0
_PANEL_TITLE_FONTSIZE = 10.0
_CBAR_FONTSIZE = 9.0


def _plt():
    """Return pyplot with a headless Agg canvas."""
    matplotlib = require("matplotlib", extra="viz", purpose=_VIZ_PURPOSE)
    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg", force=True)
    return require("matplotlib.pyplot", extra="viz", purpose=_VIZ_PURPOSE)


def _ascii_minus_on_ticks(fig: "Figure") -> None:
    """
    Persist ASCII '-' on numeric axes (including colorbars).

    ``use_style`` restores rcParams on exit, so a later ``get_ticklabels``
    would otherwise regenerate U+2212. A FuncFormatter stays on the axes.
    Categorical ticks (``H1``, feature names) keep their FixedFormatter.
    """
    matplotlib = require("matplotlib", extra="viz", purpose=_VIZ_PURPOSE)
    ticker = matplotlib.ticker
    formatter = ticker.FuncFormatter(
        lambda value, _pos: f"{value:g}".replace("\u2212", "-")
    )
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            current = axis.get_major_formatter()
            if isinstance(current, ticker.ScalarFormatter):
                axis.set_major_formatter(formatter)


def _apply_readable_fonts(ax: "Axes") -> None:
    """Enlarge title / axis / tick text for GitHub Pages thumbnails."""
    ax.tick_params(axis="both", labelsize=_TICK_FONTSIZE)
    ax.xaxis.label.set_size(_LABEL_FONTSIZE)
    ax.yaxis.label.set_size(_LABEL_FONTSIZE)
    ax.title.set_size(_PANEL_TITLE_FONTSIZE)


def _as_panel(data: Union["HabitatFeaturePanel", "HabitatFeatureComparison"]):
    """Accept a panel or a comparison that wraps one."""
    panel = getattr(data, "panel", data)
    if getattr(panel, "frame", None) is None:
        raise HABITAPIError(
            "habitat feature plots need a HabitatFeaturePanel or "
            "HabitatFeatureComparison."
        )
    return panel


def _matrix_habitats_by_features(
    panel: "HabitatFeaturePanel",
    *,
    subject_id: Optional[str],
    features: Sequence[str],
) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    Build a habitats x features matrix (mean across subjects unless one id).

    Returns:
        ``(matrix, habitat_ids, feature_names)``.
    """
    frame = panel.frame
    if subject_id is not None:
        frame = panel.for_subject(subject_id).frame
    wanted = [str(name) for name in features]
    frame = frame[frame[panel.feature_column].astype(str).isin(wanted)]
    if frame.empty:
        raise HABITAPIError(
            "plot_habitat_feature_heatmap: no rows for the requested "
            "features / subject."
        )
    pivot = frame.pivot_table(
        index=panel.habitat_column,
        columns=panel.feature_column,
        values=panel.value_column,
        aggfunc="mean",
    )
    habitat_ids = [int(h) for h in pivot.index.tolist()]
    # Keep the caller-specified feature order; drop missing columns.
    present = [name for name in wanted if name in pivot.columns]
    if not present:
        raise HABITAPIError(
            "plot_habitat_feature_heatmap: requested features are absent "
            "from the panel."
        )
    matrix = pivot[present].to_numpy(dtype=np.float64)
    return matrix, habitat_ids, present


def _zscore_columns(matrix: np.ndarray) -> np.ndarray:
    """Z-score each feature (column) across habitats; NaN-safe."""
    out = np.asarray(matrix, dtype=np.float64).copy()
    for col in range(out.shape[1]):
        values = out[:, col]
        finite = np.isfinite(values)
        if int(finite.sum()) < 2:
            out[:, col] = np.nan
            continue
        mu = float(np.mean(values[finite]))
        sd = float(np.std(values[finite], ddof=0))
        if sd == 0.0:
            out[:, col] = 0.0
        else:
            scaled = (values - mu) / sd
            scaled[~finite] = np.nan
            out[:, col] = scaled
    return out


def _readable_feature_label(
    name: str,
    *,
    max_line_len: int = 32,
    max_lines: int = 2,
) -> str:
    """
    ASCII-sanitise a radiomics name and wrap so statistics stay visible.

    Truncating ``original_firstorder_Mean_of_T2`` at 32 characters hid
    Mean / Median / Energy / Kurtosis. Underscore wrap keeps the tail
    (the statistic and optional ``of_<modality>``) on the last line.

    Args:
        name: Raw feature column / panel name.
        max_line_len: Soft width per line (characters).
        max_lines: Maximum wrapped lines (the statistic line is always kept).

    Returns:
        ASCII label, possibly containing ``\\n``.
    """
    label = sanitize_label(str(name))
    parts = [token for token in label.split("_") if token]
    # Radiomics-style ``..._Mean_of_T2``: always put the statistic on its
    # own line so Mean / Median / Energy / Kurtosis stay readable.
    if len(parts) >= 3 and parts[-2].lower() == "of":
        tail = "_".join(parts[-3:])
        head = "_".join(parts[:-3])
        if head:
            if len(head) > max_line_len and max_lines > 2:
                head = _pack_underscore_tokens(
                    parts[:-3], max_line_len=max_line_len, max_lines=max_lines - 1
                )
            return f"{head}\n{tail}"
        return tail
    if len(label) <= max_line_len:
        return label
    if len(parts) < 2:
        # No underscores: hard-wrap and keep the last chunk (distinctive tail).
        chunks = [
            label[index : index + max_line_len]
            for index in range(0, len(label), max_line_len)
        ]
        return "\n".join(chunks[:max_lines])

    tail_n = 1
    tail = "_".join(parts[-tail_n:])
    head_parts = parts[:-tail_n]
    if not head_parts:
        return tail

    head = _pack_underscore_tokens(
        head_parts, max_line_len=max_line_len, max_lines=max(int(max_lines) - 1, 1)
    )
    return f"{head}\n{tail}" if head else tail


def _pack_underscore_tokens(
    tokens: Sequence[str],
    *,
    max_line_len: int,
    max_lines: int,
) -> str:
    """
    Join underscore tokens into at most ``max_lines`` lines.

    Args:
        tokens: Name fragments already split on ``_``.
        max_line_len: Soft character budget per line.
        max_lines: Maximum lines; excess leading tokens are dropped.

    Returns:
        Joined label fragment (may contain ``\\n``).
    """
    if not tokens:
        return ""
    lines: List[str] = []
    current = tokens[0]
    for token in tokens[1:]:
        candidate = f"{current}_{token}"
        if len(candidate) <= max_line_len:
            current = candidate
        else:
            lines.append(current)
            current = token
    lines.append(current)
    budget = max(int(max_lines), 1)
    if len(lines) > budget:
        lines = lines[-budget:]
    return "\n".join(lines)


def _short_feature_label(name: str, max_len: int = 28) -> str:
    """Backward-compatible alias; prefer :func:`_readable_feature_label`."""
    return _readable_feature_label(name, max_line_len=max(int(max_len), 16))


def _select_features_for_overview(
    panel: "HabitatFeaturePanel",
    comparison: Optional["HabitatFeatureComparison"],
    *,
    features: Optional[Sequence[str]],
    max_features: int,
    pair: Optional[Tuple[int, int]],
) -> List[str]:
    """Choose a shortlist: user list, else top-k by absolute effect, else IQR."""
    if features is not None:
        return [str(name) for name in features][: max(int(max_features), 1)]
    if comparison is not None and not comparison.pairwise.empty:
        return list(
            comparison.top_features(int(max_features), pair=pair)
        )
    # Rank by across-habitat IQR of the cohort (or subject) means.
    pivot = panel.frame.pivot_table(
        index=panel.habitat_column,
        columns=panel.feature_column,
        values=panel.value_column,
        aggfunc="mean",
    )
    iqr = (pivot.quantile(0.75) - pivot.quantile(0.25)).sort_values(
        ascending=False
    )
    return [str(name) for name in iqr.index[: max(int(max_features), 1)]]


def _feature_values_for_habitat(
    frame: pd.DataFrame,
    panel: "HabitatFeaturePanel",
    feature_name: str,
    habitat_id: int,
) -> np.ndarray:
    """Return finite values for one feature x habitat as ``float64``."""
    return pd.to_numeric(
        frame.loc[
            (frame[panel.feature_column].astype(str) == str(feature_name))
            & (frame[panel.habitat_column] == int(habitat_id)),
            panel.value_column,
        ],
        errors="coerce",
    ).dropna().to_numpy(dtype=np.float64)


def plot_habitat_feature_heatmap(
    data: Union["HabitatFeaturePanel", "HabitatFeatureComparison"],
    *,
    subject_id: Optional[str] = None,
    features: Optional[Sequence[str]] = None,
    max_features: int = _DEFAULT_HEATMAP_FEATURES,
    zscore: bool = True,
    pair: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
) -> "Figure":
    """
    Habitat x feature heatmap (cohort mean or one subject).

    For tens-to-hundreds of texture features this is the overview figure:
    each column is z-scored across habitats so a family of GLCM features
    does not drown first-order ones. Pass ``features`` or rely on
    ``max_features`` (effect-size rank when a comparison is supplied).

    Args:
        data: Long panel or a :class:`HabitatFeatureComparison`.
        subject_id: If set, that subject's profile. If omitted, the
            cohort mean per habitat x feature.
        features: Optional explicit feature list.
        max_features: Cap when ``features`` is omitted.
        zscore: Z-score each feature across habitats (default True).
        pair: Optional pair used only when ranking by effect size.
        title: Optional figure title.

    Returns:
        The matplotlib ``Figure``.
    """
    panel = _as_panel(data)
    comparison = data if hasattr(data, "pairwise") else None
    if subject_id is not None:
        panel = panel.for_subject(subject_id)
    selected = _select_features_for_overview(
        panel,
        comparison,
        features=features,
        max_features=max_features,
        pair=pair,
    )
    matrix, habitat_ids, names = _matrix_habitats_by_features(
        panel, subject_id=None, features=selected
    )
    shown = _zscore_columns(matrix) if zscore else matrix
    plt = _plt()
    n_feat = max(len(names), 1)
    n_hab = max(len(habitat_ids), 1)
    # Size the axes so each cell is near-square. Do NOT combine
    # aspect="equal" with constrained_layout -- decorations then collapse
    # the image to a postage stamp and leave a tall empty canvas.
    cell_in = 0.58
    left_in, right_in, top_in, bottom_in = 0.72, 0.92, 0.52, 1.45
    fig_w = min(7.4, left_in + right_in + cell_in * n_feat)
    fig_h = min(5.4, top_in + bottom_in + cell_in * n_hab)
    with use_style("radiology") as style:
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), layout=None)
        fig.subplots_adjust(
            left=left_in / fig_w,
            right=1.0 - right_in / fig_w,
            top=1.0 - top_in / fig_h,
            bottom=bottom_in / fig_h,
        )
        finite = shown[np.isfinite(shown)]
        if zscore and finite.size:
            vmax = float(np.nanmax(np.abs(finite)))
            vmax = 1.0 if vmax == 0.0 else vmax
            vmin = -vmax
            cmap = "RdBu_r"
        else:
            vmin = None
            vmax = None
            cmap = "cividis"
        image = ax.imshow(
            shown,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_yticks(np.arange(len(habitat_ids)))
        ax.set_yticklabels(
            [sanitize_label(f"H{hid}") for hid in habitat_ids],
            fontsize=_TICK_FONTSIZE,
        )
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(
            [_readable_feature_label(name) for name in names],
            rotation=45,
            ha="right",
            va="top",
            fontsize=_TICK_FONTSIZE,
        )
        ax.set_xlabel(sanitize_label("Feature"), fontsize=_LABEL_FONTSIZE)
        ax.set_ylabel(sanitize_label("Habitat"), fontsize=_LABEL_FONTSIZE)
        cbar = fig.colorbar(image, ax=ax, fraction=0.08, pad=0.03)
        cbar.set_label(
            sanitize_label("Z-score" if zscore else "Feature value"),
            fontsize=_CBAR_FONTSIZE,
        )
        cbar.ax.tick_params(labelsize=_TICK_FONTSIZE)
        if title is not None:
            resolved = title
        elif subject_id is not None:
            resolved = f"Habitat feature profile ({subject_id})"
        else:
            resolved = "Cohort mean habitat x feature"
        ax.set_title(sanitize_label(resolved), fontsize=_TITLE_FONTSIZE)
        _apply_readable_fonts(ax)
        _ = style
        _ascii_minus_on_ticks(fig)
    return fig


def plot_habitat_feature_effect(
    comparison: "HabitatFeatureComparison",
    *,
    pair: Optional[Tuple[int, int]] = None,
    top_k: int = _DEFAULT_EFFECT_TOP_K,
    title: Optional[str] = None,
) -> "Figure":
    """
    Ranked effect-size forest (Cliff's delta or Cohen's d).

    This is the cohort figure that argues habitats differ: one pair of
    habitats, top-k features, a vertical line at 0. Filled markers are
    BH q < 0.05 when q-values exist; open markers are not significant
    or untested (single-subject / small n). The x-axis is symmetric so
    negative effects keep numeric ticks.

    Args:
        comparison: Output of ``compare_habitat_features``.
        pair: ``(habitat_a, habitat_b)``. Default: the pair with the
            largest mean absolute effect.
        top_k: Maximum features to draw.
        title: Optional figure title.

    Returns:
        The matplotlib ``Figure``.
    """
    frame = comparison.pairwise
    if frame.empty:
        raise HABITAPIError(
            "plot_habitat_feature_effect: pairwise table is empty."
        )
    resolved_pair = pair
    if resolved_pair is None:
        ranked_pairs = (
            frame.assign(_abs=frame["effect"].abs())
            .groupby(["habitat_a", "habitat_b"], sort=False)["_abs"]
            .mean()
            .sort_values(ascending=False)
        )
        if ranked_pairs.empty:
            raise HABITAPIError(
                "plot_habitat_feature_effect: no finite effect sizes."
            )
        resolved_pair = (
            int(ranked_pairs.index[0][0]),
            int(ranked_pairs.index[0][1]),
        )
    a, b = int(resolved_pair[0]), int(resolved_pair[1])
    work = frame[
        (frame["habitat_a"] == a) & (frame["habitat_b"] == b)
    ].copy()
    if work.empty:
        # Allow the swapped spelling of the same pair.
        work = frame[
            (frame["habitat_a"] == b) & (frame["habitat_b"] == a)
        ].copy()
        work["effect"] = -work["effect"]
        work["mean_diff"] = -work["mean_diff"]
        a, b = b, a
    if work.empty:
        raise HABITAPIError(
            f"plot_habitat_feature_effect: no rows for habitats {a} vs {b}."
        )
    work = work.assign(_abs=work["effect"].abs())
    work = work.sort_values("_abs", ascending=True).tail(max(int(top_k), 1))
    plt = _plt()
    n = int(len(work))
    height_mm = max(62.0, 9.5 * n + 22.0)
    effect_label = (
        "Cliff's delta"
        if comparison.effect == "cliffs_delta"
        else "Cohen's d"
    )
    with use_style("radiology") as style:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=style.figsize(columns=2, height_mm=min(height_mm, 190.0)),
            constrained_layout=True,
        )
        y = np.arange(n)
        values = work["effect"].to_numpy(dtype=np.float64)
        qvals = work["q_value"].to_numpy(dtype=np.float64)
        sig = np.isfinite(qvals) & (qvals < 0.05)
        pos_color = style.palette[0]
        neg_color = style.palette[1]
        colors = [pos_color if v >= 0 else neg_color for v in values]
        ax.axvline(0.0, color="#444444", linewidth=0.8, linestyle="-")
        for index in range(n):
            ax.plot(
                [0.0, values[index]],
                [y[index], y[index]],
                color=colors[index],
                linewidth=1.3,
                solid_capstyle="butt",
            )
            ax.scatter(
                [values[index]],
                [y[index]],
                s=36,
                color=colors[index],
                edgecolor="#222222",
                linewidth=0.6,
                facecolor=colors[index] if sig[index] else "white",
                zorder=3,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(
            [_readable_feature_label(name) for name in work["feature"]],
            fontsize=_TICK_FONTSIZE,
        )
        ax.set_xlabel(
            sanitize_label(f"{effect_label} (H{a} vs H{b})"),
            fontsize=_LABEL_FONTSIZE,
        )
        ax.set_ylabel(sanitize_label("Feature"), fontsize=_LABEL_FONTSIZE)
        finite = values[np.isfinite(values)]
        abs_max = float(np.max(np.abs(finite))) if finite.size else 1.0
        pad = max(0.15, 0.12 * abs_max)
        half = abs_max + pad
        if comparison.effect == "cliffs_delta":
            # Delta lives in [-1, 1]; keep a little room past the data.
            half = min(max(half, 0.55), 1.15)
        ax.set_xlim(-half, half)
        if title is not None:
            resolved = title
        elif comparison.is_cohort:
            n_subj = int(comparison.n_subjects)
            resolved = f"Habitat contrast H{a} vs H{b} (n={n_subj})"
        else:
            resolved = f"Single-subject contrast H{a} vs H{b}"
        ax.set_title(sanitize_label(resolved), fontsize=_TITLE_FONTSIZE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        _apply_readable_fonts(ax)
        _ascii_minus_on_ticks(fig)
    return fig


def plot_habitat_feature_violin(
    data: Union["HabitatFeaturePanel", "HabitatFeatureComparison"],
    *,
    features: Optional[Sequence[str]] = None,
    habitats: Optional[Sequence[int]] = None,
    max_features: int = _DEFAULT_DETAIL_FEATURES,
    pair: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
) -> "Figure":
    """
    Grouped violins (or box + strip when n is small) for a feature shortlist.

    Do not pass hundreds of features -- select them, or let ``max_features``
    take the top-k by absolute effect. A single-subject panel is drawn as
    points. When any habitat in a panel has fewer than 5 points, that panel
    uses a box + strip instead of a KDE violin.

    Args:
        data: Panel or comparison.
        features: Explicit shortlist. Default: top-k by absolute effect or IQR.
        habitats: Optional habitat subset.
        max_features: Cap when ``features`` is omitted.
        pair: Optional pair used only when ranking by effect size.
        title: Optional figure title.

    Returns:
        The matplotlib ``Figure``.
    """
    panel = _as_panel(data)
    comparison = data if hasattr(data, "pairwise") else None
    selected = _select_features_for_overview(
        panel,
        comparison,
        features=features,
        max_features=max_features,
        pair=pair,
    )
    frame = panel.frame[
        panel.frame[panel.feature_column].astype(str).isin(selected)
    ].copy()
    if habitats is not None:
        wanted_h = {int(h) for h in habitats}
        frame = frame[frame[panel.habitat_column].isin(wanted_h)]
    if frame.empty:
        raise HABITAPIError(
            "plot_habitat_feature_violin: no rows for the requested "
            "features / habitats."
        )
    habitat_ids = sorted({int(h) for h in frame[panel.habitat_column]})
    plt = _plt()
    n_feat = len(selected)
    n_cols = 2 if n_feat > 1 else 1
    n_rows = int(np.ceil(n_feat / n_cols))
    with use_style("radiology") as style:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=style.figsize(
                columns=2 if n_feat > 2 else 1,
                height_mm=min(210.0, 50.0 * n_rows + 20.0),
            ),
            squeeze=False,
            constrained_layout=True,
        )
        fig.set_constrained_layout_pads(
            w_pad=0.05, h_pad=0.08, wspace=0.10, hspace=0.16
        )
        palette = list(style.palette)
        for index, feature_name in enumerate(selected):
            ax = axes[index // n_cols][index % n_cols]
            data_by_h: List[np.ndarray] = []
            for hid in habitat_ids:
                data_by_h.append(
                    _feature_values_for_habitat(frame, panel, feature_name, hid)
                )
            positions = np.arange(1, len(habitat_ids) + 1)
            colors = [palette[i % len(palette)] for i in range(len(habitat_ids))]
            sizes = [int(arr.size) for arr in data_by_h if arr.size > 0]
            use_violin = bool(sizes) and min(sizes) >= _MIN_VIOLIN_N
            if use_violin:
                violin_pos = [
                    pos
                    for pos, arr in zip(positions, data_by_h)
                    if arr.size >= _MIN_VIOLIN_N
                ]
                violin_data = [
                    arr for arr in data_by_h if arr.size >= _MIN_VIOLIN_N
                ]
                parts = ax.violinplot(
                    violin_data,
                    positions=violin_pos,
                    showmeans=True,
                    showextrema=False,
                    widths=0.7,
                )
                body_colors = [
                    colors[i]
                    for i, arr in enumerate(data_by_h)
                    if arr.size >= _MIN_VIOLIN_N
                ]
                for body, color in zip(parts["bodies"], body_colors):
                    body.set_facecolor(color)
                    body.set_edgecolor("#222222")
                    body.set_alpha(0.75)
                    body.set_linewidth(0.4)
                if parts.get("cmeans") is not None:
                    parts["cmeans"].set_color("#222222")
                    parts["cmeans"].set_linewidth(0.8)
            else:
                box_pos = [
                    pos for pos, arr in zip(positions, data_by_h) if arr.size > 0
                ]
                box_data = [arr for arr in data_by_h if arr.size > 0]
                if box_data:
                    box = ax.boxplot(
                        box_data,
                        positions=box_pos,
                        widths=0.45,
                        showfliers=False,
                        patch_artist=True,
                        medianprops={"color": "#222222", "linewidth": 0.9},
                        whiskerprops={"color": "#444444", "linewidth": 0.7},
                        capprops={"color": "#444444", "linewidth": 0.7},
                        boxprops={"linewidth": 0.5},
                    )
                    box_colors = [
                        colors[i]
                        for i, arr in enumerate(data_by_h)
                        if arr.size > 0
                    ]
                    for patch, color in zip(box["boxes"], box_colors):
                        patch.set_facecolor(color)
                        patch.set_edgecolor("#222222")
                        patch.set_alpha(0.45)
            for pos, values, color in zip(positions, data_by_h, colors):
                if values.size == 0:
                    continue
                jitter = np.zeros(values.size)
                if values.size > 1:
                    rng = np.random.default_rng(0)
                    jitter = rng.uniform(-0.12, 0.12, size=values.size)
                ax.scatter(
                    np.full(values.size, pos) + jitter,
                    values,
                    s=14,
                    color=color,
                    edgecolor="#222222",
                    linewidth=0.3,
                    zorder=3,
                )
            ax.set_xticks(positions)
            ax.set_xticklabels(
                [sanitize_label(f"H{hid}") for hid in habitat_ids],
                fontsize=_TICK_FONTSIZE,
            )
            ax.set_ylabel(sanitize_label("Value"), fontsize=_LABEL_FONTSIZE)
            ax.set_title(
                _readable_feature_label(feature_name),
                fontsize=_PANEL_TITLE_FONTSIZE,
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
            ax.set_axisbelow(True)
            _apply_readable_fonts(ax)
        # Hide unused axes in the last row.
        for index in range(n_feat, n_rows * n_cols):
            axes[index // n_cols][index % n_cols].set_visible(False)
        fig.suptitle(
            sanitize_label(
                title if title is not None else "Habitat feature distributions"
            ),
            fontsize=_TITLE_FONTSIZE,
        )
        _ascii_minus_on_ticks(fig)
    return fig


def plot_habitat_feature_bars(
    data: Union["HabitatFeaturePanel", "HabitatFeatureComparison"],
    *,
    features: Optional[Sequence[str]] = None,
    habitats: Optional[Sequence[int]] = None,
    subject_id: Optional[str] = None,
    max_features: int = _DEFAULT_DETAIL_FEATURES,
    pair: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
) -> "Figure":
    """
    One bar panel per feature (independent y-axis).

    Grouped bars on a shared linear y-axis mix Energy (~1e9) with
    ``volume_fraction`` (0-1) and crush the small-scale features. Faceting
    keeps the public signature (``features=``, ``subject_id=``, ...) and
    puts each feature on its own scale. Cohort panels show mean +/- 95% CI;
    a single ``subject_id`` shows that subject's values (no error bars).

    Args:
        data: Panel or comparison.
        features: Explicit shortlist. Default: top-k by absolute effect or IQR.
        habitats: Optional habitat subset.
        subject_id: If set, that subject's values (no error bars).
        max_features: Cap when ``features`` is omitted.
        pair: Optional pair used only when ranking by effect size.
        title: Optional figure title.

    Returns:
        The matplotlib ``Figure``.
    """
    panel = _as_panel(data)
    comparison = data if hasattr(data, "pairwise") else None
    if subject_id is not None:
        panel = panel.for_subject(subject_id)
    selected = _select_features_for_overview(
        panel,
        comparison,
        features=features,
        max_features=max_features,
        pair=pair,
    )
    frame = panel.frame[
        panel.frame[panel.feature_column].astype(str).isin(selected)
    ].copy()
    if habitats is not None:
        wanted_h = {int(h) for h in habitats}
        frame = frame[frame[panel.habitat_column].isin(wanted_h)]
    if frame.empty:
        raise HABITAPIError(
            "plot_habitat_feature_bars: no rows for the requested "
            "features / habitats."
        )
    habitat_ids = sorted({int(h) for h in frame[panel.habitat_column]})
    plt = _plt()
    n_feat = len(selected)
    n_h = len(habitat_ids)
    n_cols = min(3, n_feat) if n_feat > 1 else 1
    n_rows = int(np.ceil(n_feat / float(n_cols)))
    x = np.arange(n_h, dtype=np.float64)
    with use_style("radiology") as style:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=style.figsize(
                columns=2 if n_feat > 1 else 1,
                height_mm=min(230.0, 58.0 * n_rows + 32.0),
            ),
            squeeze=False,
            constrained_layout=True,
        )
        fig.set_constrained_layout_pads(
            w_pad=0.06, h_pad=0.10, wspace=0.10, hspace=0.18
        )
        palette = list(style.palette)
        legend_handles = []
        legend_labels: List[str] = []
        for f_index, feature_name in enumerate(selected):
            ax = axes[f_index // n_cols][f_index % n_cols]
            means = np.zeros(n_h, dtype=np.float64)
            half = np.zeros(n_h, dtype=np.float64)
            for h_index, hid in enumerate(habitat_ids):
                values = _feature_values_for_habitat(
                    frame, panel, feature_name, hid
                )
                if values.size == 0:
                    means[h_index] = np.nan
                    continue
                means[h_index] = float(np.mean(values))
                if values.size >= 2 and subject_id is None:
                    sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
                    half[h_index] = 1.96 * sem
            colors = [palette[i % len(palette)] for i in range(n_h)]
            bars = ax.bar(
                x,
                np.nan_to_num(means, nan=0.0),
                width=0.72,
                yerr=half if subject_id is None else None,
                color=colors,
                edgecolor="white",
                linewidth=0.4,
                error_kw={
                    "ecolor": "#444444",
                    "elinewidth": 0.7,
                    "capsize": 2,
                },
            )
            if f_index == 0:
                legend_handles = list(bars)
                legend_labels = [sanitize_label(f"H{hid}") for hid in habitat_ids]
            ax.set_xticks(x)
            ax.set_xticklabels(
                [sanitize_label(f"H{hid}") for hid in habitat_ids],
                fontsize=_TICK_FONTSIZE,
            )
            ax.set_ylabel(sanitize_label("Feature value"), fontsize=_LABEL_FONTSIZE)
            ax.set_title(
                _readable_feature_label(feature_name),
                fontsize=_PANEL_TITLE_FONTSIZE,
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
            ax.set_axisbelow(True)
            _apply_readable_fonts(ax)
        unused = n_rows * n_cols - n_feat
        for index in range(n_feat, n_rows * n_cols):
            axes[index // n_cols][index % n_cols].set_visible(False)
        if legend_handles:
            if unused > 0:
                legend_ax = axes[n_feat // n_cols][n_feat % n_cols]
                legend_ax.set_visible(True)
                legend_ax.axis("off")
                legend_ax.legend(
                    legend_handles,
                    legend_labels,
                    loc="center",
                    frameon=False,
                    fontsize=_LEGEND_FONTSIZE,
                )
            else:
                fig.legend(
                    legend_handles,
                    legend_labels,
                    loc="lower center",
                    ncol=min(n_h, 4),
                    frameon=False,
                    fontsize=_LEGEND_FONTSIZE,
                )
        if title is not None:
            resolved = title
        elif subject_id is not None:
            resolved = f"Habitat feature bars ({subject_id})"
        else:
            resolved = "Habitat feature means (95% CI)"
        fig.suptitle(sanitize_label(resolved), fontsize=_TITLE_FONTSIZE)
        _ascii_minus_on_ticks(fig)
    return fig
