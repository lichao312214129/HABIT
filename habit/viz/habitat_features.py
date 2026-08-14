# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
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

Arrays / panel objects in, ``Figure`` out. No filesystem. Axis text is
ASCII via :func:`~habit.viz.labels.sanitize_label`.
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


def _short_feature_label(name: str, max_len: int = 28) -> str:
    """ASCII-sanitise and truncate a long radiomics name."""
    label = sanitize_label(str(name))
    if len(label) <= max_len:
        return label
    return label[: max_len - 3] + "..."


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
    width_mm = min(183.0, 28.0 + 3.2 * n_feat)
    height_mm = max(52.0, 14.0 + 8.0 * len(habitat_ids))
    with use_style("radiology") as style:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=style.figsize(columns=2, height_mm=height_mm)
            if n_feat > 12
            else style.figsize(columns=1, height_mm=height_mm),
            constrained_layout=True,
        )
        # Ignore unused width_mm when style.figsize drives geometry.
        _ = width_mm
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
        ax.set_yticklabels([sanitize_label(f"H{hid}") for hid in habitat_ids])
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(
            [_short_feature_label(name) for name in names],
            rotation=60,
            ha="right",
            fontsize=7,
        )
        ax.set_xlabel(sanitize_label("Feature"))
        ax.set_ylabel(sanitize_label("Habitat"))
        cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label(
            sanitize_label("Z-score" if zscore else "Feature value")
        )
        if title is not None:
            resolved = title
        elif subject_id is not None:
            resolved = f"Habitat feature profile ({subject_id})"
        else:
            resolved = "Cohort mean habitat x feature"
        ax.set_title(sanitize_label(resolved))
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
    or untested (single-subject / small n).

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
    height_mm = max(55.0, 8.0 * n + 18.0)
    effect_label = (
        "Cliff's delta"
        if comparison.effect == "cliffs_delta"
        else "Cohen's d"
    )
    with use_style("radiology") as style:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=style.figsize(columns=1, height_mm=min(height_mm, 180.0)),
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
                linewidth=1.1,
                solid_capstyle="butt",
            )
            ax.scatter(
                [values[index]],
                [y[index]],
                s=28,
                color=colors[index],
                edgecolor="#222222",
                linewidth=0.5,
                facecolor=colors[index] if sig[index] else "white",
                zorder=3,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(
            [_short_feature_label(name, 32) for name in work["feature"]]
        )
        ax.set_xlabel(sanitize_label(f"{effect_label} (H{a} vs H{b})"))
        ax.set_ylabel(sanitize_label("Feature"))
        if title is not None:
            resolved = title
        elif comparison.is_cohort:
            resolved = f"Habitat contrast H{a} vs H{b}"
        else:
            resolved = f"Single-subject contrast H{a} vs H{b}"
        ax.set_title(sanitize_label(resolved))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
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
    Grouped violins for a shortlist of features (cohort distributions).

    Do not pass hundreds of features -- select them, or let ``max_features``
    take the top-k by absolute effect. A single-subject panel is drawn as points.

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
                height_mm=min(200.0, 42.0 * n_rows + 16.0),
            ),
            squeeze=False,
            constrained_layout=True,
        )
        palette = list(style.palette)
        for index, feature_name in enumerate(selected):
            ax = axes[index // n_cols][index % n_cols]
            sub = frame[frame[panel.feature_column].astype(str) == feature_name]
            data_by_h: List[np.ndarray] = []
            for hid in habitat_ids:
                values = pd.to_numeric(
                    sub.loc[
                        sub[panel.habitat_column] == hid, panel.value_column
                    ],
                    errors="coerce",
                ).dropna().to_numpy(dtype=np.float64)
                data_by_h.append(values)
            positions = np.arange(1, len(habitat_ids) + 1)
            colors = [palette[i % len(palette)] for i in range(len(habitat_ids))]
            violin_pos = [
                pos for pos, arr in zip(positions, data_by_h) if arr.size > 1
            ]
            violin_data = [arr for arr in data_by_h if arr.size > 1]
            if violin_data and panel.n_subjects >= 2:
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
                    if arr.size > 1
                ]
                for body, color in zip(parts["bodies"], body_colors):
                    body.set_facecolor(color)
                    body.set_edgecolor("#222222")
                    body.set_alpha(0.75)
                    body.set_linewidth(0.4)
                if parts.get("cmeans") is not None:
                    parts["cmeans"].set_color("#222222")
                    parts["cmeans"].set_linewidth(0.8)
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
                    s=10,
                    color=color,
                    edgecolor="#222222",
                    linewidth=0.3,
                    zorder=3,
                )
            ax.set_xticks(positions)
            ax.set_xticklabels(
                [sanitize_label(f"H{hid}") for hid in habitat_ids]
            )
            ax.set_ylabel(sanitize_label("Value"))
            ax.set_title(_short_feature_label(feature_name, 36), fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
            ax.set_axisbelow(True)
        # Hide unused axes in the last row.
        for index in range(n_feat, n_rows * n_cols):
            axes[index // n_cols][index % n_cols].set_visible(False)
        fig.suptitle(
            sanitize_label(
                title if title is not None else "Habitat feature distributions"
            )
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
    Grouped bars: cohort mean +/- 95% CI, or one subject's values.

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
    x = np.arange(n_feat, dtype=np.float64)
    width = 0.8 / max(n_h, 1)
    with use_style("radiology") as style:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=style.figsize(columns=1, height_mm=62.0),
            constrained_layout=True,
        )
        palette = list(style.palette)
        for h_index, hid in enumerate(habitat_ids):
            means = np.zeros(n_feat, dtype=np.float64)
            half = np.zeros(n_feat, dtype=np.float64)
            for f_index, feature_name in enumerate(selected):
                values = pd.to_numeric(
                    frame.loc[
                        (frame[panel.feature_column].astype(str) == feature_name)
                        & (frame[panel.habitat_column] == hid),
                        panel.value_column,
                    ],
                    errors="coerce",
                ).dropna().to_numpy(dtype=np.float64)
                if values.size == 0:
                    means[f_index] = np.nan
                    continue
                means[f_index] = float(np.mean(values))
                if values.size >= 2 and subject_id is None:
                    sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
                    half[f_index] = 1.96 * sem
            offset = (h_index - (n_h - 1) / 2.0) * width
            ax.bar(
                x + offset,
                np.nan_to_num(means, nan=0.0),
                width=width * 0.92,
                yerr=half if subject_id is None else None,
                color=palette[h_index % len(palette)],
                edgecolor="white",
                linewidth=0.4,
                label=sanitize_label(f"H{hid}"),
                error_kw={
                    "ecolor": "#444444",
                    "elinewidth": 0.7,
                    "capsize": 2,
                },
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_short_feature_label(name, 18) for name in selected],
            rotation=30,
            ha="right",
        )
        ax.set_ylabel(sanitize_label("Feature value"))
        ax.legend(frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        if title is not None:
            resolved = title
        elif subject_id is not None:
            resolved = f"Habitat feature bars ({subject_id})"
        else:
            resolved = "Habitat feature means (95% CI)"
        ax.set_title(sanitize_label(resolved))
        _ascii_minus_on_ticks(fig)
    return fig
