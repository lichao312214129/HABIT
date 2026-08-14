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

These figures exist so a reviewer can see that habitats are genuinely
different and interpretable. The default story is cohort-level:

* heatmap -- features (rows) x habitats (columns), z-scored per feature;
* effect-size forest -- one habitat pair, ranked Cliff's delta / Cohen's d,
  filled marker = BH q < 0.05;
* box+strip (or violin) -- only the top contrasting features for that pair.

High-dimensional texture is never drawn as one violin per feature.
Absent habitats stay masked / omitted (NaN), never zero-filled.

Arrays / panel objects in, ``Figure`` out. No filesystem. Axis text is
ASCII via :func:`~habit.viz.labels.sanitize_label`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from habit.exceptions import HABITAPIError
from habit.utils.optional_deps import require
from habit.viz.labels import sanitize_label
from habit.viz.style import use_style

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from habit.contracts.table import FeatureTable
    from habit.domain.habitat_features.compare import (
        HabitatFeatureComparison,
        HabitatFeaturePanel,
    )

__all__ = [
    "plot_habitat_feature_heatmap",
    "plot_habitat_feature_effect",
    "plot_habitat_feature_violin",
    "plot_habitat_feature_bars",
    "plot_habitat_graph_pair_matrix",
]

_VIZ_PURPOSE = "habitat feature contrast figures"

#: Default cap so a 200-feature heatmap stays readable in one column.
_DEFAULT_HEATMAP_FEATURES = 40
#: Distributions / bars are a shortlist, not the full texture bank.
_DEFAULT_DETAIL_FEATURES = 4
_DEFAULT_EFFECT_TOP_K = 20
#: Distinct fill for masked (NaN) heatmap cells -- not the zero colour.
_MISSING_CELL = "#E6E6E6"
#: Unordered pair columns: ``pair_h{a}_h{b}_{metric}``.
_PAIR_GRAPH_COLUMN = re.compile(r"^pair_h(\d+)_h(\d+)_(.+)$")

# Pyradiomics class prefixes. The class token is dropped for first-order
# (Mean, Skewness) and kept as a short tag for texture families.
_RADIOMICS_LABEL = re.compile(
    r"^(?:original_)?(firstorder|glcm|glrlm|glszm|gldm|ngtdm|shape|shape2d)"
    r"_(.+?)(?:_of_.+)?$",
    re.IGNORECASE,
)
_CLASS_TAG = {
    "firstorder": "",
    "glcm": "GLCM ",
    "glrlm": "GLRLM ",
    "glszm": "GLSZM ",
    "gldm": "GLDM ",
    "ngtdm": "NGTDM ",
    "shape": "shape ",
    "shape2d": "shape ",
}


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


def _as_comparison(
    data: Union["HabitatFeaturePanel", "HabitatFeatureComparison"],
) -> Optional["HabitatFeatureComparison"]:
    """Return the comparison when ``data`` carries a pairwise table."""
    return data if hasattr(data, "pairwise") else None


def _resolve_pair(
    comparison: Optional["HabitatFeatureComparison"],
    pair: Optional[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    """Use the caller pair, else the comparison's strongest pair."""
    if pair is not None:
        return int(pair[0]), int(pair[1])
    if comparison is None:
        return None
    pairwise = getattr(comparison, "pairwise", None)
    if pairwise is None or getattr(pairwise, "empty", True):
        return None
    try:
        return comparison.strongest_pair()
    except HABITAPIError:
        return None


def _matrix_habitats_by_features(
    panel: "HabitatFeaturePanel",
    *,
    subject_id: Optional[str],
    features: Sequence[str],
    habitat_ids: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    Build a habitats x features matrix (mean across subjects unless one id).

    ``habitat_ids`` pins the row set (cohort labels). A habitat that is
    absent for this subject (or this feature) stays NaN -- it is not
    dropped and not filled with zero.

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
    present = [name for name in wanted if name in pivot.columns]
    if not present:
        raise HABITAPIError(
            "plot_habitat_feature_heatmap: requested features are absent "
            "from the panel."
        )
    if habitat_ids is None:
        resolved_h = [int(h) for h in pivot.index.tolist()]
    else:
        resolved_h = [int(h) for h in habitat_ids]
    pivot = pivot.reindex(index=resolved_h, columns=present)
    matrix = pivot.to_numpy(dtype=np.float64)
    return matrix, resolved_h, present


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
            scaled = np.zeros_like(values)
            scaled[~finite] = np.nan
            out[:, col] = scaled
        else:
            scaled = (values - mu) / sd
            scaled[~finite] = np.nan
            out[:, col] = scaled
    return out


def _short_feature_label(name: str, max_len: int = 36) -> str:
    """
    ASCII-sanitise a radiomics / graph name for a journal axis.

    Drops ``original_`` / ``firstorder_`` boilerplate and the trailing
    ``_of_<modality>`` so ``original_firstorder_Mean_of_LAP`` becomes
    ``Mean``. Texture families keep a short tag (``GLCM Contrast``).
    """
    label = sanitize_label(str(name))
    match = _RADIOMICS_LABEL.match(label)
    if match is not None:
        tag = _CLASS_TAG.get(match.group(1).lower(), "")
        label = tag + match.group(2).replace("_", " ")
    else:
        if label.lower().startswith("original_"):
            label = label[9:]
        label = label.replace("_", " ")
    if len(label) <= max_len:
        return label
    return label[: max_len - 3] + "..."


def _select_features(
    panel: "HabitatFeaturePanel",
    comparison: Optional["HabitatFeatureComparison"],
    *,
    features: Optional[Sequence[str]],
    max_features: int,
    pair: Optional[Tuple[int, int]],
    rank_by: str,
) -> List[str]:
    """
    Choose a shortlist of feature names.

    ``rank_by='panel'`` keeps first-seen order (overview heatmap).
    ``rank_by='iqr'`` ranks by across-habitat IQR of cohort means.
    ``rank_by='effect'`` ranks by absolute effect size when a comparison
    is supplied, else falls back to IQR.
    """
    cap = max(int(max_features), 1)
    if features is not None:
        return [str(name) for name in features][:cap]
    names = [str(name) for name in panel.feature_names]
    if rank_by == "panel" or (rank_by == "iqr" and len(names) <= cap):
        return names[:cap]
    if rank_by == "effect" and comparison is not None and not comparison.pairwise.empty:
        ranked = list(comparison.top_features(cap, pair=pair))
        if ranked:
            return ranked
    pivot = panel.frame.pivot_table(
        index=panel.habitat_column,
        columns=panel.feature_column,
        values=panel.value_column,
        aggfunc="mean",
    )
    iqr = (pivot.quantile(0.75) - pivot.quantile(0.25)).sort_values(
        ascending=False
    )
    return [str(name) for name in iqr.index[:cap]]


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
    Feature x habitat heatmap (cohort mean or one subject).

    This is the overview figure: one glance that habitats have different
    profiles. Features are rows and habitats are columns so a 20-40
    feature bank stays readable in a Radiology column. Each feature is
    z-scored across habitats (default) so a GLCM family does not drown
    first-order values. Missing habitat x feature cells are masked
    (light grey), never drawn as zero.

    The default shortlist is panel order (or IQR when truncated). Pass
    ``features`` to pin a paper-specific bank. Effect-size ranking
    belongs on :func:`plot_habitat_feature_effect`, not here.

    Args:
        data: Long panel or a :class:`HabitatFeatureComparison`.
        subject_id: If set, that subject's profile (one case). If
            omitted, the cohort mean per habitat x feature.
        features: Optional explicit feature list.
        max_features: Cap when ``features`` is omitted.
        zscore: Z-score each feature across habitats (default True).
        pair: Unused for ranking; accepted so callers can share kwargs.
        title: Optional figure title. Default states cohort vs one case.

    Returns:
        The matplotlib ``Figure``.
    """
    del pair  # Overview is not the ranked-effect figure.
    panel = _as_panel(data)
    cohort_habitats = list(panel.habitat_ids)
    if subject_id is not None:
        panel = panel.for_subject(subject_id)
    selected = _select_features(
        panel,
        None,
        features=features,
        max_features=max_features,
        pair=None,
        rank_by="panel",
    )
    matrix, habitat_ids, names = _matrix_habitats_by_features(
        panel,
        subject_id=None,
        features=selected,
        habitat_ids=cohort_habitats,
    )
    # Display as features (rows) x habitats (columns).
    shown = _zscore_columns(matrix) if zscore else matrix
    display = np.ma.masked_invalid(shown.T)
    plt = _plt()
    n_feat = max(len(names), 1)
    n_h = max(len(habitat_ids), 1)
    # Radiology 1-column for a few habitats; 2-column when many features
    # need a taller, wider reading pane.
    columns = 2 if n_feat > 18 else 1
    height_mm = max(58.0, 16.0 + 4.6 * n_feat)
    height_mm = min(height_mm, 200.0)
    with use_style("radiology") as style:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=style.figsize(columns=columns, height_mm=height_mm),
            constrained_layout=True,
        )
        cmap = plt.get_cmap("RdBu_r").copy() if zscore else plt.get_cmap("cividis").copy()
        cmap.set_bad(_MISSING_CELL)
        if zscore:
            finite = shown[np.isfinite(shown)]
            vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
            vmax = 1.0 if vmax == 0.0 else vmax
            vmin = -vmax
        else:
            vmin = None
            vmax = None
        image = ax.imshow(
            display,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xticks(np.arange(n_h))
        ax.set_xticklabels([sanitize_label(f"H{hid}") for hid in habitat_ids])
        ax.set_yticks(np.arange(n_feat))
        ax.set_yticklabels(
            [_short_feature_label(name, 40) for name in names],
            fontsize=7,
        )
        ax.set_xlabel(sanitize_label("Habitat"))
        ax.set_ylabel(sanitize_label("Feature"))
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, shrink=0.86)
        cbar.set_label(
            sanitize_label("Z-score" if zscore else "Feature value")
        )
        if title is not None:
            resolved = title
        elif subject_id is not None:
            resolved = f"One case ({subject_id})"
        else:
            n_subj = int(panel.n_subjects)
            if zscore:
                resolved = f"Cohort habitat x feature (z-scored, n={n_subj})"
            else:
                resolved = f"Cohort habitat x feature (n={n_subj})"
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
    Ranked effect-size forest for one habitat pair (the claim figure).

    One pair, top-k features, a vertical line at 0. Filled markers are
    BH q < 0.05 when q-values exist; open markers are not significant
    or untested (single-subject / small n). This is the figure that
    argues habitats differ.

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
    resolved_pair = _resolve_pair(comparison, pair)
    if resolved_pair is None:
        raise HABITAPIError(
            "plot_habitat_feature_effect: no finite effect sizes."
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
    height_mm = max(58.0, 7.2 * n + 22.0)
    effect_label = (
        "Cliff's delta"
        if comparison.effect == "cliffs_delta"
        else "Cohen's d"
    )
    n_subj = int(comparison.n_subjects)
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
                s=32,
                color=colors[index],
                edgecolor="#222222",
                linewidth=0.5,
                facecolor=colors[index] if sig[index] else "white",
                zorder=3,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(
            [_short_feature_label(name, 40) for name in work["feature"]]
        )
        ax.set_xlabel(
            sanitize_label(
                f"{effect_label} (H{a} vs H{b}, n={n_subj}); "
                "filled = q < 0.05"
            )
        )
        ax.set_ylabel(sanitize_label("Feature"))
        if title is not None:
            resolved = title
        elif comparison.is_cohort:
            resolved = f"Habitats differ: H{a} vs H{b}"
        else:
            resolved = f"One-case contrast: H{a} vs H{b}"
        ax.set_title(sanitize_label(resolved))
        # Keep markers off the axis frame so |delta| = 1 stays readable.
        finite_effects = values[np.isfinite(values)]
        if finite_effects.size:
            span = float(np.nanmax(np.abs(finite_effects)))
            limit = 1.15 if span <= 1.0 else span * 1.12
            ax.set_xlim(-limit, limit)
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
    kind: str = "box",
    title: Optional[str] = None,
) -> "Figure":
    """
    Shortlist distributions for the habitats that the claim is about.

    Default is box + strip (honest at small n). Pass ``kind='violin'``
    for kernel densities when the cohort is large. When ``data`` is a
    comparison and ``habitats`` is omitted, only the contrasted pair is
    drawn -- one message: those two habitats separate in the cohort.

    Do not pass hundreds of features. Select them, or let
    ``max_features`` take the top-k by absolute effect.

    Args:
        data: Panel or comparison.
        features: Explicit shortlist. Default: top-k by absolute effect
            or IQR.
        habitats: Optional habitat subset. Default: the contrasted pair
            when a comparison is supplied, else every habitat.
        max_features: Cap when ``features`` is omitted.
        pair: Habitat pair used for ranking and the default habitat
            subset. Default: the strongest pair on a comparison.
        kind: ``'box'`` (default) or ``'violin'``.
        title: Optional figure title.

    Returns:
        The matplotlib ``Figure``.
    """
    kind_name = str(kind).strip().lower()
    if kind_name not in {"box", "violin"}:
        raise HABITAPIError(
            "plot_habitat_feature_violin: kind must be 'box' or "
            f"'violin'; got {kind!r}."
        )
    panel = _as_panel(data)
    comparison = _as_comparison(data)
    resolved_pair = _resolve_pair(comparison, pair)
    selected = _select_features(
        panel,
        comparison,
        features=features,
        max_features=max_features,
        pair=resolved_pair,
        rank_by="effect",
    )
    if habitats is not None:
        wanted_h = [int(h) for h in habitats]
    elif resolved_pair is not None:
        wanted_h = [int(resolved_pair[0]), int(resolved_pair[1])]
    else:
        wanted_h = list(panel.habitat_ids)
    frame = panel.frame[
        panel.frame[panel.feature_column].astype(str).isin(selected)
    ].copy()
    frame = frame[frame[panel.habitat_column].isin(set(wanted_h))]
    if frame.empty:
        raise HABITAPIError(
            "plot_habitat_feature_violin: no rows for the requested "
            "features / habitats."
        )
    # Keep the pair / caller order, not sorted-id order, so H_a vs H_b
    # matches the effect figure.
    habitat_ids = [hid for hid in wanted_h if hid in set(frame[panel.habitat_column].astype(int))]
    plt = _plt()
    n_feat = len(selected)
    n_cols = n_feat if n_feat <= 4 else 2
    n_rows = int(np.ceil(n_feat / n_cols))
    columns = 2 if n_feat > 2 else 1
    with use_style("radiology") as style:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=style.figsize(
                columns=columns,
                height_mm=min(190.0, 48.0 * n_rows + 18.0),
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
            if kind_name == "violin" and panel.n_subjects >= 8:
                violin_pos = [
                    pos for pos, arr in zip(positions, data_by_h) if arr.size > 1
                ]
                violin_data = [arr for arr in data_by_h if arr.size > 1]
                if violin_data:
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
            else:
                box_pos = [
                    pos for pos, arr in zip(positions, data_by_h) if arr.size >= 2
                ]
                box_data = [arr for arr in data_by_h if arr.size >= 2]
                if box_data:
                    boxes = ax.boxplot(
                        box_data,
                        positions=box_pos,
                        widths=0.55,
                        patch_artist=True,
                        showfliers=False,
                        medianprops={"color": "#222222", "linewidth": 0.9},
                        whiskerprops={"color": "#444444", "linewidth": 0.7},
                        capprops={"color": "#444444", "linewidth": 0.7},
                        boxprops={"linewidth": 0.5},
                    )
                    box_colors = [
                        colors[i]
                        for i, arr in enumerate(data_by_h)
                        if arr.size >= 2
                    ]
                    for patch, color in zip(boxes["boxes"], box_colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.55)
                        patch.set_edgecolor("#222222")
            for pos, values, color in zip(positions, data_by_h, colors):
                if values.size == 0:
                    continue
                jitter = np.zeros(values.size)
                if values.size > 1:
                    rng = np.random.default_rng(0)
                    jitter = rng.uniform(-0.10, 0.10, size=values.size)
                ax.scatter(
                    np.full(values.size, pos) + jitter,
                    values,
                    s=12,
                    color=color,
                    edgecolor="#222222",
                    linewidth=0.3,
                    zorder=3,
                )
            ax.set_xticks(positions)
            ax.set_xticklabels(
                [sanitize_label(f"H{hid}") for hid in habitat_ids]
            )
            if index % n_cols == 0:
                ax.set_ylabel(sanitize_label("Value"))
            ax.set_title(_short_feature_label(feature_name, 36), fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
            ax.set_axisbelow(True)
        for index in range(n_feat, n_rows * n_cols):
            axes[index // n_cols][index % n_cols].set_visible(False)
        if title is not None:
            resolved = title
        elif resolved_pair is not None:
            a, b = resolved_pair
            resolved = f"Top features that separate H{a} and H{b}"
        else:
            resolved = "Habitat feature distributions"
        fig.suptitle(sanitize_label(resolved))
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
    zscore: bool = True,
    title: Optional[str] = None,
) -> "Figure":
    """
    Grouped bars for a shortlist (z-scored by default).

    Mixed radiomics units cannot share one raw y-axis (Energy ~ 1e9
    hides volume fraction). Default ``zscore=True`` makes habitats
    comparable within each feature. Missing habitat x feature cells
    are omitted (NaN), not drawn as zero.

    Prefer :func:`plot_habitat_feature_violin` for the cohort
    distribution claim. This helper is a compact profile, including
    one-case bars.

    Args:
        data: Panel or comparison.
        features: Explicit shortlist. Default: top-k by absolute effect
            or IQR.
        habitats: Optional habitat subset.
        subject_id: If set, that subject's values (no error bars).
        max_features: Cap when ``features`` is omitted.
        pair: Optional pair used only when ranking by effect size.
        zscore: Z-score each feature across habitats (default True).
        title: Optional figure title.

    Returns:
        The matplotlib ``Figure``.
    """
    panel = _as_panel(data)
    comparison = _as_comparison(data)
    if subject_id is not None:
        panel = panel.for_subject(subject_id)
    resolved_pair = _resolve_pair(comparison, pair)
    selected = _select_features(
        panel,
        comparison,
        features=features,
        max_features=max_features,
        pair=resolved_pair,
        rank_by="effect",
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
    n_feat = len(selected)
    n_h = len(habitat_ids)
    means = np.full((n_h, n_feat), np.nan, dtype=np.float64)
    half = np.zeros((n_h, n_feat), dtype=np.float64)
    for h_index, hid in enumerate(habitat_ids):
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
                continue
            means[h_index, f_index] = float(np.mean(values))
            if values.size >= 2 and subject_id is None:
                sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
                half[h_index, f_index] = 1.96 * sem
    if zscore:
        shown = _zscore_columns(means.T).T
        # Error bars on raw SEM are not on the z scale; omit them.
        half = np.zeros_like(half)
        ylabel = "Z-score"
    else:
        shown = means
        ylabel = "Feature value"
    plt = _plt()
    x = np.arange(n_feat, dtype=np.float64)
    width = 0.8 / max(n_h, 1)
    with use_style("radiology") as style:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=style.figsize(columns=1, height_mm=64.0),
            constrained_layout=True,
        )
        palette = list(style.palette)
        for h_index, hid in enumerate(habitat_ids):
            offset = (h_index - (n_h - 1) / 2.0) * width
            heights = shown[h_index]
            errs = half[h_index]
            finite = np.isfinite(heights)
            if not bool(finite.any()):
                continue
            yerr = None
            if subject_id is None and not zscore and np.any(errs[finite] > 0):
                yerr = errs[finite]
            ax.bar(
                x[finite] + offset,
                heights[finite],
                width=width * 0.92,
                yerr=yerr,
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
            [_short_feature_label(name, 22) for name in selected],
            rotation=30,
            ha="right",
        )
        ax.set_ylabel(sanitize_label(ylabel))
        ax.legend(frameon=False, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        if title is not None:
            resolved = title
        elif subject_id is not None:
            resolved = f"One case ({subject_id})"
        else:
            resolved = "Habitat feature means (z-scored)"
        ax.set_title(sanitize_label(resolved))
        _ascii_minus_on_ticks(fig)
    return fig


def plot_habitat_graph_pair_matrix(
    data: Union["FeatureTable", pd.DataFrame],
    *,
    metric: str = "contact_voxels_sum",
    subject_id: Optional[str] = None,
    subject_column: Optional[str] = None,
    title: Optional[str] = None,
) -> "Figure":
    """
    Cohort (or one-case) habitat-pair graph metric as a matrix.

    Default extract writes ``pair_h{a}_h{b}_{metric}`` columns. Those
    values are not a per-habitat measurement, so they do not melt
    through :func:`~habit.to_graph_habitat_panel`. This figure is the
    honest pair-level contrast: which habitats contact, and how much.

    The matrix is symmetric. The diagonal is NaN (a habitat is not a
    pair with itself) and missing pairs stay NaN, never zero.

    Args:
        data: Wide graph :class:`~habit.contracts.FeatureTable` or a
            DataFrame with ``pair_h*_h*`` columns.
        metric: Suffix after ``pair_h{a}_h{b}_``. Default
            ``contact_voxels_sum`` (shared-boundary voxels).
        subject_id: If set, that subject only. If omitted, the cohort
            mean across subjects (NaN-safe).
        subject_column: Subject id column when ``data`` is a DataFrame.
            Default: the table's first id column, or ``subject``.
        title: Optional figure title.

    Returns:
        The matplotlib ``Figure``.

    Raises:
        HABITAPIError: If no columns match the requested metric.
    """
    from habit.contracts.table import FeatureTable

    if isinstance(data, FeatureTable):
        frame = data.frame.copy()
        subject = (
            subject_column
            if subject_column is not None
            else (data.id_columns[0] if data.id_columns else "subject")
        )
        columns = list(data.feature_columns)
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
        subject = subject_column if subject_column is not None else "subject"
        columns = [name for name in frame.columns if name != subject]
    else:
        raise HABITAPIError(
            "plot_habitat_graph_pair_matrix expects a FeatureTable or "
            f"DataFrame; got {type(data).__name__}."
        )
    if subject_id is not None:
        if subject not in frame.columns:
            raise HABITAPIError(
                "plot_habitat_graph_pair_matrix: subject column "
                f"{subject!r} is missing."
            )
        frame = frame[frame[subject].astype(str) == str(subject_id)]
        if frame.empty:
            raise HABITAPIError(
                f"plot_habitat_graph_pair_matrix: no subject {subject_id!r}."
            )
    wanted = str(metric)
    parsed: List[Tuple[int, int, str]] = []
    for name in columns:
        match = _PAIR_GRAPH_COLUMN.match(str(name))
        if match is None:
            continue
        if match.group(3) != wanted:
            continue
        parsed.append((int(match.group(1)), int(match.group(2)), str(name)))
    if not parsed:
        raise HABITAPIError(
            "plot_habitat_graph_pair_matrix: no pair_h*_h* columns for "
            f"metric {wanted!r}."
        )
    habitat_ids = sorted({hid for a, b, _ in parsed for hid in (a, b)})
    index = {hid: i for i, hid in enumerate(habitat_ids)}
    n_h = len(habitat_ids)
    matrix = np.full((n_h, n_h), np.nan, dtype=np.float64)
    for habitat_a, habitat_b, column in parsed:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        value = float(np.mean(finite))
        i, j = index[habitat_a], index[habitat_b]
        matrix[i, j] = value
        matrix[j, i] = value
    display = np.ma.masked_invalid(matrix)
    plt = _plt()
    n_subj = (
        1
        if subject_id is not None
        else (
            int(frame[subject].nunique())
            if subject in frame.columns
            else int(len(frame))
        )
    )
    with use_style("radiology") as style:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=style.figsize(columns=1, height_mm=max(62.0, 12.0 * n_h + 28.0)),
            constrained_layout=True,
        )
        cmap = plt.get_cmap("cividis").copy()
        cmap.set_bad(_MISSING_CELL)
        image = ax.imshow(
            display,
            aspect="equal",
            cmap=cmap,
            interpolation="nearest",
        )
        ticks = np.arange(n_h)
        labels = [sanitize_label(f"H{hid}") for hid in habitat_ids]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.set_xlabel(sanitize_label("Habitat"))
        ax.set_ylabel(sanitize_label("Habitat"))
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, shrink=0.86)
        cbar.set_label(sanitize_label(_short_feature_label(wanted, 28)))
        if title is not None:
            resolved = title
        elif subject_id is not None:
            resolved = (
                f"One case ({subject_id}): "
                f"{_short_feature_label(wanted, 24)}"
            )
        else:
            resolved = (
                f"Cohort inter-habitat {_short_feature_label(wanted, 24)} "
                f"(n={n_subj})"
            )
        ax.set_title(sanitize_label(resolved))
        _ascii_minus_on_ticks(fig)
    return fig
