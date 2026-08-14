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
"""Compare per-habitat features within a subject or across a cohort.

``each_habitat`` (and any table that uses the same wide column pattern
``habitat_{id}_{feature}``) stores one row per subject. Reviewers typically
need the opposite view: for each feature, do habitats differ, and by how
much? This module melts that wide table into a long panel and, when more
than one subject is present, runs paired habitat-vs-habitat tests.

The cohort path is the one that supports a methods claim ("habitats are
distinct and interpretable"). The single-subject path is the same objects
with inferential columns left as NaN -- a profile, not a p-value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from habit.contracts.table import FeatureTable
from habit.exceptions import HABITAPIError
from habit.utils.progress_utils import CustomTqdm

__all__ = [
    "HabitatFeaturePanel",
    "HabitatFeatureComparison",
    "to_habitat_feature_panel",
    "compare_habitat_features",
]

#: Wide ``each_habitat`` columns: ``habitat_{id}_{feature}``.
#: ``has_habitat_{id}`` does not match (different prefix).
_WIDE_HABITAT_COLUMN = re.compile(r"^habitat_(\d+)_(.+)$")

_DEFAULT_SUBJECT = "subject"
_DEFAULT_HABITAT = "habitat"
_DEFAULT_FEATURE = "feature"
_DEFAULT_VALUE = "value"

#: Below this paired n, Wilcoxon / FDR are not computed (effect size still is).
_MIN_PAIRED_FOR_TEST = 3

PanelInput = Union["HabitatFeaturePanel", FeatureTable, pd.DataFrame]


@dataclass(frozen=True, eq=False)
class HabitatFeaturePanel:
    """
    Long table of one value per subject x habitat x feature.

    Attributes:
        frame: Columns named by the four ``*_column`` fields.
        subject_column: Subject identifier column.
        habitat_column: Habitat id column (integer labels).
        feature_column: Feature name column.
        value_column: Numeric measurement column. Absent habitats are
            omitted (not filled with zero).
    """

    frame: pd.DataFrame
    subject_column: str = _DEFAULT_SUBJECT
    habitat_column: str = _DEFAULT_HABITAT
    feature_column: str = _DEFAULT_FEATURE
    value_column: str = _DEFAULT_VALUE

    def __post_init__(self) -> None:
        """Validate that the declared columns exist and habitats are integer."""
        required = (
            self.subject_column,
            self.habitat_column,
            self.feature_column,
            self.value_column,
        )
        missing = [name for name in required if name not in self.frame.columns]
        if missing:
            raise HABITAPIError(
                "HabitatFeaturePanel is missing columns: "
                f"{missing}. Present: {list(self.frame.columns)}."
            )
        habitats = pd.to_numeric(
            self.frame[self.habitat_column], errors="coerce"
        )
        if habitats.isna().any():
            raise HABITAPIError(
                "HabitatFeaturePanel habitat column must be integer ids; "
                "got non-numeric values."
            )

    @property
    def n_subjects(self) -> int:
        """Number of distinct subjects in the panel."""
        return int(self.frame[self.subject_column].nunique())

    @property
    def habitat_ids(self) -> Tuple[int, ...]:
        """Sorted habitat ids that have at least one finite value."""
        ids = pd.to_numeric(self.frame[self.habitat_column], errors="coerce")
        unique = sorted({int(value) for value in ids.dropna().unique()})
        return tuple(unique)

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """Feature names in first-seen order."""
        names = self.frame[self.feature_column].astype(str)
        return tuple(dict.fromkeys(names.tolist()))

    def for_subject(self, subject_id: str) -> "HabitatFeaturePanel":
        """
        Return a panel containing only ``subject_id``.

        Args:
            subject_id: Subject identifier to keep.

        Returns:
            A new panel with one subject.

        Raises:
            HABITAPIError: If the subject is absent.
        """
        key = str(subject_id)
        mask = self.frame[self.subject_column].astype(str) == key
        if not bool(mask.any()):
            raise HABITAPIError(
                f"HabitatFeaturePanel has no subject {key!r}."
            )
        return HabitatFeaturePanel(
            frame=self.frame.loc[mask].reset_index(drop=True),
            subject_column=self.subject_column,
            habitat_column=self.habitat_column,
            feature_column=self.feature_column,
            value_column=self.value_column,
        )


@dataclass(frozen=True, eq=False)
class HabitatFeatureComparison:
    """
    Descriptive and (when n>=2) inferential habitat-vs-habitat contrast.

    Attributes:
        panel: The long panel the comparison was computed from.
        summary: Per habitat x feature: n, mean, median, q25, q75.
        pairwise: Per feature x habitat pair: n_paired, mean_diff,
            cliffs_delta, p_value, q_value. Inferential columns are NaN
            when the panel has one subject or too few complete pairs.
        n_subjects: Distinct subjects in ``panel``.
        paired: Whether pairwise tests used a paired design.
        effect: Effect-size name (``cliffs_delta`` or ``cohens_d``).
    """

    panel: HabitatFeaturePanel
    summary: pd.DataFrame
    pairwise: pd.DataFrame
    n_subjects: int
    paired: bool
    effect: str

    @property
    def is_cohort(self) -> bool:
        """True when at least two subjects contribute."""
        return int(self.n_subjects) >= 2

    def top_features(
        self,
        k: int = 8,
        *,
        pair: Optional[Tuple[int, int]] = None,
    ) -> Tuple[str, ...]:
        """
        Rank features by absolute effect size (largest first).

        Args:
            k: Maximum number of names to return.
            pair: Optional ``(habitat_a, habitat_b)`` to restrict ranking.
                When omitted, each feature uses its largest |effect|
                across pairs.

        Returns:
            Feature names, length ``min(k, n_features)``.
        """
        frame = self.pairwise
        if frame.empty:
            return self.panel.feature_names[: max(int(k), 0)]
        work = frame
        if pair is not None:
            a, b = int(pair[0]), int(pair[1])
            work = work[
                ((work["habitat_a"] == a) & (work["habitat_b"] == b))
                | ((work["habitat_a"] == b) & (work["habitat_b"] == a))
            ]
        if work.empty:
            return self.panel.feature_names[: max(int(k), 0)]
        ranked = (
            work.assign(_abs=work["effect"].abs())
            .groupby("feature", sort=False)["_abs"]
            .max()
            .sort_values(ascending=False)
        )
        return tuple(str(name) for name in ranked.index[: max(int(k), 0)])


def to_habitat_feature_panel(
    data: PanelInput,
    *,
    subject_column: Optional[str] = None,
    habitat_column: str = _DEFAULT_HABITAT,
    feature_column: str = _DEFAULT_FEATURE,
    value_column: str = _DEFAULT_VALUE,
) -> HabitatFeaturePanel:
    """
    Melt a wide ``each_habitat`` table, or wrap an already-long frame.

    Wide columns must match ``habitat_{id}_{feature}`` (the
    :class:`~habit.domain.habitat_features.each_habitat.EachHabitatRadiomicsFeatures`
    layout). Presence flags ``has_habitat_{id}`` are ignored. Rows whose
    value is NaN are dropped -- that is the honest "habitat absent /
    not measured" state, not a zero.

    A DataFrame that already has habitat / feature / value columns is
    used as-is (plus a subject column).

    Args:
        data: :class:`~habit.contracts.FeatureTable`, a wide or long
            DataFrame, or an existing panel.
        subject_column: Subject id column. Defaults to the table's first
            id column, or ``subject``.
        habitat_column: Habitat id column name in a long frame, and in
            the returned panel.
        feature_column: Feature-name column in a long frame.
        value_column: Numeric column in a long frame.

    Returns:
        A long :class:`HabitatFeaturePanel`.

    Raises:
        HABITAPIError: If no habitat feature columns can be found.
    """
    if isinstance(data, HabitatFeaturePanel):
        return data

    if isinstance(data, FeatureTable):
        frame = data.frame.copy()
        subject = (
            subject_column
            if subject_column is not None
            else (data.id_columns[0] if data.id_columns else _DEFAULT_SUBJECT)
        )
        candidate_columns = list(data.feature_columns)
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
        subject = (
            subject_column if subject_column is not None else _DEFAULT_SUBJECT
        )
        candidate_columns = [
            name
            for name in frame.columns
            if name
            not in {subject, habitat_column, feature_column, value_column}
        ]
    else:
        raise HABITAPIError(
            "to_habitat_feature_panel expects a FeatureTable, DataFrame, "
            f"or HabitatFeaturePanel; got {type(data).__name__}."
        )

    if subject not in frame.columns:
        raise HABITAPIError(
            f"to_habitat_feature_panel: subject column {subject!r} is "
            f"missing. Present: {list(frame.columns)}."
        )

    long_ready = (
        habitat_column in frame.columns
        and feature_column in frame.columns
        and value_column in frame.columns
    )
    if long_ready:
        long_frame = frame[
            [subject, habitat_column, feature_column, value_column]
        ].copy()
    else:
        parsed: List[Tuple[str, int, str]] = []
        for name in candidate_columns:
            match = _WIDE_HABITAT_COLUMN.match(str(name))
            if match is None:
                continue
            parsed.append((str(name), int(match.group(1)), match.group(2)))
        if not parsed:
            raise HABITAPIError(
                "to_habitat_feature_panel found no columns matching "
                "'habitat_{id}_{feature}'. Pass a wide each_habitat "
                "FeatureTable or a long frame with habitat/feature/value."
            )
        pieces: List[pd.DataFrame] = []
        for column, habitat_id, feature_name in parsed:
            piece = pd.DataFrame(
                {
                    subject: frame[subject].to_numpy(),
                    habitat_column: np.full(len(frame), habitat_id, dtype=int),
                    feature_column: feature_name,
                    value_column: pd.to_numeric(frame[column], errors="coerce"),
                }
            )
            pieces.append(piece)
        long_frame = pd.concat(pieces, ignore_index=True)

    values = pd.to_numeric(long_frame[value_column], errors="coerce")
    long_frame = long_frame.loc[values.notna()].copy()
    long_frame[value_column] = values.loc[values.notna()]
    long_frame[habitat_column] = pd.to_numeric(
        long_frame[habitat_column], errors="coerce"
    ).astype(int)
    long_frame[feature_column] = long_frame[feature_column].astype(str)
    long_frame[subject] = long_frame[subject].astype(str)
    if long_frame.empty:
        raise HABITAPIError(
            "to_habitat_feature_panel: every habitat feature value is "
            "NaN (no measured habitat)."
        )
    return HabitatFeaturePanel(
        frame=long_frame.reset_index(drop=True),
        subject_column=subject,
        habitat_column=habitat_column,
        feature_column=feature_column,
        value_column=value_column,
    )


def compare_habitat_features(
    data: PanelInput,
    *,
    habitats: Optional[Sequence[int]] = None,
    features: Optional[Sequence[str]] = None,
    paired: bool = True,
    effect: str = "cliffs_delta",
    subject_id: Optional[str] = None,
) -> HabitatFeatureComparison:
    """
    Contrast features across habitats for one subject or a cohort.

    Cohort (the intended reviewer figure): each subject contributes a
    paired observation per habitat pair. Default test is Wilcoxon
    signed-rank; default effect is paired Cliff's delta (dominance of
    habitat A over B). p-values are Benjamini-Hochberg adjusted across
    all feature x pair tests. A missing habitat in a subject drops that
    pair for that subject -- it is not imputed.

    Single subject: the same summary and pairwise mean differences are
    returned; p / q / effect-size columns that need a sample are NaN.

    Args:
        data: Panel, wide ``each_habitat`` :class:`FeatureTable`, or
            long/wide DataFrame.
        habitats: Habitat ids to include. Default: all present.
        features: Feature names to include. Default: all present.
        paired: If True (default), pair on subject. If False, treat
            habitat groups as independent (unpaired Cliff's delta /
            Mann-Whitney).
        effect: ``cliffs_delta`` (default) or ``cohens_d``.
        subject_id: If set, restrict the panel to that subject first.

    Returns:
        :class:`HabitatFeatureComparison` with ``summary`` and
        ``pairwise`` frames.

    Raises:
        HABITAPIError: If ``effect`` is unknown or fewer than two
            habitats remain.
    """
    effect_name = str(effect).strip().lower()
    if effect_name not in {"cliffs_delta", "cohens_d"}:
        raise HABITAPIError(
            "compare_habitat_features: effect must be 'cliffs_delta' "
            f"or 'cohens_d'; got {effect!r}."
        )
    panel = to_habitat_feature_panel(data)
    if subject_id is not None:
        panel = panel.for_subject(subject_id)

    work = panel.frame
    if habitats is not None:
        wanted = {int(h) for h in habitats}
        work = work[work[panel.habitat_column].isin(wanted)]
    if features is not None:
        wanted_f = {str(name) for name in features}
        work = work[work[panel.feature_column].isin(wanted_f)]
    if work.empty:
        raise HABITAPIError(
            "compare_habitat_features: no rows left after habitat/feature "
            "filters."
        )
    panel = HabitatFeaturePanel(
        frame=work.reset_index(drop=True),
        subject_column=panel.subject_column,
        habitat_column=panel.habitat_column,
        feature_column=panel.feature_column,
        value_column=panel.value_column,
    )
    habitat_ids = panel.habitat_ids
    if len(habitat_ids) < 2:
        raise HABITAPIError(
            "compare_habitat_features needs at least two habitats; "
            f"got {habitat_ids}."
        )

    summary = _habitat_feature_summary(panel)
    pairwise = _habitat_feature_pairwise(
        panel,
        habitat_ids=habitat_ids,
        paired=bool(paired),
        effect=effect_name,
    )
    return HabitatFeatureComparison(
        panel=panel,
        summary=summary,
        pairwise=pairwise,
        n_subjects=panel.n_subjects,
        paired=bool(paired),
        effect=effect_name,
    )


def _habitat_feature_summary(panel: HabitatFeaturePanel) -> pd.DataFrame:
    """Per-habitat, per-feature descriptive statistics."""
    grouped = panel.frame.groupby(
        [panel.habitat_column, panel.feature_column], sort=True
    )[panel.value_column]
    rows: List[dict] = []
    for (habitat_id, feature_name), series in grouped:
        values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(
            dtype=np.float64
        )
        if values.size == 0:
            continue
        rows.append(
            {
                "habitat": int(habitat_id),
                "feature": str(feature_name),
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
            }
        )
    return pd.DataFrame(rows)


def _habitat_feature_pairwise(
    panel: HabitatFeaturePanel,
    *,
    habitat_ids: Sequence[int],
    paired: bool,
    effect: str,
) -> pd.DataFrame:
    """Pairwise habitat contrasts for every feature."""
    feature_names = panel.feature_names
    pairs = list(combinations(habitat_ids, 2))
    n_tests = len(feature_names) * len(pairs)
    show_bar = n_tests > 50
    iterator: Iterable[Tuple[str, Tuple[int, int]]] = (
        (feature_name, pair)
        for feature_name in feature_names
        for pair in pairs
    )
    if show_bar:
        iterator = CustomTqdm(
            list(iterator),
            desc="Habitat feature contrasts",
            total=n_tests,
        )

    rows: List[dict] = []
    for feature_name, (habitat_a, habitat_b) in iterator:
        a_vals, b_vals = _aligned_habitat_values(
            panel,
            feature_name=feature_name,
            habitat_a=int(habitat_a),
            habitat_b=int(habitat_b),
            paired=paired,
        )
        n_paired = int(a_vals.size)
        if n_paired == 0:
            continue
        mean_diff = float(np.mean(a_vals - b_vals)) if paired else float(
            np.mean(a_vals) - np.mean(b_vals)
        )
        effect_value = _effect_size(a_vals, b_vals, paired=paired, effect=effect)
        p_value = _pairwise_p_value(
            a_vals, b_vals, paired=paired, n_subjects=panel.n_subjects
        )
        rows.append(
            {
                "feature": feature_name,
                "habitat_a": int(habitat_a),
                "habitat_b": int(habitat_b),
                "n_paired": n_paired,
                "mean_diff": mean_diff,
                "effect": effect_value,
                "p_value": p_value,
            }
        )
    pairwise = pd.DataFrame(rows)
    if pairwise.empty:
        pairwise = pd.DataFrame(
            columns=[
                "feature",
                "habitat_a",
                "habitat_b",
                "n_paired",
                "mean_diff",
                "effect",
                "p_value",
                "q_value",
            ]
        )
        return pairwise
    pairwise["q_value"] = _benjamini_hochberg(
        pairwise["p_value"].to_numpy(dtype=np.float64)
    )
    return pairwise


def _aligned_habitat_values(
    panel: HabitatFeaturePanel,
    *,
    feature_name: str,
    habitat_a: int,
    habitat_b: int,
    paired: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return finite values for one feature and two habitats."""
    frame = panel.frame
    feat = frame[panel.feature_column].astype(str) == str(feature_name)
    subset = frame.loc[feat]
    if subset.empty:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    if not paired:
        a = pd.to_numeric(
            subset.loc[
                subset[panel.habitat_column] == habitat_a, panel.value_column
            ],
            errors="coerce",
        ).dropna().to_numpy(dtype=np.float64)
        b = pd.to_numeric(
            subset.loc[
                subset[panel.habitat_column] == habitat_b, panel.value_column
            ],
            errors="coerce",
        ).dropna().to_numpy(dtype=np.float64)
        return a, b

    wide = subset.pivot_table(
        index=panel.subject_column,
        columns=panel.habitat_column,
        values=panel.value_column,
        aggfunc="mean",
    )
    if habitat_a not in wide.columns or habitat_b not in wide.columns:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    pair = wide[[habitat_a, habitat_b]].dropna()
    return (
        pair[habitat_a].to_numpy(dtype=np.float64),
        pair[habitat_b].to_numpy(dtype=np.float64),
    )


def _effect_size(
    a: np.ndarray,
    b: np.ndarray,
    *,
    paired: bool,
    effect: str,
) -> float:
    """Cliff's delta or Cohen's d; NaN when undefined."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    if effect == "cohens_d":
        return _cohens_d(a, b, paired=paired)
    if paired:
        return _paired_cliffs_delta(a, b)
    return _unpaired_cliffs_delta(a, b)


def _paired_cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Paired dominance / paired Cliff's delta.

    ``(n_a>b - n_a<b) / n`` on finite pairs. Ties count in the
    denominator only, so the value stays in ``[-1, 1]``.
    """
    if a.size != b.size or a.size == 0:
        return float("nan")
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = int(diff.size)
    n_pos = int(np.sum(diff > 0.0))
    n_neg = int(np.sum(diff < 0.0))
    return float(n_pos - n_neg) / float(n)


def _unpaired_cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Unpaired Cliff's delta: (n_gt - n_lt) / (n_a * n_b)."""
    x = np.asarray(a, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(b, dtype=np.float64).reshape(1, -1)
    if x.size == 0 or y.size == 0:
        return float("nan")
    n_gt = float(np.sum(x > y))
    n_lt = float(np.sum(x < y))
    return (n_gt - n_lt) / float(x.size * y.size)


def _cohens_d(a: np.ndarray, b: np.ndarray, *, paired: bool) -> float:
    """Paired or unpaired Cohen's d; NaN when the SD is zero."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if paired:
        if x.size != y.size or x.size < 2:
            return float("nan")
        diff = x - y
        sd = float(np.std(diff, ddof=1))
        if sd == 0.0:
            return 0.0 if float(np.mean(diff)) == 0.0 else float("nan")
        return float(np.mean(diff) / sd)
    if x.size < 2 or y.size < 2:
        return float("nan")
    n_x = float(x.size)
    n_y = float(y.size)
    var = ((n_x - 1.0) * float(np.var(x, ddof=1))) + (
        (n_y - 1.0) * float(np.var(y, ddof=1))
    )
    denom = n_x + n_y - 2.0
    if denom <= 0.0:
        return float("nan")
    pooled = float(np.sqrt(var / denom))
    if pooled == 0.0:
        return 0.0 if float(np.mean(x) - np.mean(y)) == 0.0 else float("nan")
    return float((np.mean(x) - np.mean(y)) / pooled)


def _pairwise_p_value(
    a: np.ndarray,
    b: np.ndarray,
    *,
    paired: bool,
    n_subjects: int,
) -> float:
    """Wilcoxon (paired) or Mann-Whitney (unpaired); NaN if underpowered."""
    if n_subjects < 2:
        return float("nan")
    if paired:
        if a.size < _MIN_PAIRED_FOR_TEST or a.size != b.size:
            return float("nan")
        if np.allclose(a, b):
            return 1.0
        try:
            result = stats.wilcoxon(
                a, b, zero_method="wilcox", alternative="two-sided"
            )
        except ValueError:
            return float("nan")
        return float(result.pvalue)
    if a.size < _MIN_PAIRED_FOR_TEST or b.size < _MIN_PAIRED_FOR_TEST:
        return float("nan")
    try:
        result = stats.mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        return float("nan")
    return float(result.pvalue)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """BH-FDR q-values; NaN p-values stay NaN."""
    p = np.asarray(p_values, dtype=np.float64).reshape(-1)
    q = np.full(p.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(p)
    if not bool(finite.any()):
        return q
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx], kind="mergesort")]
    m = int(order.size)
    ranked = p[order]
    raw = ranked * float(m) / np.arange(1, m + 1, dtype=np.float64)
    adj = np.minimum.accumulate(raw[::-1])[::-1]
    q[order] = np.clip(adj, 0.0, 1.0)
    return q
