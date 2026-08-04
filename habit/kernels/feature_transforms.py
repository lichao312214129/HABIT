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
"""Fit/apply kernels for tabular feature transforms.

Everything here operates on a bare ``DataFrame`` whose rows are samples and
whose columns are features. Rows being anonymous is the entire point: it is
why one set of functions serves all three places HABIT rescales features --
voxel matrices and supervoxel matrices on the way to a habitat definition
(``habit.domain.feature_preprocessing``), and subject-by-feature modelling
tables on the way to an outcome model (``habit.domain.table_preprocessing``).
Winsorising is the same column-wise computation in all three, and a formula
implemented three times is a formula that will eventually disagree with
itself.

Each transform is split into ``fit_*`` (learn state from one matrix) and
``apply_*`` (replay that state on another). Callers decide what the split
means: a stateless per-subject chain fits and applies to the same matrix,
while a stateful chain fits once on training data and applies everywhere.

The numbers are the v0.1 numbers, with one deliberate correction in
``fit_impute``/``apply_impute``. v0.1 handled non-finite values as a hard-coded
chain prologue that mixed two sources of statistics: NaNs were filled from the
fit-time column means, but infinities were replaced by the means of the block
being transformed (``pipeline._prepare_feature_block`` calling
``value_transforms.handle_extreme_values``). In a stateful chain that let test
data influence its own transformation. Here both come from the fitted state,
so a cohort chain replays training statistics exactly and a per-subject chain
still uses that subject's own -- the distinction now follows from which chain
holds the state rather than from a special case. The two behaviours differ only
on data that actually contains non-finite values.

One v0.1 rule is kept because it is protective rather than incidental: a column
with no finite value at all imputes to 0.0 instead of propagating NaN, so a
single degenerate modality cannot poison an otherwise usable subject.

Every ``fit_*`` here returns JSON-serialisable state: per-column statistics are
plain ``{column: float}`` mappings rather than ``Series``, and ``fit_binning``
stores bin edges rather than a fitted ``KBinsDiscretizer``. This is a hard
requirement, not tidiness -- a cohort chain's state travels inside
``HabitatModel``, and that artefact is deliberately not a bare pickle so it
stays readable across HABIT and scikit-learn versions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "IMPUTE_STRATEGIES",
    "apply_binning",
    "apply_impute",
    "apply_log",
    "apply_minmax",
    "apply_robust",
    "apply_winsorize",
    "apply_zscore",
    "fit_binning",
    "fit_impute",
    "fit_log",
    "fit_minmax",
    "fit_robust",
    "fit_winsorize",
    "fit_zscore",
    "select_correlation_columns",
    "select_variance_columns",
]

#: Statistics available for replacing non-finite values.
IMPUTE_STRATEGIES: Tuple[str, ...] = ("mean", "median", "zero")


# ---------------------------------------------------------------------------
# Serialisable per-column statistics
# ---------------------------------------------------------------------------


def _as_state(statistics: pd.Series) -> Dict[str, float]:
    """
    Convert per-column statistics into a JSON-serialisable mapping.

    Args:
        statistics: Series indexed by feature name.

    Returns:
        Plain ``{column: float}``. Column names are stringified because JSON
        object keys are strings; a non-string feature name would otherwise
        come back as a string and silently stop matching its column.
    """
    return {str(name): float(value) for name, value in statistics.items()}


def _as_series(state: Mapping[str, Any], columns: Sequence[Any]) -> pd.Series:
    """
    Rebuild aligned per-column statistics from serialised state.

    Args:
        state: Mapping produced by :func:`_as_state`.
        columns: Columns of the block being transformed, in order.

    Returns:
        A float Series indexed by ``columns``.

    Raises:
        KeyError: If a column has no learned statistic. Applying a chain to a
            block it was not fitted on is a caller error worth naming, not
            something to paper over with a default.
    """
    missing = [name for name in columns if str(name) not in state]
    if missing:
        raise KeyError(
            "feature preprocessing state has no statistics for column(s) "
            f"{missing}; the chain was fitted on a different feature set."
        )
    return pd.Series(
        [float(state[str(name)]) for name in columns],
        index=list(columns),
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Non-finite imputation
# ---------------------------------------------------------------------------


def fit_impute(block: pd.DataFrame, strategy: str) -> Dict[str, Any]:
    """
    Learn the per-column value that will replace non-finite entries.

    Args:
        block: Unit-by-feature matrix to learn from.
        strategy: One of :data:`IMPUTE_STRATEGIES`. ``mean`` and ``median``
            are computed over each column's FINITE entries only, so an
            infinity cannot contaminate the statistic meant to replace it.

    Returns:
        Serialisable state for :func:`apply_impute`.

    Raises:
        ValueError: If ``strategy`` is not recognised.
    """
    if strategy not in IMPUTE_STRATEGIES:
        raise ValueError(
            f"Unknown impute strategy {strategy!r}; expected one of "
            f"{list(IMPUTE_STRATEGIES)}."
        )
    if strategy == "zero":
        fills = pd.Series(0.0, index=block.columns, dtype=np.float64)
        return {"strategy": strategy, "fills": _as_state(fills)}
    finite_only = block.where(np.isfinite(block.to_numpy(dtype=np.float64)))
    fills = finite_only.mean() if strategy == "mean" else finite_only.median()
    # A column with no finite value yields NaN here; zero keeps a degenerate
    # modality from propagating NaN into the clustering input (v0.1 rule).
    return {
        "strategy": strategy,
        "fills": _as_state(fills.fillna(0.0).astype(np.float64)),
    }


def apply_impute(block: pd.DataFrame, state: Mapping[str, Any]) -> pd.DataFrame:
    """
    Replace every non-finite entry with its column's learned fill value.

    Args:
        block: Matrix to repair.
        state: State from :func:`fit_impute`.

    Returns:
        A matrix with no NaN and no infinity, index and columns preserved.
    """
    if block.empty:
        return block.copy()
    values = block.to_numpy(dtype=np.float64, copy=True)
    non_finite = ~np.isfinite(values)
    if non_finite.any():
        fills = state["fills"]
        for position, column in enumerate(block.columns):
            column_mask = non_finite[:, position]
            if column_mask.any():
                values[column_mask, position] = float(fills.get(str(column), 0.0))
    return pd.DataFrame(values, columns=block.columns, index=block.index)


# ---------------------------------------------------------------------------
# Affine scalers: minmax / zscore / robust
# ---------------------------------------------------------------------------


def fit_minmax(block: pd.DataFrame, across_features: bool) -> Dict[str, Any]:
    """
    Learn min/max scaling state.

    Args:
        block: Matrix to learn from.
        across_features: Pool every column into one (min, max) pair instead
            of keeping per-column statistics.

    Returns:
        Serialisable state for :func:`apply_minmax`.
    """
    if across_features:
        return {
            "across_features": True,
            "min": float(block.to_numpy().min()),
            "max": float(block.to_numpy().max()),
        }
    return {
        "across_features": False,
        "mins": _as_state(block.min()),
        "maxs": _as_state(block.max()),
    }


def apply_minmax(block: pd.DataFrame, state: Mapping[str, Any]) -> pd.DataFrame:
    """
    Scale features to [0, 1] with learned bounds.

    Args:
        block: Matrix to transform.
        state: State from :func:`fit_minmax`.

    Returns:
        The scaled matrix. A degenerate range divides by 1.0, mapping the
        column to 0 instead of NaN.
    """
    if state["across_features"]:
        span = state["max"] - state["min"]
        denominator = span if span != 0 else 1.0
        return (block - state["min"]) / denominator
    columns = list(block.columns)
    mins = _as_series(state["mins"], columns)
    denominator = (_as_series(state["maxs"], columns) - mins).replace(0, 1.0)
    return (block - mins) / denominator


def fit_zscore(block: pd.DataFrame, across_features: bool) -> Dict[str, Any]:
    """
    Learn z-score standardisation state.

    Args:
        block: Matrix to learn from.
        across_features: Pool every column into one (mean, std) pair.

    Returns:
        Serialisable state for :func:`apply_zscore`.
    """
    if across_features:
        values = block.to_numpy()
        deviation = float(values.std())
        return {
            "across_features": True,
            "mean": float(values.mean()),
            "std": deviation if deviation != 0 else 1.0,
        }
    return {
        "across_features": False,
        "means": _as_state(block.mean()),
        "stds": _as_state(block.std().replace(0, 1.0)),
    }


def apply_zscore(block: pd.DataFrame, state: Mapping[str, Any]) -> pd.DataFrame:
    """
    Standardise features with learned mean/std.

    Args:
        block: Matrix to transform.
        state: State from :func:`fit_zscore`.

    Returns:
        The standardised matrix.
    """
    if state["across_features"]:
        return (block - state["mean"]) / state["std"]
    columns = list(block.columns)
    return (block - _as_series(state["means"], columns)) / _as_series(
        state["stds"], columns
    )


def fit_robust(block: pd.DataFrame, across_features: bool) -> Dict[str, Any]:
    """
    Learn median/IQR robust-scaling state.

    Args:
        block: Matrix to learn from.
        across_features: Pool every column into one (median, q1, q3) triple.

    Returns:
        Serialisable state for :func:`apply_robust`.
    """
    if across_features:
        flat = block.to_numpy().flatten()
        return {
            "across_features": True,
            "median": float(np.median(flat)),
            "q1": float(np.percentile(flat, 25)),
            "q3": float(np.percentile(flat, 75)),
        }
    return {
        "across_features": False,
        "medians": _as_state(block.median()),
        "q1s": _as_state(block.quantile(0.25)),
        "q3s": _as_state(block.quantile(0.75)),
    }


def apply_robust(block: pd.DataFrame, state: Mapping[str, Any]) -> pd.DataFrame:
    """
    Centre on the median and divide by the IQR.

    Args:
        block: Matrix to transform.
        state: State from :func:`fit_robust`.

    Returns:
        The robust-scaled matrix; a zero IQR divides by 1.0.
    """
    if state["across_features"]:
        spread = state["q3"] - state["q1"]
        return (block - state["median"]) / (spread if spread != 0 else 1.0)
    columns = list(block.columns)
    spread = (
        _as_series(state["q3s"], columns) - _as_series(state["q1s"], columns)
    ).replace(0, 1.0)
    return (block - _as_series(state["medians"], columns)) / spread


# ---------------------------------------------------------------------------
# Value-shape transforms: winsorize / binning / log
# ---------------------------------------------------------------------------


def fit_winsorize(
    block: pd.DataFrame,
    winsor_limits: Tuple[float, float],
    across_features: bool,
) -> Dict[str, Any]:
    """
    Learn clipping bounds at the requested tail quantiles.

    Args:
        block: Matrix to learn from.
        winsor_limits: Lower and upper tail fractions.
        across_features: Derive one bound pair from the pooled values.

    Returns:
        Serialisable state for :func:`apply_winsorize`.
    """
    lower_fraction, upper_fraction = winsor_limits
    if across_features:
        flat = block.to_numpy().flatten()
        return {
            "across_features": True,
            "lower": float(np.percentile(flat, lower_fraction * 100)),
            "upper": float(np.percentile(flat, (1 - upper_fraction) * 100)),
        }
    return {
        "across_features": False,
        "lower": _as_state(block.quantile(lower_fraction)),
        "upper": _as_state(block.quantile(1 - upper_fraction)),
    }


def apply_winsorize(block: pd.DataFrame, state: Mapping[str, Any]) -> pd.DataFrame:
    """
    Clip values at the learned bounds.

    Args:
        block: Matrix to transform.
        state: State from :func:`fit_winsorize`.

    Returns:
        The clipped matrix.
    """
    if state["across_features"]:
        return block.clip(lower=state["lower"], upper=state["upper"])
    columns = list(block.columns)
    return block.clip(
        lower=_as_series(state["lower"], columns),
        upper=_as_series(state["upper"], columns),
        axis=1,
    )


def fit_binning(
    block: pd.DataFrame,
    n_bins: int,
    bin_strategy: str,
    across_features: bool,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Learn discretisation bin edges.

    Args:
        block: Matrix to learn from.
        n_bins: Number of bins.
        bin_strategy: ``uniform``, ``quantile`` or ``kmeans``.
        across_features: Learn one set of edges from the pooled values.
        random_state: Seed for the stochastic ``kmeans`` strategy.

    Returns:
        Serialisable state for :func:`apply_binning`: the learned bin edges
        and per-column bin counts. The fitted ``KBinsDiscretizer`` itself is
        deliberately NOT kept -- this state ends up inside a shareable
        ``HabitatModel``, which must not depend on unpickling a scikit-learn
        object built by some other version.
    """
    from sklearn.preprocessing import KBinsDiscretizer

    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy=bin_strategy,
        random_state=random_state,
    )
    source = (
        block.to_numpy().flatten().reshape(-1, 1)
        if across_features
        else block.to_numpy()
    )
    discretizer.fit(source)
    return {
        "across_features": bool(across_features),
        "edges": [[float(edge) for edge in edges] for edges in discretizer.bin_edges_],
        "n_bins": [int(count) for count in np.atleast_1d(discretizer.n_bins_)],
    }


def _digitize(values: np.ndarray, edges: Sequence[float], n_bins: int) -> np.ndarray:
    """
    Assign ordinal bin indices, reproducing ``KBinsDiscretizer``.

    Args:
        values: 1-D values to bin.
        edges: Full edge list including the outer two bounds.
        n_bins: Number of bins for this column.

    Returns:
        Float indices in ``[0, n_bins - 1]``. The outer edges are dropped
        before searching and the result is clipped, which is exactly how
        scikit-learn keeps values beyond the fitted range inside the extreme
        bins instead of inventing new ones.
    """
    indices = np.searchsorted(np.asarray(edges[1:-1], dtype=np.float64), values, side="right")
    return np.clip(indices, 0, max(n_bins - 1, 0)).astype(np.float64)


def apply_binning(block: pd.DataFrame, state: Mapping[str, Any]) -> pd.DataFrame:
    """
    Map values onto learned ordinal bin indices.

    Args:
        block: Matrix to transform.
        state: State from :func:`fit_binning`.

    Returns:
        The binned matrix, same shape as the input.
    """
    edges = state["edges"]
    counts = state["n_bins"]
    values = block.to_numpy(dtype=np.float64)
    if state["across_features"]:
        flat = _digitize(values.flatten(), edges[0], counts[0])
        binned = flat.reshape(values.shape)
    else:
        binned = np.empty_like(values, dtype=np.float64)
        for position in range(values.shape[1]):
            binned[:, position] = _digitize(
                values[:, position], edges[position], counts[position]
            )
    return pd.DataFrame(binned, columns=block.columns, index=block.index)


def fit_log(block: pd.DataFrame, across_features: bool) -> Dict[str, Any]:
    """
    Learn the shift that keeps the logarithm's argument positive.

    Args:
        block: Matrix to learn from.
        across_features: Use one pooled minimum as the shift.

    Returns:
        Serialisable state for :func:`apply_log`.
    """
    if across_features:
        return {"across_features": True, "offset": float(block.to_numpy().min())}
    return {"across_features": False, "offsets": _as_state(block.min())}


def apply_log(block: pd.DataFrame, state: Mapping[str, Any]) -> pd.DataFrame:
    """
    Apply ``log(x - offset + 1)`` with the learned offset.

    Args:
        block: Matrix to transform.
        state: State from :func:`fit_log`.

    Returns:
        The log-transformed matrix.
    """
    if state["across_features"]:
        shifted = block.to_numpy() - state["offset"] + 1.0
    else:
        offsets = _as_series(state["offsets"], list(block.columns)).to_numpy()
        shifted = block.to_numpy() - offsets + 1.0
    return pd.DataFrame(
        np.log(shifted), columns=block.columns, index=block.index
    )


# ---------------------------------------------------------------------------
# Column-dropping selections: variance / correlation
# ---------------------------------------------------------------------------


def select_variance_columns(block: pd.DataFrame, threshold: float) -> List[str]:
    """
    Return the columns whose variance exceeds a threshold.

    Args:
        block: Matrix to inspect.
        threshold: Columns with ``var <= threshold`` are dropped; ``0.0``
            removes only constant columns.

    Returns:
        Surviving column names. When nothing survives, the single
        highest-variance column is kept so the chain never empties the
        matrix (the v0.1 rule).
    """
    variances = block.var()
    selected = variances[variances > threshold].index.tolist()
    if not selected:
        selected = [variances.sort_values(ascending=False).index[0]]
    return [str(column) for column in selected]


def select_correlation_columns(
    block: pd.DataFrame,
    threshold: float,
    corr_method: str,
) -> List[str]:
    """
    Return the columns surviving greedy absolute-correlation pruning.

    Walks columns left to right and drops later columns correlating above
    ``threshold`` with a kept one, so the surviving subset is deterministic
    and favours earlier columns -- the v0.1 algorithm.

    Args:
        block: Matrix to inspect.
        threshold: Absolute-correlation cut-off.
        corr_method: ``pearson``, ``spearman`` or ``kendall``.

    Returns:
        Surviving column names.
    """
    if block.shape[1] <= 1:
        return [str(column) for column in block.columns]
    correlation = block.corr(method=corr_method).abs().fillna(0.0)
    kept: Sequence[Any] = list(block.columns)
    position = 0
    while position < len(kept):
        current = kept[position]
        dropped = {
            kept[other]
            for other in range(position + 1, len(kept))
            if correlation.loc[current, kept[other]] > threshold
        }
        kept = [column for column in kept if column not in dropped]
        position += 1
    if not kept:
        kept = [block.columns[0]]
    return [str(column) for column in kept]
