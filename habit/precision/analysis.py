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
"""Precision analysis on voxel feature fields (L3).

A *panel* is the per-feature ICC table of ONE subject under ONE experiment
(e.g. original vs perturbed at the base setting). Panels are aggregated
across the cohort by the per-feature median -- the paper's aggregation
(Radiol Artif Intell 2024;6(2):e230118) -- and a feature is *precise* when
its median lower confidence limit clears the threshold in EVERY experiment.

Default pairing (``pair_mode="common_index"``) aligns voxels on shared
ROI coordinates and drops a row if any condition is NaN. Pass
``pair_mode="prior_pad"`` to copy the reference scripts
(``metrics_repeat.py`` / ``metrics_repro.py``): each condition drops its
own NaNs independently, then the shorter finite vector is padded with
zeros. ``round_decimals=3`` matches those scripts' per-lesion ICC
rounding; the ICC formula itself is the erratum-corrected form in
:mod:`habit.kernels.voxel_icc`.

The per-condition min-max scaling of the reference implementation is kept
(``scale=True``): it removes the arbitrary intensity scale of each feature
map before agreement is measured, so the ICC reflects pattern agreement.
"""

from __future__ import annotations

from typing import List, Literal, Mapping, Optional, Sequence

import warnings

import numpy as np
import pandas as pd

from habit.contracts.habitat import VoxelFeatureField
from habit.precision.precise_set import PreciseFeatureSet
from habit.exceptions import HABITAPIError
from habit.kernels.voxel_icc import icc3a_1, icc3c_1
from habit.spec.specs import Spec

__all__ = ["precision_panel", "aggregate_panels", "identify_precise_features"]

#: ICC flavours selectable for a panel; the paper uses absolute agreement
#: for repeatability (same condition replicated) and consistency for
#: reproducibility (changing conditions).
_AGREEMENTS = {"absolute": icc3a_1, "consistency": icc3c_1}

#: Minimum number of paired voxels for a trustworthy ICC; below this the
#: feature is reported as unmeasurable (NaN) rather than trusted.
DEFAULT_MIN_VOXELS = 10

#: How conditions are lined up before the ICC. ``common_index`` is the
#: HABIT default (spatial join). ``prior_pad`` is the paper GitHub.
PAIR_MODES = ("common_index", "prior_pad")
PairMode = Literal["common_index", "prior_pad"]


def _flat_voxel_index(field: VoxelFeatureField) -> np.ndarray:
    """
    Flatten ``(z, y, x)`` voxel coordinates to 1-D grid positions.

    Args:
        field: Field whose ``voxel_index`` is flattened against its own
            geometry shape.

    Returns:
        One integer grid position per row of the field.
    """
    shape = tuple(int(v) for v in field.geometry.shape)
    return np.ravel_multi_index(field.voxel_index.T.astype(np.int64), shape)


def _aligned_matrices(
    conditions: Mapping[str, VoxelFeatureField],
) -> tuple:
    """
    Align condition fields on their common ROI voxels.

    Args:
        conditions: Condition name to voxel feature field; all fields must
            share the geometry and the feature-name set.

    Returns:
        ``(feature_names, matrices)`` where ``matrices`` maps the condition
        name to its ``(n_common_voxels, n_features)`` array restricted to
        the common voxels, in identical row order.

    Raises:
        HABITAPIError: If the geometries or feature sets differ, or the
            conditions share no voxels.
    """
    names = list(conditions)
    first = conditions[names[0]]
    shape = tuple(int(v) for v in first.geometry.shape)
    row_positions = []
    common: Optional[np.ndarray] = None
    for name in names:
        field = conditions[name]
        if not field.geometry.is_compatible_with(first.geometry):
            raise HABITAPIError(
                f"precision_panel: condition {name!r} does not share the "
                f"voxel grid of condition {names[0]!r}."
            )
        if set(field.feature_names) != set(first.feature_names):
            raise HABITAPIError(
                f"precision_panel: condition {name!r} features differ from "
                f"condition {names[0]!r}; align the extractor settings first."
            )
        flat = _flat_voxel_index(field)
        position = np.empty(flat.max() + 1 if flat.size else 0, dtype=np.int64)
        position[flat] = np.arange(flat.size)
        common = flat if common is None else np.intersect1d(common, flat)
        row_positions.append((flat, position))
    if common is None or common.size == 0:
        raise HABITAPIError("precision_panel: the conditions share no ROI voxels.")
    matrices = {}
    for (flat, position), (name, field) in zip(row_positions, conditions.items()):
        rows = position[common]
        # Reorder columns to the first condition's feature order.
        column_order = [field.feature_names.index(f) for f in first.feature_names]
        matrices[name] = np.asarray(field.values, dtype=np.float64)[rows][
            :, column_order
        ]
    return tuple(first.feature_names), matrices


def _finite_feature_columns(
    conditions: Mapping[str, VoxelFeatureField],
    feature: str,
) -> List[np.ndarray]:
    """
    Drop non-finite values of one feature independently per condition.

    This is their ``feat_arr[~np.isnan(feat_arr)]``: voxel order is the
    field's stored row order, not a shared coordinate join.

    Args:
        conditions: Condition name to voxel feature field; insertion
            order becomes the ICC column order.
        feature: Feature name present in every field.

    Returns:
        One 1-D finite array per condition.
    """
    columns: List[np.ndarray] = []
    for field in conditions.values():
        index = field.feature_names.index(feature)
        values = np.asarray(field.values, dtype=np.float64)[:, index]
        columns.append(values[np.isfinite(values)])
    return columns


def _pad_columns_with_zeros(columns: Sequence[np.ndarray]) -> np.ndarray:
    """
    Pad shorter 1-D columns with trailing zeros to a common length.

    Args:
        columns: Finite (already scaled, if requested) 1-D arrays.

    Returns:
        Array of shape ``(n_padded, n_conditions)``.
    """
    n_pad = max((int(column.size) for column in columns), default=0)
    data = np.zeros((n_pad, len(columns)), dtype=np.float64)
    for j, column in enumerate(columns):
        data[: int(column.size), j] = column
    return data


def _minmax_scale(column: np.ndarray) -> np.ndarray:
    """
    Scale one column to ``[0, 1]``; a constant column maps to zeros.

    This matches ``sklearn.preprocessing.MinMaxScaler`` including its
    zero-variance behaviour, without taking an sklearn call at this layer.

    Args:
        column: Feature values of one condition.

    Returns:
        The scaled column.
    """
    low = float(column.min())
    high = float(column.max())
    if high == low:
        return np.zeros_like(column)
    return (column - low) / (high - low)


def precision_panel(
    conditions: Mapping[str, VoxelFeatureField],
    *,
    agreement: str = "absolute",
    alpha: float = 0.05,
    scale: bool = True,
    min_voxels: int = DEFAULT_MIN_VOXELS,
    pair_mode: PairMode = "common_index",
    round_decimals: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute the per-feature ICC panel of ONE subject under ONE experiment.

    Args:
        conditions: Condition name to voxel feature field of the same
            subject (e.g. ``{"original": f0, "perturbed": f1}``, or
            ``{"R1": f1, "R3": f3}``); at least two.
        agreement: ``"absolute"`` for ICC(3A,1) (repeatability across
            replications of the same condition) or ``"consistency"`` for
            ICC(3C,1) (reproducibility across changing conditions).
        alpha: Two-sided significance level of the confidence limits.
        scale: Min-max scale every feature per condition before the ICC
            (the paper's preprocessing).
        min_voxels: Minimum number of paired, NaN-free voxels; features
            below it are reported as NaN (unmeasurable, fails the screen).
        pair_mode: ``"common_index"`` joins on shared voxel coordinates
            and drops pairwise-incomplete rows. ``"prior_pad"`` drops
            NaNs independently per condition and pads the shorter vector
            with zeros (Prior GitHub ``metrics_repeat.py``).
        round_decimals: If set, round ``value`` / ``lcl`` / ``ucl`` to
            this many decimals after the ICC (their scripts use ``3``).
            ``None`` keeps full precision.

    Returns:
        DataFrame indexed by feature name with columns ``value``, ``lcl``,
        ``ucl`` and ``n_voxels``.

    Raises:
        HABITAPIError: For fewer than two conditions, an unknown agreement
            flavour or pair mode, misaligned inputs, or no shared voxels.
    """
    if len(conditions) < 2:
        raise HABITAPIError(
            f"precision_panel: at least two conditions are required; "
            f"got {len(conditions)}."
        )
    try:
        kernel = _AGREEMENTS[agreement]
    except KeyError:
        raise HABITAPIError(
            f"precision_panel: agreement must be one of {sorted(_AGREEMENTS)}; "
            f"got {agreement!r}."
        ) from None
    if pair_mode not in PAIR_MODES:
        raise HABITAPIError(
            f"precision_panel: pair_mode must be one of {list(PAIR_MODES)}; "
            f"got {pair_mode!r}."
        )
    if round_decimals is not None and int(round_decimals) < 0:
        raise HABITAPIError(
            f"precision_panel: round_decimals must be >= 0 or None; "
            f"got {round_decimals!r}."
        )
    records = []
    if pair_mode == "common_index":
        feature_names, matrices = _aligned_matrices(conditions)
        stacked = np.stack([matrices[name] for name in conditions], axis=1)
        feature_data = []
        for column, feature in enumerate(feature_names):
            data = stacked[:, :, column]
            complete = ~np.isnan(data).any(axis=1)
            feature_data.append((feature, data[complete]))
    else:
        first = next(iter(conditions.values()))
        feature_names = tuple(first.feature_names)
        for field in conditions.values():
            if set(field.feature_names) != set(feature_names):
                raise HABITAPIError(
                    "precision_panel: prior_pad requires every condition "
                    "to share the same feature-name set."
                )
        feature_data = [
            (feature, _pad_columns_with_zeros(
                [
                    _minmax_scale(column) if scale and column.size else column
                    for column in _finite_feature_columns(conditions, feature)
                ]
            ))
            for feature in feature_names
        ]
    for feature, data in feature_data:
        if data.shape[0] < min_voxels:
            records.append((feature, np.nan, np.nan, np.nan, int(data.shape[0])))
            continue
        # prior_pad already min-max'd the finite values before the zero pad.
        if scale and pair_mode == "common_index":
            data = np.apply_along_axis(_minmax_scale, 0, data)
        estimate = kernel(data, alpha=alpha)
        value, lcl, ucl = estimate.value, estimate.lcl, estimate.ucl
        if round_decimals is not None:
            digits = int(round_decimals)
            value = float(np.round(value, digits))
            lcl = float(np.round(lcl, digits))
            ucl = float(np.round(ucl, digits))
        records.append((feature, value, lcl, ucl, int(data.shape[0])))
    frame = pd.DataFrame.from_records(
        records, columns=["feature", "value", "lcl", "ucl", "n_voxels"]
    )
    return frame.set_index("feature")


def aggregate_panels(panels: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate per-subject panels into the cohort-level panel.

    The aggregation is the per-feature MEDIAN of ``value`` / ``lcl`` /
    ``ucl`` across subjects -- the paper's aggregation of per-lesion ICCs.
    A subject whose feature was unmeasurable (NaN) does not veto the
    feature; only a feature unmeasurable in EVERY subject comes out NaN.

    Args:
        panels: Per-subject panels from :func:`precision_panel`, all with
            the same feature index; at least one.

    Returns:
        The cohort-level panel, same columns as the input panels
        (``n_voxels`` is the per-feature median across subjects).

    Raises:
        HABITAPIError: If no panel is given or the feature indices differ.
    """
    if not panels:
        raise HABITAPIError("aggregate_panels: at least one panel is required.")
    index = panels[0].index
    for panel in panels[1:]:
        if not panel.index.equals(index):
            raise HABITAPIError(
                "aggregate_panels: all panels must share the same feature index."
            )
    metrics = np.stack(
        [panel[["value", "lcl", "ucl"]].to_numpy(dtype=np.float64) for panel in panels]
    )
    voxels = np.stack([panel["n_voxels"].to_numpy(dtype=np.float64) for panel in panels])
    with warnings.catch_warnings():
        # All-NaN slices (a feature unmeasurable in every subject) emit a
        # RuntimeWarning via nanmedian; that outcome is intended and is
        # reported as NaN, which fails the precision screen.
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(metrics, axis=0)
        median_voxels = np.nanmedian(voxels, axis=0)
    frame = pd.DataFrame(
        {
            "value": medians[:, 0],
            "lcl": medians[:, 1],
            "ucl": medians[:, 2],
            "n_voxels": median_voxels,
        },
        index=index,
    )
    return frame


def identify_precise_features(
    experiments: Mapping[str, pd.DataFrame],
    *,
    lcl_threshold: float = 0.5,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> PreciseFeatureSet:
    """
    Select the features that clear the LCL threshold in EVERY experiment.

    Args:
        experiments: Experiment name to cohort-level panel (e.g.
            ``{"repeatability": ..., "reproducibility_radius": ...,
            "reproducibility_binwidth": ...}``); at least one.
        lcl_threshold: Lower-confidence-limit cutoff; ``0.5`` is the
            paper's "at least good" boundary.
        include: Expert overrides added regardless of the criteria (the
            paper used this for NGTDM Coarseness); must name real features.
        exclude: Features removed regardless of the criteria.

    Returns:
        The precise feature set with the evidence panels attached.

    Raises:
        HABITAPIError: If no experiment is given, the feature sets differ,
            or an override names an unknown feature.
    """
    if not experiments:
        raise HABITAPIError(
            "identify_precise_features: at least one experiment is required."
        )
    names = list(experiments)
    features = list(experiments[names[0]].index)
    for name in names[1:]:
        if set(experiments[name].index) != set(features):
            raise HABITAPIError(
                f"identify_precise_features: experiment {name!r} features "
                f"differ from experiment {names[0]!r}."
            )
    known = set(features)
    for override, label in ((include, "include"), (exclude, "exclude")):
        unknown = [f for f in override if f not in known]
        if unknown:
            raise HABITAPIError(
                f"identify_precise_features: {label} names unknown features "
                f"{unknown}."
            )
    passes = pd.Series(True, index=features)
    for name in names:
        passes &= experiments[name]["lcl"].reindex(features) >= lcl_threshold
    selected = [f for f in features if bool(passes[f])]
    for feature in include:
        if feature not in selected:
            selected.append(feature)
    selected = [f for f in selected if f not in set(exclude)]
    spec = Spec(
        name="identify_precise_features",
        params={
            "experiments": names,
            "lcl_threshold": float(lcl_threshold),
            "include": list(include),
            "exclude": list(exclude),
        },
    )
    from habit.contracts.provenance import Provenance

    provenance = Provenance.source("precision_analysis").derive(
        produced_by="identify_precise_features",
        spec_fingerprint=spec.fingerprint(),
    )
    return PreciseFeatureSet(
        feature_names=tuple(selected),
        lcl_threshold=float(lcl_threshold),
        experiments=tuple(names),
        panels={name: experiments[name] for name in names},
        provenance=provenance,
    )
