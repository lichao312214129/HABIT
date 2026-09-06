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
"""Per-subject multi-panel GMM habitats from one loaded voxel field.

The caller loads :class:`~habit.contracts.habitat.VoxelFeatureField` once.
Each named panel is an independent column subset: drop non-finite voxels,
drop zero-variance columns, Spearman-filter, then BIC-gradient GMM.
Panel names are caller-defined; this module does not hardcode all / precise
/ Prior-26.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from habit.contracts.habitat import VoxelFeatureField
from habit.exceptions import HABITAPIError
from habit.kernels.cluster_selection import prior2024_bic_gradient_k
from habit.kernels.feature_transforms import select_precise_correlation_columns

# Locked to the already-written per-panel protocol (Scheme A / train80 GMM).
_GMM_RANDOM_STATE: int = 123
_GMM_COVARIANCE: str = "full"
_GMM_N_INIT: int = 1
_K_MIN: int = 1
_K_MAX: int = 5


def drop_nonfinite_voxels(
    matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Drop rows with any non-finite value. No imputation.

    Args:
        matrix: Voxel-by-feature array, shape ``(n_voxels, n_features)``.

    Returns:
        ``(kept_matrix, keep_mask, n_dropped, n_voxels)``. ``keep_mask`` has
        length ``n_voxels`` and is True for rows that remain.
    """
    mat: np.ndarray = np.asarray(matrix, dtype=np.float64)
    n_vox: int = int(mat.shape[0])
    keep_mask: np.ndarray = np.isfinite(mat).all(axis=1)
    n_drop: int = int((~keep_mask).sum())
    return mat[keep_mask], keep_mask, n_drop, n_vox


def varying_column_names(
    matrix: np.ndarray, names: Sequence[str]
) -> Tuple[List[str], int]:
    """Keep columns whose finite variance is strictly positive.

    Args:
        matrix: Finite voxel-by-feature array.
        names: Column names aligned with ``matrix`` columns.

    Returns:
        ``(kept_names, n_constant)``.
    """
    mat: np.ndarray = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[1] == 0:
        return [], 0
    col_var: np.ndarray = (
        np.var(mat, axis=0) if mat.shape[0] else np.zeros(mat.shape[1])
    )
    varying: np.ndarray = np.isfinite(col_var) & (col_var > 0.0)
    kept: List[str] = [str(n) for n, ok in zip(names, varying) if bool(ok)]
    return kept, int((~varying).sum())


def _gmm_labels(matrix: np.ndarray, k: int) -> np.ndarray:
    """Fit one GMM and return 1-based labels.

    A separate fit (not the BIC-search instance) is intentional: the
    already-written npz files used this extra ``random_state=123`` refit.
    Reusing the BIC-search estimator is *probably* bit-identical with
    ``n_init=1``, but that is not proven here, so we refit to keep labels
    identical to the old per-panel worker.
    """
    model = GaussianMixture(
        n_components=int(k),
        random_state=_GMM_RANDOM_STATE,
        covariance_type=_GMM_COVARIANCE,
        n_init=_GMM_N_INIT,
    )
    model.fit(matrix)
    return np.asarray(model.predict(matrix), dtype=np.int32) + 1


def fit_gmm_or_singleton(matrix: np.ndarray) -> Tuple[np.ndarray, int]:
    """BIC-gradient GMM on a finite matrix; ``k=1`` when clustering is impossible.

    Candidates are ``k`` in ``1..min(5, n_voxels)``. Selection is
    :func:`prior2024_bic_gradient_k`, not minimum BIC. The chosen ``k`` is
    then refit (see :func:`_gmm_labels`).

    Args:
        matrix: Finite array, shape ``(n_kept_voxels, n_features)``.

    Returns:
        ``(labels, k)`` with labels in ``1..k``.
    """
    mat: np.ndarray = np.asarray(matrix, dtype=np.float64)
    n_vox: int = int(mat.shape[0])
    n_feat: int = int(mat.shape[1]) if mat.ndim == 2 else 0
    if n_vox <= 0:
        return np.zeros(0, dtype=np.int32), 1
    if n_vox < 2 or n_feat < 1:
        return np.ones(n_vox, dtype=np.int32), 1
    k_grid: List[int] = list(range(_K_MIN, min(_K_MAX + 1, n_vox + 1)))
    if len(k_grid) == 1:
        return _gmm_labels(mat, 1), 1
    bic: List[float] = [
        float(
            GaussianMixture(
                n_components=int(kk),
                random_state=_GMM_RANDOM_STATE,
                covariance_type=_GMM_COVARIANCE,
                n_init=_GMM_N_INIT,
            )
            .fit(mat)
            .bic(mat)
        )
        for kk in k_grid
    ]
    k: int = int(prior2024_bic_gradient_k(bic, k_grid))
    return _gmm_labels(mat, k), k


def fit_one_habitat_panel(
    field: VoxelFeatureField,
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    """Fit one named column panel on an already-loaded field.

    Args:
        field: Loaded voxel feature field. Not modified.
        feature_names: Requested columns. Missing names are dropped; at least
            one name must exist on the field.

    Returns:
        Dict with ``labels`` (``np.ndarray`` of kept voxels), ``k``,
        ``kept_names``, ``keep_mask``, ``n_drop``, ``n_vox``, ``n_const``,
        ``n_keep``.

    Raises:
        HABITAPIError: If no requested column exists on ``field``.
    """
    index: Dict[str, int] = {str(n): i for i, n in enumerate(field.feature_names)}
    shared: List[str] = [str(n) for n in feature_names if str(n) in index]
    if len(shared) < 1:
        raise HABITAPIError(
            f"no shared features for panel on subject {field.subject_id!r}"
        )
    cols: List[int] = [index[n] for n in shared]
    values: np.ndarray = np.asarray(field.values, dtype=np.float64)
    full_all: np.ndarray = values[:, cols]
    full_mat, keep_mask, n_drop, n_vox = drop_nonfinite_voxels(full_all)
    n_keep: int = int(full_mat.shape[0])
    varying_names, n_const = varying_column_names(full_mat, shared)
    kept: List[str]
    labels: np.ndarray
    k: int
    if n_keep < 2 or len(varying_names) < 1:
        labels = np.ones(n_keep, dtype=np.int32)
        k = 1
        kept = list(varying_names)
    elif len(varying_names) < 2:
        kept = list(varying_names)
        labels, k = fit_gmm_or_singleton(full_mat[:, [shared.index(kept[0])]])
    else:
        var_idx: List[int] = [shared.index(c) for c in varying_names]
        df0 = pd.DataFrame(full_mat[:, var_idx], columns=varying_names)
        kept = select_precise_correlation_columns(df0)
        if len(kept) < 1:
            labels = np.ones(n_keep, dtype=np.int32)
            k = 1
            kept = list(varying_names[:1])
        else:
            labels, k = fit_gmm_or_singleton(
                full_mat[:, [shared.index(c) for c in kept]]
            )
    return {
        "labels": np.asarray(labels, dtype=np.int32),
        "k": int(k),
        "kept_names": list(kept),
        "keep_mask": np.asarray(keep_mask, dtype=bool),
        "n_drop": int(n_drop),
        "n_vox": int(n_vox),
        "n_const": int(n_const),
        "n_keep": int(n_keep),
    }


def fit_habitat_panels(
    field: VoxelFeatureField,
    panels: Mapping[str, Sequence[str]],
    *,
    skip_names: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Fit many named column panels independently on one loaded field.

    Each panel is a caller-supplied name -> column list. Panels do not share
    dropped-voxel masks or Spearman survivors; that matches the old
    per-panel protocol. ``skip_names`` omits panels whose labels are already
    on disk (the caller decides existence).

    Args:
        field: Loaded :class:`VoxelFeatureField`. Load it once; this function
            does not read zips.
        panels: Panel name -> requested feature names. Order of iteration is
            the insertion order of ``panels``.
        skip_names: Panel names to ignore (already-written labels).

    Returns:
        Panel name -> :func:`fit_one_habitat_panel` result. Skipped names
        are absent.

    Raises:
        HABITAPIError: If a non-skipped panel shares no columns with ``field``.
        ValueError: If ``panels`` is empty after skips.
    """
    skip = {str(n) for n in (skip_names or ())}
    pending: Dict[str, Sequence[str]] = {
        str(name): list(cols) for name, cols in panels.items() if str(name) not in skip
    }
    if not pending:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for name, cols in pending.items():
        out[name] = fit_one_habitat_panel(field, cols)
    return out


def panel_npz_payload(result: Mapping[str, Any]) -> Dict[str, Union[np.ndarray, int]]:
    """Arrays written by the old per-panel ``{pid}_labels.npz`` contract."""
    return {
        "labels": np.asarray(result["labels"]),
        "k": np.asarray([int(result["k"])], dtype=np.int32),
        "kept_names": np.asarray(result["kept_names"], dtype=object),
        "keep_mask": np.asarray(result["keep_mask"], dtype=bool),
        "n_drop": np.asarray([int(result["n_drop"])], dtype=np.int32),
        "n_vox": np.asarray([int(result["n_vox"])], dtype=np.int32),
        "n_const": np.asarray([int(result["n_const"])], dtype=np.int32),
    }
