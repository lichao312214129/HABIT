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
"""
Vectorized CPU formulas for stacked multi-label radiomics matrices.

Every function takes a leading-label batch ``[K, ...]`` and returns one
1-D array per requested feature. Formulas follow PyRadiomics 3.1.0
(``radiomics/glcm.py``, ``glrlm.py``, ``glszm.py``, ``gldm.py``,
``ngtdm.py``, ``firstorder.py``). No SimpleITK / TorchRadiomics calculator
is constructed.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

_EPS = float(np.spacing(1))

# Column order returned by ``sv_calculate_firstorder``.
FIRSTORDER_CEXT_COLUMNS: Sequence[str] = (
    "Energy",
    "TotalEnergy",
    "Entropy",
    "Minimum",
    "10Percentile",
    "90Percentile",
    "Maximum",
    "Mean",
    "Median",
    "InterquartileRange",
    "Range",
    "MeanAbsoluteDeviation",
    "RobustMeanAbsoluteDeviation",
    "RootMeanSquared",
    "Skewness",
    "Kurtosis",
    "Uniformity",
)

# Column order returned by ``sv_glcm_formulas`` (MCC is the last column).
GLCM_FORMULA_COLUMNS: Sequence[str] = (
    "Autocorrelation",
    "JointAverage",
    "ClusterProminence",
    "ClusterShade",
    "ClusterTendency",
    "Contrast",
    "Correlation",
    "DifferenceAverage",
    "DifferenceEntropy",
    "DifferenceVariance",
    "JointEnergy",
    "JointEntropy",
    "Imc1",
    "Imc2",
    "Idm",
    "Idmn",
    "Id",
    "Idn",
    "InverseVariance",
    "MaximumProbability",
    "SumAverage",
    "SumEntropy",
    "SumSquares",
    "MCC",
)

# Column order returned by ``sv_glrlm_formulas``.
GLRLM_FORMULA_COLUMNS: Sequence[str] = (
    "ShortRunEmphasis",
    "LongRunEmphasis",
    "GrayLevelNonUniformity",
    "GrayLevelNonUniformityNormalized",
    "RunLengthNonUniformity",
    "RunLengthNonUniformityNormalized",
    "RunPercentage",
    "GrayLevelVariance",
    "RunVariance",
    "RunEntropy",
    "LowGrayLevelRunEmphasis",
    "HighGrayLevelRunEmphasis",
    "ShortRunLowGrayLevelEmphasis",
    "ShortRunHighGrayLevelEmphasis",
    "LongRunLowGrayLevelEmphasis",
    "LongRunHighGrayLevelEmphasis",
)


def _as_float(array: np.ndarray) -> np.ndarray:
    """Cast a matrix to float64 without copying when already float64."""
    return np.asarray(array, dtype=np.float64)


def _nanmean_angles(values: np.ndarray) -> np.ndarray:
    """
    Average feature values over the last (angle) axis, ignoring NaNs.

    Args:
        values: Array of shape ``[K, Na]``.

    Returns:
        np.ndarray: Shape ``[K]``.
    """
    return np.nanmean(values, axis=-1)


def _gray_vector(n_gray: int, gray_levels: Optional[np.ndarray]) -> np.ndarray:
    """Return 1-indexed gray-level values of length ``n_gray``."""
    if gray_levels is not None and np.asarray(gray_levels).size == n_gray:
        return np.asarray(gray_levels, dtype=np.float64).reshape(-1)
    return np.arange(1, n_gray + 1, dtype=np.float64)


def _mcc_pruned_eigvals(
    p_norm: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    empty: np.ndarray,
    n_gray: int,
    n_labels: int,
    n_angles: int,
    eps: float,
) -> np.ndarray:
    """
    MCC matching PyRadiomics ``getMCCFeatureValue``.

    Q is not symmetric:

        Q(i, j) = sum_k P(i,k) P(j,k) / (p_x(i) p_y(k))

    so the second-largest eigenvalue must come from ``eigvals``, not
    ``eigvalsh`` / Jacobi-on-Q. Unused gray levels (``p_x = 0``) add only
    zero eigenvalues and can be dropped.

    Args:
        p_norm: Normalised GLCM ``[K, Ng, Ng, Na]``.
        px: Row marginals ``[K, Ng, 1, Na]``.
        py: Column marginals ``[K, 1, Ng, Na]``.
        empty: Boolean mask of labels/angles with zero mass ``[K, Na]``.
        n_gray: Union-bin gray-level count.
        n_labels: Number of labels.
        n_angles: Number of angles.
        eps: Floor added to the Q denominator (PyRadiomics ``eps``).

    Returns:
        np.ndarray: MCC averaged over angles, shape ``[K]``.
    """
    if n_gray < 2:
        return np.ones(n_labels, dtype=np.float64)
    p_ij = np.nan_to_num(p_norm, nan=0.0)
    px_va = np.nan_to_num(px[:, :, 0, :], nan=0.0)
    py_va = np.nan_to_num(py[:, 0, :, :], nan=0.0)
    ang_vals = np.full((n_labels, n_angles), np.nan, dtype=np.float64)
    for v in range(n_labels):
        for a in range(n_angles):
            keep = np.flatnonzero(px_va[v, :, a] > 0.0)
            if keep.size < 2:
                # PyRadiomics returns 1 when every GLCM is 1x1 (flat region).
                ang_vals[v, a] = 1.0
                continue
            s = p_ij[v][np.ix_(keep, keep)][:, :, a]
            px_k = px_va[v, keep, a]
            py_k = py_va[v, keep, a]
            # True (non-symmetric) Q; match radiomics/glcm.py getMCCFeatureValue.
            q = (s[:, None, :] * s[None, :, :]) / (
                px_k[:, None, None] * py_k[None, None, :] + eps
            )
            q = q.sum(axis=-1)
            ev = np.linalg.eigvals(q)
            ev.sort()
            ang_vals[v, a] = float(np.real(np.sqrt(ev[-2])))
    ang_vals[empty] = np.nan
    return _nanmean_angles(ang_vals)


# Backward-compatible name used by older tests / call sites.
_mcc_pruned_eigvalsh = _mcc_pruned_eigvals


def glcm_features(
    p_counts: np.ndarray,
    feature_names: Sequence[str],
    *,
    gray_levels: Optional[np.ndarray] = None,
    ng_full: Optional[np.ndarray] = None,
    symmetrical: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Evaluate GLCM features on stacked integer co-occurrence counts.

    Args:
        p_counts: ``[K, Ng, Ng, Na]`` raw (un-normalised) counts.
        feature_names: Enabled GLCM feature names.
        gray_levels: Optional 1-indexed gray-level values of length ``Ng``.
        ng_full: Per-label ``Ng`` used by Idn / Idmn (union bin: one value).
        symmetrical: When True, add the transpose before normalising.

    Returns:
        Dict[str, np.ndarray]: Feature name -> ``[K]`` values.
    """
    p_raw = _as_float(p_counts)
    if p_raw.ndim != 4:
        raise ValueError(f"GLCM matrix must be [K, Ng, Ng, Na]; got {p_raw.shape}")
    n_labels, n_gray, _, n_angles = p_raw.shape
    needed = set(feature_names)
    ivector = _gray_vector(n_gray, gray_levels)
    if ng_full is None:
        ng_scale = np.full(n_labels, float(n_gray), dtype=np.float64)
    else:
        ng_scale = np.asarray(ng_full, dtype=np.float64).reshape(-1)
        if ng_scale.size == 1:
            ng_scale = np.full(n_labels, float(ng_scale[0]), dtype=np.float64)

    # Empty stacked GLCM: skip every formula (including MCC eigvalsh).
    if not np.any(p_raw):
        return {
            name: np.full(n_labels, np.nan, dtype=np.float64) for name in feature_names
        }

    c_names = [name for name in feature_names if name in GLCM_FORMULA_COLUMNS]
    out: Dict[str, np.ndarray] = {}
    try:
        from habit.kernels.radiomics.cext import (
            cext_backend,
            glcm_formulas as c_glcm_formulas,
        )

        use_c = cext_backend() == "native"
    except Exception:
        use_c = False
        c_glcm_formulas = None  # type: ignore[assignment]
    if use_c and c_glcm_formulas is not None and c_names:
        packed = c_glcm_formulas(p_raw, ivector, ng_scale, int(bool(symmetrical)))
        name_to_col = {name: idx for idx, name in enumerate(GLCM_FORMULA_COLUMNS)}
        finite_c = False
        for name in c_names:
            col = np.asarray(packed[:, name_to_col[name]], dtype=np.float64)
            out[name] = col
            if np.isfinite(col).any():
                finite_c = True
        if not finite_c:
            out.clear()
        else:
            needed = needed - set(c_names)
            if not needed:
                return out

    if symmetrical:
        p_raw = p_raw + np.transpose(p_raw, (0, 2, 1, 3))

    sum_p = p_raw.sum(axis=(1, 2))
    sum_p = sum_p.astype(np.float64, copy=False)
    empty = sum_p == 0
    sum_p[empty] = np.nan
    p_norm = p_raw / sum_p[:, None, None, :]

    ivector = _gray_vector(n_gray, gray_levels)
    i_grid = ivector[:, None]
    j_grid = ivector[None, :]
    eps = _EPS

    need_px = bool(needed & {"Imc1", "Imc2", "MCC", "Correlation"})
    need_ux = bool(
        needed
        & {
            "JointAverage",
            "ClusterProminence",
            "ClusterShade",
            "ClusterTendency",
            "Correlation",
            "SumSquares",
        }
    )
    px = p_norm.sum(axis=2, keepdims=True) if need_px else None
    py = p_norm.sum(axis=1, keepdims=True) if need_px else None
    ux = (
        (p_norm * i_grid[None, :, :, None]).sum(axis=(1, 2), keepdims=True)
        if need_ux
        else None
    )
    uy = (
        (p_norm * j_grid[None, :, :, None]).sum(axis=(1, 2), keepdims=True)
        if need_ux
        else None
    )

    need_diff = needed & {
        "DifferenceAverage",
        "DifferenceEntropy",
        "DifferenceVariance",
        "Idm",
        "Idmn",
        "Id",
        "Idn",
        "InverseVariance",
    }
    need_sum = needed & {"SumAverage", "SumEntropy"}
    px_suby = None
    px_addy = None
    if need_diff:
        px_suby = np.zeros((n_labels, n_gray, n_angles), dtype=np.float64)
        abs_diff = np.abs(i_grid - j_grid)
        for k in range(n_gray):
            mask = abs_diff == k
            if np.any(mask):
                px_suby[:, k, :] = p_norm[:, mask, :].sum(axis=1)
    if need_sum:
        n_sum = 2 * n_gray - 1
        px_addy = np.zeros((n_labels, n_sum, n_angles), dtype=np.float64)
        add = i_grid + j_grid
        for k_idx, k in enumerate(range(2, 2 * n_gray + 1)):
            mask = add == k
            if np.any(mask):
                px_addy[:, k_idx, :] = p_norm[:, mask, :].sum(axis=1)

    if ng_full is None:
        ng_scale = np.full(n_labels, float(n_gray), dtype=np.float64)
    else:
        ng_scale = np.asarray(ng_full, dtype=np.float64).reshape(-1)
        if ng_scale.size == 1:
            ng_scale = np.full(n_labels, float(ng_scale[0]), dtype=np.float64)

    k_diff = np.arange(n_gray, dtype=np.float64)
    k_sum = np.arange(2, 2 * n_gray + 1, dtype=np.float64)

    if "Autocorrelation" in needed:
        out["Autocorrelation"] = _nanmean_angles(
            (p_norm * (i_grid * j_grid)[None, :, :, None]).sum(axis=(1, 2))
        )
    if "JointAverage" in needed and ux is not None:
        out["JointAverage"] = _nanmean_angles(ux.reshape(n_labels, n_angles))
    if needed & {"ClusterProminence", "ClusterShade", "ClusterTendency"}:
        delta = (i_grid + j_grid)[None, :, :, None] - ux - uy
        if "ClusterTendency" in needed:
            out["ClusterTendency"] = _nanmean_angles((p_norm * (delta ** 2)).sum(axis=(1, 2)))
        if "ClusterShade" in needed:
            out["ClusterShade"] = _nanmean_angles((p_norm * (delta ** 3)).sum(axis=(1, 2)))
        if "ClusterProminence" in needed:
            out["ClusterProminence"] = _nanmean_angles((p_norm * (delta ** 4)).sum(axis=(1, 2)))
    if "Contrast" in needed:
        out["Contrast"] = _nanmean_angles(
            (p_norm * (np.abs(i_grid - j_grid)[None, :, :, None] ** 2)).sum(axis=(1, 2))
        )
    if "Correlation" in needed and ux is not None and uy is not None:
        sigx = np.sqrt((p_norm * ((i_grid[None, :, :, None] - ux) ** 2)).sum(axis=(1, 2), keepdims=True))
        sigy = np.sqrt((p_norm * ((j_grid[None, :, :, None] - uy) ** 2)).sum(axis=(1, 2), keepdims=True))
        corm = (p_norm * (i_grid[None, :, :, None] - ux) * (j_grid[None, :, :, None] - uy)).sum(
            axis=(1, 2), keepdims=True
        )
        corr = corm / (sigx * sigy + eps)
        corr[(sigx * sigy) == 0] = 1.0
        out["Correlation"] = np.nanmean(corr, axis=(1, 2, 3))
    if "DifferenceAverage" in needed and px_suby is not None:
        out["DifferenceAverage"] = _nanmean_angles((k_diff[None, :, None] * px_suby).sum(axis=1))
    if "DifferenceEntropy" in needed and px_suby is not None:
        out["DifferenceEntropy"] = _nanmean_angles(
            (-1.0) * (px_suby * np.log2(px_suby + eps)).sum(axis=1)
        )
    if "DifferenceVariance" in needed and px_suby is not None:
        diffavg = (k_diff[None, :, None] * px_suby).sum(axis=1, keepdims=True)
        out["DifferenceVariance"] = _nanmean_angles(
            (px_suby * ((k_diff[None, :, None] - diffavg) ** 2)).sum(axis=1)
        )
    if "JointEnergy" in needed:
        out["JointEnergy"] = _nanmean_angles((p_norm ** 2).sum(axis=(1, 2)))
    hxy = None
    if needed & {"JointEntropy", "Imc1", "Imc2"}:
        hxy = (-1.0) * (p_norm * np.log2(p_norm + eps)).sum(axis=(1, 2))
    if "JointEntropy" in needed and hxy is not None:
        out["JointEntropy"] = _nanmean_angles(hxy)
    if "Imc1" in needed and hxy is not None and px is not None and py is not None:
        hx = (-1.0) * (px * np.log2(px + eps)).sum(axis=(1, 2))
        hy = (-1.0) * (py * np.log2(py + eps)).sum(axis=(1, 2))
        hxy1 = (-1.0) * (p_norm * np.log2(px * py + eps)).sum(axis=(1, 2))
        div = np.fmax(hx, hy)
        imc1 = hxy - hxy1
        imc1 = np.divide(imc1, div, out=np.zeros_like(imc1), where=div != 0)
        out["Imc1"] = _nanmean_angles(imc1)
    if "Imc2" in needed and hxy is not None and px is not None and py is not None:
        hxy2 = (-1.0) * ((px * py) * np.log2(px * py + eps)).sum(axis=(1, 2))
        imc2 = np.sqrt(np.clip(1.0 - np.exp(-2.0 * (hxy2 - hxy)), 0.0, None))
        imc2[hxy2 == hxy] = 0.0
        out["Imc2"] = _nanmean_angles(imc2)
    if "Idm" in needed and px_suby is not None:
        out["Idm"] = _nanmean_angles((px_suby / (1.0 + k_diff[None, :, None] ** 2)).sum(axis=1))
    if "MCC" in needed and px is not None and py is not None:
        # Unused gray levels make zero rows of Q and do not change the
        # second-largest eigenvalue. Prune per (label, angle) so eigvals
        # runs on Ng_used x Ng_used, not the union-bin Ng.
        out["MCC"] = _mcc_pruned_eigvals(
            p_norm, px, py, empty, n_gray, n_labels, n_angles, eps
        )
    if "Idmn" in needed and px_suby is not None:
        scale = (ng_scale ** 2)[:, None, None]
        out["Idmn"] = _nanmean_angles(
            (px_suby / (1.0 + (k_diff[None, :, None] ** 2) / scale)).sum(axis=1)
        )
    if "Id" in needed and px_suby is not None:
        out["Id"] = _nanmean_angles((px_suby / (1.0 + k_diff[None, :, None])).sum(axis=1))
    if "Idn" in needed and px_suby is not None:
        scale = ng_scale[:, None, None]
        out["Idn"] = _nanmean_angles(
            (px_suby / (1.0 + k_diff[None, :, None] / scale)).sum(axis=1)
        )
    if "InverseVariance" in needed and px_suby is not None:
        out["InverseVariance"] = _nanmean_angles(
            (px_suby[:, 1:, :] / (k_diff[None, 1:, None] ** 2)).sum(axis=1)
        )
    if "MaximumProbability" in needed:
        out["MaximumProbability"] = _nanmean_angles(p_norm.max(axis=(1, 2)))
    if "SumAverage" in needed and px_addy is not None:
        out["SumAverage"] = _nanmean_angles((k_sum[None, :, None] * px_addy).sum(axis=1))
    if "SumEntropy" in needed and px_addy is not None:
        out["SumEntropy"] = _nanmean_angles(
            (-1.0) * (px_addy * np.log2(px_addy + eps)).sum(axis=1)
        )
    if "SumSquares" in needed:
        out["SumSquares"] = _nanmean_angles(
            (p_norm * ((i_grid[None, :, :, None] - ux) ** 2)).sum(axis=(1, 2))
        )
    return out


def glrlm_features(
    p_glrlm: np.ndarray,
    feature_names: Sequence[str],
    *,
    gray_levels: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Evaluate GLRLM features on stacked run-length count matrices.

    Args:
        p_glrlm: ``[K, Ng, Nr, Na]`` integer run counts (column 0 = length 1).
        feature_names: Enabled GLRLM feature names.
        gray_levels: Optional 1-indexed gray-level values of length ``Ng``.

    Returns:
        Dict[str, np.ndarray]: Feature name -> ``[K]`` values.
    """
    p_mat = _as_float(p_glrlm)
    if p_mat.ndim != 4:
        raise ValueError(f"GLRLM matrix must be [K, Ng, Nr, Na]; got {p_mat.shape}")
    n_labels, n_gray, n_run, _n_angles = p_mat.shape
    ivector = _gray_vector(n_gray, gray_levels)
    needed_early = set(feature_names)
    c_rlm_names = [name for name in feature_names if name in GLRLM_FORMULA_COLUMNS]
    if c_rlm_names:
        try:
            from habit.kernels.radiomics.cext import (
                cext_backend,
                glrlm_formulas as c_glrlm_formulas,
            )

            if cext_backend() == "native":
                packed = c_glrlm_formulas(p_mat, ivector)
                name_to_col = {name: idx for idx, name in enumerate(GLRLM_FORMULA_COLUMNS)}
                out_c: Dict[str, np.ndarray] = {}
                finite_c = False
                for name in c_rlm_names:
                    col = np.asarray(packed[:, name_to_col[name]], dtype=np.float64)
                    out_c[name] = col
                    if np.isfinite(col).any():
                        finite_c = True
                leftover = needed_early - set(c_rlm_names)
                if finite_c and not leftover:
                    return out_c
                if finite_c:
                    # Keep C values; numpy only fills names the C kernel omits.
                    feature_names = [name for name in feature_names if name not in out_c]
                    if not feature_names:
                        return out_c
                    # Merge after the numpy pass below.
                    _c_partial = out_c
                else:
                    _c_partial = None
            else:
                _c_partial = None
        except Exception:
            _c_partial = None
    else:
        _c_partial = None
    jvector = np.arange(1, n_run + 1, dtype=np.float64)
    i2 = ivector ** 2
    j2 = jvector ** 2
    pr = p_mat.sum(axis=1)
    pg = p_mat.sum(axis=2)
    nr = p_mat.sum(axis=(1, 2))
    nr = nr.astype(np.float64, copy=False)
    nr[nr == 0] = np.nan
    needed = set(feature_names)
    out: Dict[str, np.ndarray] = {}

    if "ShortRunEmphasis" in needed:
        out["ShortRunEmphasis"] = _nanmean_angles(
            (pr / j2[None, :, None]).sum(axis=1) / nr
        )
    if "LongRunEmphasis" in needed:
        out["LongRunEmphasis"] = _nanmean_angles(
            (pr * j2[None, :, None]).sum(axis=1) / nr
        )
    if "GrayLevelNonUniformity" in needed:
        out["GrayLevelNonUniformity"] = _nanmean_angles((pg ** 2).sum(axis=1) / nr)
    if "GrayLevelNonUniformityNormalized" in needed:
        out["GrayLevelNonUniformityNormalized"] = _nanmean_angles(
            (pg ** 2).sum(axis=1) / (nr ** 2)
        )
    if "RunLengthNonUniformity" in needed:
        out["RunLengthNonUniformity"] = _nanmean_angles((pr ** 2).sum(axis=1) / nr)
    if "RunLengthNonUniformityNormalized" in needed:
        out["RunLengthNonUniformityNormalized"] = _nanmean_angles(
            (pr ** 2).sum(axis=1) / (nr ** 2)
        )
    if "RunPercentage" in needed:
        n_p = (pr * jvector[None, :, None]).sum(axis=1)
        out["RunPercentage"] = _nanmean_angles(nr / n_p)
    if "GrayLevelVariance" in needed:
        pg_n = pg / nr[:, None, :]
        u_i = (pg_n * ivector[None, :, None]).sum(axis=1, keepdims=True)
        out["GrayLevelVariance"] = _nanmean_angles(
            (pg_n * (ivector[None, :, None] - u_i) ** 2).sum(axis=1)
        )
    if "RunVariance" in needed:
        pr_n = pr / nr[:, None, :]
        u_j = (pr_n * jvector[None, :, None]).sum(axis=1, keepdims=True)
        out["RunVariance"] = _nanmean_angles(
            (pr_n * (jvector[None, :, None] - u_j) ** 2).sum(axis=1)
        )
    if "RunEntropy" in needed:
        p_n = p_mat / nr[:, None, None, :]
        out["RunEntropy"] = _nanmean_angles(
            (-1.0) * (p_n * np.log2(p_n + _EPS)).sum(axis=(1, 2))
        )
    if "LowGrayLevelRunEmphasis" in needed:
        out["LowGrayLevelRunEmphasis"] = _nanmean_angles(
            (pg / i2[None, :, None]).sum(axis=1) / nr
        )
    if "HighGrayLevelRunEmphasis" in needed:
        out["HighGrayLevelRunEmphasis"] = _nanmean_angles(
            (pg * i2[None, :, None]).sum(axis=1) / nr
        )
    if needed & {
        "ShortRunLowGrayLevelEmphasis",
        "ShortRunHighGrayLevelEmphasis",
        "LongRunLowGrayLevelEmphasis",
        "LongRunHighGrayLevelEmphasis",
    }:
        i2_b = i2[None, :, None, None]
        j2_b = j2[None, None, :, None]
        if "ShortRunLowGrayLevelEmphasis" in needed:
            out["ShortRunLowGrayLevelEmphasis"] = _nanmean_angles(
                (p_mat / (i2_b * j2_b)).sum(axis=(1, 2)) / nr
            )
        if "ShortRunHighGrayLevelEmphasis" in needed:
            out["ShortRunHighGrayLevelEmphasis"] = _nanmean_angles(
                (p_mat * i2_b / j2_b).sum(axis=(1, 2)) / nr
            )
        if "LongRunLowGrayLevelEmphasis" in needed:
            out["LongRunLowGrayLevelEmphasis"] = _nanmean_angles(
                (p_mat * j2_b / i2_b).sum(axis=(1, 2)) / nr
            )
        if "LongRunHighGrayLevelEmphasis" in needed:
            out["LongRunHighGrayLevelEmphasis"] = _nanmean_angles(
                (p_mat * i2_b * j2_b).sum(axis=(1, 2)) / nr
            )
    del n_labels  # used only to document batch width
    if _c_partial:
        merged = dict(_c_partial)
        merged.update(out)
        return merged
    return out


def glszm_features(
    p_glszm: np.ndarray,
    feature_names: Sequence[str],
    *,
    gray_levels: Optional[np.ndarray] = None,
    n_voxels: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Evaluate GLSZM features on stacked zone-size count matrices.

    Args:
        p_glszm: ``[K, Ng, Ns]`` zone counts (column 0 = size 1).
        feature_names: Enabled GLSZM feature names.
        gray_levels: Optional 1-indexed gray-level values of length ``Ng``.
        n_voxels: Per-label voxel counts ``Np`` (ZonePercentage denominator).

    Returns:
        Dict[str, np.ndarray]: Feature name -> ``[K]`` values.
    """
    p_mat = _as_float(p_glszm)
    if p_mat.ndim != 3:
        raise ValueError(f"GLSZM matrix must be [K, Ng, Ns]; got {p_mat.shape}")
    n_gray, n_size = p_mat.shape[1], p_mat.shape[2]
    ivector = _gray_vector(n_gray, gray_levels)
    jvector = np.arange(1, n_size + 1, dtype=np.float64)
    ps = p_mat.sum(axis=1)
    pg = p_mat.sum(axis=2)
    nz = p_mat.sum(axis=(1, 2))
    nz = nz.astype(np.float64, copy=False)
    nz[nz == 0] = np.nan
    np_vox = np.asarray(n_voxels, dtype=np.float64).reshape(-1)
    needed = set(feature_names)
    out: Dict[str, np.ndarray] = {}

    if "SmallAreaEmphasis" in needed:
        out["SmallAreaEmphasis"] = (ps / (jvector[None, :] ** 2)).sum(axis=1) / nz
    if "LargeAreaEmphasis" in needed:
        out["LargeAreaEmphasis"] = (ps * (jvector[None, :] ** 2)).sum(axis=1) / nz
    if "GrayLevelNonUniformity" in needed:
        out["GrayLevelNonUniformity"] = (pg ** 2).sum(axis=1) / nz
    if "GrayLevelNonUniformityNormalized" in needed:
        out["GrayLevelNonUniformityNormalized"] = (pg ** 2).sum(axis=1) / (nz ** 2)
    if "SizeZoneNonUniformity" in needed:
        out["SizeZoneNonUniformity"] = (ps ** 2).sum(axis=1) / nz
    if "SizeZoneNonUniformityNormalized" in needed:
        out["SizeZoneNonUniformityNormalized"] = (ps ** 2).sum(axis=1) / (nz ** 2)
    if "ZonePercentage" in needed:
        out["ZonePercentage"] = nz / np_vox
    if "GrayLevelVariance" in needed:
        pg_n = pg / nz[:, None]
        u_i = (pg_n * ivector[None, :]).sum(axis=1, keepdims=True)
        out["GrayLevelVariance"] = (pg_n * (ivector[None, :] - u_i) ** 2).sum(axis=1)
    if "ZoneVariance" in needed:
        ps_n = ps / nz[:, None]
        u_j = (ps_n * jvector[None, :]).sum(axis=1, keepdims=True)
        out["ZoneVariance"] = (ps_n * (jvector[None, :] - u_j) ** 2).sum(axis=1)
    if "ZoneEntropy" in needed:
        p_n = p_mat / nz[:, None, None]
        out["ZoneEntropy"] = (-1.0) * (p_n * np.log2(p_n + _EPS)).sum(axis=(1, 2))
    if "LowGrayLevelZoneEmphasis" in needed:
        out["LowGrayLevelZoneEmphasis"] = (pg / (ivector[None, :] ** 2)).sum(axis=1) / nz
    if "HighGrayLevelZoneEmphasis" in needed:
        out["HighGrayLevelZoneEmphasis"] = (pg * (ivector[None, :] ** 2)).sum(axis=1) / nz
    if "SmallAreaLowGrayLevelEmphasis" in needed:
        out["SmallAreaLowGrayLevelEmphasis"] = (
            p_mat / ((ivector[None, :, None] ** 2) * (jvector[None, None, :] ** 2))
        ).sum(axis=(1, 2)) / nz
    if "SmallAreaHighGrayLevelEmphasis" in needed:
        out["SmallAreaHighGrayLevelEmphasis"] = (
            p_mat * (ivector[None, :, None] ** 2) / (jvector[None, None, :] ** 2)
        ).sum(axis=(1, 2)) / nz
    if "LargeAreaLowGrayLevelEmphasis" in needed:
        out["LargeAreaLowGrayLevelEmphasis"] = (
            p_mat * (jvector[None, None, :] ** 2) / (ivector[None, :, None] ** 2)
        ).sum(axis=(1, 2)) / nz
    if "LargeAreaHighGrayLevelEmphasis" in needed:
        out["LargeAreaHighGrayLevelEmphasis"] = (
            p_mat * (ivector[None, :, None] ** 2) * (jvector[None, None, :] ** 2)
        ).sum(axis=(1, 2)) / nz
    return out


def gldm_features(
    p_gldm: np.ndarray,
    feature_names: Sequence[str],
    *,
    gray_levels: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Evaluate GLDM features on stacked dependence-count matrices.

    Args:
        p_gldm: ``[K, Ng, Nd]`` dependence counts (column 0 = dependence 1).
        feature_names: Enabled GLDM feature names.
        gray_levels: Optional 1-indexed gray-level values of length ``Ng``.

    Returns:
        Dict[str, np.ndarray]: Feature name -> ``[K]`` values.
    """
    p_mat = _as_float(p_gldm)
    if p_mat.ndim != 3:
        raise ValueError(f"GLDM matrix must be [K, Ng, Nd]; got {p_mat.shape}")
    n_gray, n_dep = p_mat.shape[1], p_mat.shape[2]
    ivector = _gray_vector(n_gray, gray_levels)
    jvector = np.arange(1, n_dep + 1, dtype=np.float64)
    pd = p_mat.sum(axis=1)
    pg = p_mat.sum(axis=2)
    nz = p_mat.sum(axis=(1, 2))
    nz = nz.astype(np.float64, copy=False)
    nz[nz == 0] = np.nan
    needed = set(feature_names)
    out: Dict[str, np.ndarray] = {}

    if "SmallDependenceEmphasis" in needed:
        out["SmallDependenceEmphasis"] = (pd / (jvector[None, :] ** 2)).sum(axis=1) / nz
    if "LargeDependenceEmphasis" in needed:
        out["LargeDependenceEmphasis"] = (pd * (jvector[None, :] ** 2)).sum(axis=1) / nz
    if "GrayLevelNonUniformity" in needed:
        out["GrayLevelNonUniformity"] = (pg ** 2).sum(axis=1) / nz
    if "DependenceNonUniformity" in needed:
        out["DependenceNonUniformity"] = (pd ** 2).sum(axis=1) / nz
    if "DependenceNonUniformityNormalized" in needed:
        out["DependenceNonUniformityNormalized"] = (pd ** 2).sum(axis=1) / (nz ** 2)
    if "GrayLevelVariance" in needed:
        pg_n = pg / nz[:, None]
        u_i = (pg_n * ivector[None, :]).sum(axis=1, keepdims=True)
        out["GrayLevelVariance"] = (pg_n * (ivector[None, :] - u_i) ** 2).sum(axis=1)
    if "DependenceVariance" in needed:
        pd_n = pd / nz[:, None]
        u_j = (pd_n * jvector[None, :]).sum(axis=1, keepdims=True)
        out["DependenceVariance"] = (pd_n * (jvector[None, :] - u_j) ** 2).sum(axis=1)
    if "DependenceEntropy" in needed:
        p_n = p_mat / nz[:, None, None]
        out["DependenceEntropy"] = (-1.0) * (p_n * np.log2(p_n + _EPS)).sum(axis=(1, 2))
    if "LowGrayLevelEmphasis" in needed:
        out["LowGrayLevelEmphasis"] = (pg / (ivector[None, :] ** 2)).sum(axis=1) / nz
    if "HighGrayLevelEmphasis" in needed:
        out["HighGrayLevelEmphasis"] = (pg * (ivector[None, :] ** 2)).sum(axis=1) / nz
    if "SmallDependenceLowGrayLevelEmphasis" in needed:
        out["SmallDependenceLowGrayLevelEmphasis"] = (
            p_mat / ((ivector[None, :, None] ** 2) * (jvector[None, None, :] ** 2))
        ).sum(axis=(1, 2)) / nz
    if "SmallDependenceHighGrayLevelEmphasis" in needed:
        out["SmallDependenceHighGrayLevelEmphasis"] = (
            p_mat * (ivector[None, :, None] ** 2) / (jvector[None, None, :] ** 2)
        ).sum(axis=(1, 2)) / nz
    if "LargeDependenceLowGrayLevelEmphasis" in needed:
        out["LargeDependenceLowGrayLevelEmphasis"] = (
            p_mat * (jvector[None, None, :] ** 2) / (ivector[None, :, None] ** 2)
        ).sum(axis=(1, 2)) / nz
    if "LargeDependenceHighGrayLevelEmphasis" in needed:
        out["LargeDependenceHighGrayLevelEmphasis"] = (
            p_mat * (ivector[None, :, None] ** 2) * (jvector[None, None, :] ** 2)
        ).sum(axis=(1, 2)) / nz
    return out


def ngtdm_features(
    p_ngtdm: np.ndarray,
    feature_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    """
    Evaluate NGTDM features on stacked ``[K, Ng, 3]`` matrices.

    Column 0 is ``n_i``, column 1 is ``s_i``, column 2 is the gray-level value.

    Args:
        p_ngtdm: C-extension NGTDM batch.
        feature_names: Enabled NGTDM feature names.

    Returns:
        Dict[str, np.ndarray]: Feature name -> ``[K]`` values.
    """
    p_mat = _as_float(p_ngtdm)
    if p_mat.ndim != 3 or p_mat.shape[2] != 3:
        raise ValueError(f"NGTDM matrix must be [K, Ng, 3]; got {p_mat.shape}")
    n_i = p_mat[:, :, 0]
    s_i = p_mat[:, :, 1]
    ivector = p_mat[:, :, 2]
    nvp = n_i.sum(axis=1)
    nvp_safe = nvp.copy()
    nvp_safe[nvp_safe == 0] = 1.0
    p_i = n_i / nvp_safe[:, None]
    ngp = (n_i > 0).sum(axis=1).astype(np.float64)
    p_zero = p_i == 0
    needed = set(feature_names)
    out: Dict[str, np.ndarray] = {}

    if "Coarseness" in needed:
        coarse = (p_i * s_i).sum(axis=1)
        result = np.empty_like(coarse)
        nonzero = coarse != 0
        result[nonzero] = 1.0 / coarse[nonzero]
        result[~nonzero] = 1e6
        out["Coarseness"] = result
    if "Contrast" in needed:
        div = ngp * (ngp - 1.0)
        contrast = (
            (p_i[:, :, None] * p_i[:, None, :] * (ivector[:, :, None] - ivector[:, None, :]) ** 2).sum(
                axis=(1, 2)
            )
            * s_i.sum(axis=1)
            / nvp_safe
        )
        contrast = np.divide(contrast, div, out=np.zeros_like(contrast), where=div != 0)
        out["Contrast"] = contrast
    if "Busyness" in needed:
        i_pi = ivector * p_i
        absdiff = np.abs(i_pi[:, :, None] - i_pi[:, None, :])
        absdiff[p_zero[:, :, None] | p_zero[:, None, :]] = 0.0
        denom = absdiff.sum(axis=(1, 2))
        busyness = (p_i * s_i).sum(axis=1)
        busyness = np.divide(busyness, denom, out=np.zeros_like(busyness), where=denom != 0)
        out["Busyness"] = busyness
    if "Complexity" in needed:
        pi_si = p_i * s_i
        numerator = pi_si[:, :, None] + pi_si[:, None, :]
        numerator[p_zero[:, :, None] | p_zero[:, None, :]] = 0.0
        divisor = p_i[:, :, None] + p_i[:, None, :]
        divisor[divisor == 0] = 1.0
        out["Complexity"] = (
            np.abs(ivector[:, :, None] - ivector[:, None, :]) * numerator / divisor
        ).sum(axis=(1, 2)) / nvp_safe
    if "Strength" in needed:
        strength_num = (
            (p_i[:, :, None] + p_i[:, None, :])
            * (ivector[:, :, None] - ivector[:, None, :]) ** 2
        )
        strength_num[p_zero[:, :, None] | p_zero[:, None, :]] = 0.0
        s_sum = s_i.sum(axis=1)
        strength = strength_num.sum(axis=(1, 2))
        strength = np.divide(strength, s_sum, out=np.zeros_like(strength), where=s_sum != 0)
        out["Strength"] = strength
    return out


def entropy_uniformity_from_disc(
    discretized: np.ndarray,
    sv_map: np.ndarray,
    labels: Sequence[int],
    n_gray: int,
    feature_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    """
    Entropy / Uniformity from the same discretized bins as the texture matrices.

    Args:
        discretized: 1-indexed bin volume (union- or per-label digitize).
        sv_map: Multi-label map aligned with ``discretized``.
        labels: Label ids in output order.
        n_gray: Maximum gray index (``Ng``).
        feature_names: Subset of ``Entropy`` / ``Uniformity``.

    Returns:
        Dict[str, np.ndarray]: Requested histogram features, shape ``[K]``.
    """
    label_ids = [int(v) for v in labels]
    n_labels = len(label_ids)
    needed = set(feature_names)
    out: Dict[str, np.ndarray] = {
        name: np.full(n_labels, np.nan, dtype=np.float64)
        for name in needed
        if name in ("Entropy", "Uniformity")
    }
    if not out:
        return out
    map_i = np.asarray(sv_map, dtype=np.int32).ravel()
    disc_i = np.asarray(discretized, dtype=np.int32).ravel()
    valid = (map_i > 0) & (disc_i > 0)
    if not np.any(valid):
        return out
    max_lab = int(map_i.max())
    stride = int(n_gray) + 1
    combined = map_i[valid].astype(np.int64) * stride + disc_i[valid].astype(np.int64)
    hist = np.bincount(combined, minlength=(max_lab + 1) * stride)
    hist = hist.reshape(max_lab + 1, stride)
    for row, lab in enumerate(label_ids):
        if lab < 0 or lab > max_lab:
            continue
        counts = hist[lab, 1 : n_gray + 1]
        n = float(counts.sum())
        if n <= 0.0:
            continue
        p_i = counts.astype(np.float64) / n
        pos = p_i > 0.0
        if "Entropy" in out:
            out["Entropy"][row] = float((-p_i[pos] * np.log2(p_i[pos] + _EPS)).sum())
        if "Uniformity" in out:
            out["Uniformity"][row] = float((p_i[pos] ** 2).sum())
    return out


def firstorder_from_cext_stats(
    stats: np.ndarray,
    feature_names: Sequence[str],
    *,
    n_voxels: np.ndarray,
    voxel_array_shift: float,
) -> Dict[str, np.ndarray]:
    """
    Map ``calculate_firstorder`` columns onto requested names.

    ``Variance`` is not stored in the 17-column C buffer; it is recovered
    from Energy / Mean / n as ``Energy/n - (Mean + shift)^2``.

    Args:
        stats: ``[K, 17]`` from ``calculate_firstorder``.
        feature_names: Requested first-order names.
        n_voxels: Per-label voxel counts.
        voxel_array_shift: Same shift used to build Energy.

    Returns:
        Dict[str, np.ndarray]: Feature name -> ``[K]``. Entropy / Uniformity
        are omitted (those must come from the shared discretize).
    """
    arr = np.asarray(stats, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < len(FIRSTORDER_CEXT_COLUMNS):
        raise ValueError(f"firstorder C stats must be [K, 17]; got {arr.shape}")
    name_to_idx = {name: idx for idx, name in enumerate(FIRSTORDER_CEXT_COLUMNS)}
    skip = {"Entropy", "Uniformity"}
    out: Dict[str, np.ndarray] = {}
    for name in feature_names:
        if name in skip:
            continue
        if name == "Variance":
            energy = arr[:, name_to_idx["Energy"]]
            mean = arr[:, name_to_idx["Mean"]]
            n = np.asarray(n_voxels, dtype=np.float64).reshape(-1)
            safe = np.maximum(n, 1.0)
            var = energy / safe - (mean + float(voxel_array_shift)) ** 2
            var[n <= 0] = np.nan
            var[var < 0.0] = 0.0
            out["Variance"] = var
            continue
        if name in name_to_idx:
            out[name] = arr[:, name_to_idx[name]]
    return out


def firstorder_features(
    image: np.ndarray,
    sv_map: np.ndarray,
    labels: Sequence[int],
    feature_names: Sequence[str],
    *,
    discretized: Optional[np.ndarray] = None,
    voxel_array_shift: float = 0.0,
    voxel_volume: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Evaluate first-order features per label from the raw intensity crop.

    Energy / TotalEnergy / RMS use ``voxel_array_shift`` and ``voxel_volume``
    exactly as PyRadiomics. Entropy / Uniformity use ``discretized`` (the
    same bins as the texture matrices) when provided.

    Args:
        image: Raw intensity volume (float).
        sv_map: Multi-label map aligned with ``image``.
        labels: Label ids in output order.
        feature_names: Enabled first-order names.
        discretized: Optional 1-indexed union- or per-label bin image.
        voxel_array_shift: Intensity offset for Energy / TotalEnergy / RMS.
        voxel_volume: ``prod(spacing)`` for TotalEnergy.

    Returns:
        Dict[str, np.ndarray]: Feature name -> ``[K]`` values.
    """
    image_f = np.asarray(image, dtype=np.float64).ravel()
    map_i = np.asarray(sv_map).ravel()
    label_ids = [int(v) for v in labels]
    n_labels = len(label_ids)
    needed = set(feature_names)
    out: Dict[str, np.ndarray] = {name: np.full(n_labels, np.nan, dtype=np.float64) for name in needed}
    disc_flat = None if discretized is None else np.asarray(discretized).ravel()
    shift = float(voxel_array_shift)
    volume = float(voxel_volume)
    # One argsort so each label is a contiguous slice instead of 100 boolean scans.
    order = np.argsort(map_i, kind="stable")
    sorted_labs = map_i[order]
    sorted_vals = image_f[order]
    sorted_disc = None if disc_flat is None else disc_flat[order]
    starts = np.searchsorted(sorted_labs, label_ids, side="left")
    stops = np.searchsorted(sorted_labs, label_ids, side="right")

    for row, label in enumerate(label_ids):
        voxels = sorted_vals[int(starts[row]) : int(stops[row])]
        if voxels.size == 0:
            continue
        n = float(voxels.size)
        mean = float(voxels.mean())
        shifted = voxels + shift
        energy = float(np.dot(shifted, shifted))
        if "Energy" in needed:
            out["Energy"][row] = energy
        if "TotalEnergy" in needed:
            out["TotalEnergy"][row] = energy * volume
        if "Minimum" in needed:
            out["Minimum"][row] = float(voxels.min())
        if "Maximum" in needed:
            out["Maximum"][row] = float(voxels.max())
        if "Mean" in needed:
            out["Mean"][row] = mean
        if "Median" in needed:
            out["Median"][row] = float(np.median(voxels))
        if "Range" in needed:
            out["Range"][row] = float(voxels.max() - voxels.min())
        if "10Percentile" in needed:
            out["10Percentile"][row] = float(np.percentile(voxels, 10))
        if "90Percentile" in needed:
            out["90Percentile"][row] = float(np.percentile(voxels, 90))
        if "InterquartileRange" in needed:
            q75, q25 = np.percentile(voxels, [75, 25])
            out["InterquartileRange"][row] = float(q75 - q25)
        if "MeanAbsoluteDeviation" in needed:
            out["MeanAbsoluteDeviation"][row] = float(np.mean(np.abs(voxels - mean)))
        if "RobustMeanAbsoluteDeviation" in needed:
            p10, p90 = np.percentile(voxels, [10, 90])
            subset = voxels[(voxels >= p10) & (voxels <= p90)]
            if subset.size:
                out["RobustMeanAbsoluteDeviation"][row] = float(
                    np.mean(np.abs(subset - subset.mean()))
                )
        if "RootMeanSquared" in needed:
            out["RootMeanSquared"][row] = float(np.sqrt(energy / n))
        if "Variance" in needed:
            out["Variance"][row] = float(np.mean((voxels - mean) ** 2))
        if "Skewness" in needed or "Kurtosis" in needed:
            centered = voxels - mean
            m2 = float(np.mean(centered ** 2))
            if m2 == 0.0:
                if "Skewness" in needed:
                    out["Skewness"][row] = 0.0
                if "Kurtosis" in needed:
                    out["Kurtosis"][row] = 0.0
            else:
                if "Skewness" in needed:
                    out["Skewness"][row] = float(np.mean(centered ** 3) / (m2 ** 1.5))
                if "Kurtosis" in needed:
                    out["Kurtosis"][row] = float(np.mean(centered ** 4) / (m2 ** 2.0))
        if ("Entropy" in needed or "Uniformity" in needed) and sorted_disc is not None:
            bins = sorted_disc[int(starts[row]) : int(stops[row])]
            bins = bins[bins > 0]
            if bins.size:
                _, counts = np.unique(bins, return_counts=True)
                p_i = counts.astype(np.float64) / float(bins.size)
                if "Entropy" in needed:
                    out["Entropy"][row] = float((-p_i * np.log2(p_i + _EPS)).sum())
                if "Uniformity" in needed:
                    out["Uniformity"][row] = float((p_i ** 2).sum())
    return out


def feature_column_name(feature_class: str, feature_name: str, image_name: str) -> str:
    """Build a PyRadiomics-style ``original_{class}_{name}[-suffix]`` column."""
    col = f"original_{feature_class}_{feature_name}"
    if image_name:
        col = f"{col}-{image_name}"
    return col


def assign_feature_rows(
    rows: Sequence[Mapping[str, object]],
    feature_class: str,
    values: Mapping[str, np.ndarray],
    image_name: str,
) -> None:
    """
    Write batched formula outputs into per-label row dicts.

    Args:
        rows: Mutable row mappings (must already contain ``supervoxel_id``).
        feature_class: PyRadiomics class name.
        values: Feature name -> ``[K]`` array.
        image_name: Optional column suffix.
    """
    for feature_name, array in values.items():
        col = feature_column_name(feature_class, feature_name, image_name)
        flat = np.asarray(array, dtype=np.float64).reshape(-1)
        for idx, row in enumerate(rows):
            dest = row if isinstance(row, dict) else None
            if dest is None:
                continue
            dest[col] = float(flat[idx]) if idx < flat.size else float("nan")


def assign_feature_columns(
    columns: Dict[str, object],
    feature_class: str,
    values: Mapping[str, np.ndarray],
    image_name: str,
) -> None:
    """
    Write batched formula outputs as DataFrame columns (no per-row Python loop).

    Args:
        columns: Mutable mapping that already contains ``supervoxel_id``.
        feature_class: PyRadiomics class name.
        values: Feature name -> ``[K]`` array.
        image_name: Optional column suffix.
    """
    for feature_name, array in values.items():
        col = feature_column_name(feature_class, feature_name, image_name)
        columns[col] = np.asarray(array, dtype=np.float64).reshape(-1)
