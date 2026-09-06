# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Sparse COO working store for voxel texture matrices.

Dense ``(B, Ng, Ng, Na)`` / ``(B, Ng, Nr, Na)`` / ``(B, Ng, Ns)`` tensors
are the wrong data structure once Ng is large or Nr is ``max(shape)``.
A kernel window of radius ``r`` produces at most a handful of co-occurrence
pairs, runs, or zones per listed voxel. Those events are stored as COO
rows ``(voxel, gray, j, count)`` and every IBSI feature is a reduction
over that list. The dense gold kernels stay in this package for
parity tests and for callers that opt out of sparse.

GPU texture arithmetic is float64. float32 is opt-in only: sparse f32
GLCM ClusterShade failed a 1e-3 relative check on HCC_041.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


GLCM_SPARSE_FEATURES: Tuple[str, ...] = (
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
    "Idm",
    "Idmn",
    "Id",
    "Idn",
    "InverseVariance",
    "MaximumProbability",
    "SumAverage",
    "SumEntropy",
    "SumSquares",
)
# Eigenvalue / mutual-information GLCM features need a dense joint. The
# voxel path already disables them; requesting one forces the dense kernel.
GLCM_DENSE_ONLY_FEATURES: Tuple[str, ...] = ("MCC", "Imc1", "Imc2")

GLRLM_SPARSE_FEATURES: Tuple[str, ...] = (
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

GLSZM_SPARSE_FEATURES: Tuple[str, ...] = (
    "SmallAreaEmphasis",
    "LargeAreaEmphasis",
    "GrayLevelNonUniformity",
    "GrayLevelNonUniformityNormalized",
    "SizeZoneNonUniformity",
    "SizeZoneNonUniformityNormalized",
    "ZonePercentage",
    "GrayLevelVariance",
    "ZoneVariance",
    "ZoneEntropy",
    "LowGrayLevelZoneEmphasis",
    "HighGrayLevelZoneEmphasis",
    "SmallAreaLowGrayLevelEmphasis",
    "SmallAreaHighGrayLevelEmphasis",
    "LargeAreaLowGrayLevelEmphasis",
    "LargeAreaHighGrayLevelEmphasis",
)


@dataclass
class SparseGLCM:
    """COO GLCM: one row per nonzero ``(voxel, i, j, angle)``.

    ``i_idx`` / ``j_idx`` are 0-based gray indices; gray-level *values*
    are ``i_idx + 1`` (PyRadiomics 1..Ng). ``count`` holds raw
    co-occurrence counts (exact small integers) in the compute dtype.
    """

    voxel: torch.Tensor
    i_idx: torch.Tensor
    j_idx: torch.Tensor
    angle: torch.Tensor
    count: torch.Tensor
    n_vox: int
    ng: int
    na: int


@dataclass
class SparseGLRLM:
    """COO GLRLM: one row per nonzero ``(voxel, gray, run, angle)``.

    ``gray`` and ``run`` are 0-based. Run length value is ``run + 1``.
    """

    voxel: torch.Tensor
    gray: torch.Tensor
    run: torch.Tensor
    angle: torch.Tensor
    count: torch.Tensor
    n_vox: int
    ng: int
    na: int
    nr_cap: int


@dataclass
class SparseGLSZM:
    """COO GLSZM: one row per nonzero ``(voxel, gray, zone_size)``.

    ``gray`` is 0-based. Zone-size value is ``size + 1``.
    """

    voxel: torch.Tensor
    gray: torch.Tensor
    size: torch.Tensor
    count: torch.Tensor
    n_vox: int
    ng: int
    ns_cap: int


def unique_counts(
    keys: torch.Tensor,
    count_dtype: torch.dtype,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reduce integer keys to unique keys and summed weights.

    ``bincount(inverse)`` is only correct when every key is a unit
    event. After ``P + P.T`` the same ``(v, i, j, a)`` can already
    carry count > 1; those weights must be ``index_add_``-ed, not
    recounted as occurrences. CPU torch unique has been wrong on
    large duplicate keys in this repo, so the host path uses numpy
    unique and a host ``index_add``.

    Args:
        keys: 1-D int64 keys.
        count_dtype: Dtype of the returned counts (float64 default).
        weights: Optional per-key weight (same length as ``keys``).
            ``None`` means every key has weight 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(uniq_keys, counts)``.
    """
    if keys.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=keys.device)
        return empty, torch.empty(0, dtype=count_dtype, device=keys.device)
    if keys.is_cuda:
        uniq, inverse = torch.unique(keys, return_inverse=True)
        if weights is None:
            return uniq, torch.bincount(inverse).to(dtype=count_dtype)
        merged = torch.zeros(uniq.numel(), dtype=count_dtype, device=keys.device)
        merged.index_add_(0, inverse, weights.to(dtype=count_dtype))
        return uniq, merged
    keys_np = keys.detach().cpu().numpy()
    uniq_np, inverse = np.unique(keys_np, return_inverse=True)
    if weights is None:
        counts_np = np.bincount(inverse)
    else:
        counts_np = np.zeros(uniq_np.shape[0], dtype=np.float64)
        np.add.at(counts_np, inverse, weights.detach().cpu().numpy().astype(np.float64))
    return (
        torch.as_tensor(uniq_np, dtype=torch.long, device=keys.device),
        torch.as_tensor(counts_np, dtype=count_dtype, device=keys.device),
    )


def pack_glcm_key(
    voxel: torch.Tensor,
    i_idx: torch.Tensor,
    j_idx: torch.Tensor,
    angle: torch.Tensor,
    ng: int,
    na: int,
) -> torch.Tensor:
    """Pack ``(v, i, j, a)`` into one int64 C-order key."""
    return ((voxel * int(ng) + i_idx) * int(ng) + j_idx) * int(na) + angle


def unpack_glcm_key(
    keys: torch.Tensor,
    ng: int,
    na: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of :func:`pack_glcm_key`."""
    na_i = int(ng) * int(na)
    voxel = torch.div(keys, int(ng) * na_i, rounding_mode="floor")
    rem = keys - voxel * (int(ng) * na_i)
    i_idx = torch.div(rem, na_i, rounding_mode="floor")
    rem = rem - i_idx * na_i
    j_idx = torch.div(rem, int(na), rounding_mode="floor")
    angle = rem - j_idx * int(na)
    return voxel, i_idx, j_idx, angle


def symmetrize_glcm_coo(sparse: SparseGLCM) -> SparseGLCM:
    """``P + P.T`` on COO. The diagonal is doubled, matching HABIT."""
    if sparse.count.numel() == 0:
        return sparse
    voxel = torch.cat([sparse.voxel, sparse.voxel], dim=0)
    i_idx = torch.cat([sparse.i_idx, sparse.j_idx], dim=0)
    j_idx = torch.cat([sparse.j_idx, sparse.i_idx], dim=0)
    angle = torch.cat([sparse.angle, sparse.angle], dim=0)
    count = torch.cat([sparse.count, sparse.count], dim=0)
    key = pack_glcm_key(voxel, i_idx, j_idx, angle, sparse.ng, sparse.na)
    uniq, merged = unique_counts(key, count.dtype, weights=count)
    voxel_u, i_u, j_u, a_u = unpack_glcm_key(uniq, sparse.ng, sparse.na)
    return SparseGLCM(
        voxel=voxel_u,
        i_idx=i_u,
        j_idx=j_u,
        angle=a_u,
        count=merged,
        n_vox=sparse.n_vox,
        ng=sparse.ng,
        na=sparse.na,
    )


def _nanmean_angles(mat: torch.Tensor) -> np.ndarray:
    """Mean over the last axis, ignoring NaN (empty angles)."""
    return torch.nanmean(mat, dim=-1).detach().cpu().numpy()


def _mean_angles_propagate_nan(mat: torch.Tensor) -> np.ndarray:
    """HABIT JointAverage: ``mean`` not ``nanmean``. One empty angle ? NaN."""
    return mat.mean(dim=-1).detach().cpu().numpy()


def glcm_features_from_coo(
    sparse: SparseGLCM,
    *,
    symmetrical: bool = True,
) -> Dict[str, np.ndarray]:
    """
    IBSI / HABIT 21 GLCM features from COO. No dense ``(B, Ng, Ng, Na)``.

    Marginals ``pxAddy`` / ``pxSuby`` are linear in Ng. Empty angles are
    NaN and dropped by ``nanmean``, matching TorchRadiomicsGLCM.

    Args:
        sparse: Raw (un-normalised) COO counts.
        symmetrical: When True, apply HABIT ``P + P.T`` first.

    Returns:
        Dict[str, np.ndarray]: Per-voxel feature vectors, float64 host.
    """
    if symmetrical:
        sparse = symmetrize_glcm_coo(sparse)
    device = sparse.count.device
    dtype = sparse.count.dtype
    eps = torch.finfo(dtype).eps
    n_vox, ng, na = sparse.n_vox, sparse.ng, sparse.na
    empty = np.full((n_vox,), np.nan, dtype=np.float64)
    if n_vox == 0:
        return {name: empty for name in GLCM_SPARSE_FEATURES}

    i_val = (sparse.i_idx + 1).to(dtype)
    j_val = (sparse.j_idx + 1).to(dtype)
    va = sparse.voxel * na + sparse.angle
    sum_p_flat = torch.zeros(n_vox * na, dtype=dtype, device=device)
    if sparse.count.numel() > 0:
        sum_p_flat.index_add_(0, va, sparse.count)
    sum_p = sum_p_flat.view(n_vox, na)
    if sparse.count.numel() == 0:
        nan_mat = torch.full((n_vox, na), torch.nan, dtype=dtype, device=device)
        out = {name: _nanmean_angles(nan_mat) for name in GLCM_SPARSE_FEATURES}
        return out

    sum_flat = sum_p_flat[va]
    valid = sum_flat > 0
    prob = torch.zeros_like(sparse.count)
    prob[valid] = sparse.count[valid] / sum_flat[valid]

    def _reduce(values: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(n_vox * na, dtype=dtype, device=device)
        out.index_add_(0, va, values)
        out = out.view(n_vox, na)
        out[sum_p == 0] = torch.nan
        return out

    ac = _reduce(prob * i_val * j_val)
    ux = _reduce(prob * i_val)
    uy = _reduce(prob * j_val)
    ux_e = ux[sparse.voxel, sparse.angle]
    uy_e = uy[sparse.voxel, sparse.angle]
    contrast = _reduce(prob * (i_val - j_val) ** 2)
    energy = _reduce(prob * prob)
    entropy = _reduce(-prob * torch.log2(prob + eps))
    maxprob = torch.zeros(n_vox * na, dtype=dtype, device=device)
    maxprob.scatter_reduce_(0, va, prob, reduce="amax", include_self=True)
    maxprob = maxprob.view(n_vox, na)
    maxprob[sum_p == 0] = torch.nan
    ss = _reduce(prob * (i_val - ux_e) ** 2)
    sigx = _reduce(prob * (i_val - ux_e) ** 2).clamp_min(0.0).sqrt()
    sigy = _reduce(prob * (j_val - uy_e) ** 2).clamp_min(0.0).sqrt()
    corm = _reduce(prob * (i_val - ux_e) * (j_val - uy_e))
    corr = corm / (sigx * sigy + eps)
    corr[sigx * sigy == 0] = 1.0
    delta = i_val + j_val - ux_e - uy_e
    cp = _reduce(prob * delta ** 4)
    cs = _reduce(prob * delta ** 3)
    ct = _reduce(prob * delta ** 2)

    px_sub = torch.zeros(n_vox, ng, na, dtype=dtype, device=device)
    px_add = torch.zeros(n_vox, 2 * ng - 1, na, dtype=dtype, device=device)
    d_idx = (sparse.i_idx - sparse.j_idx).abs()
    s_idx = sparse.i_idx + sparse.j_idx
    px_sub.view(-1).index_add_(
        0, (sparse.voxel * ng + d_idx) * na + sparse.angle, sparse.count
    )
    px_add.view(-1).index_add_(
        0, (sparse.voxel * (2 * ng - 1) + s_idx) * na + sparse.angle, sparse.count
    )
    px_sub = px_sub / sum_p[:, None, :]
    px_add = px_add / sum_p[:, None, :]
    k_diff = torch.arange(ng, dtype=dtype, device=device)
    k_sum = torch.arange(2, 2 * ng + 1, dtype=dtype, device=device)
    diffavg = (k_diff[None, :, None] * px_sub).sum(1)
    difent = -(px_sub * torch.log2(px_sub + eps)).sum(1)
    diffvar = (px_sub * (k_diff[None, :, None] - diffavg[:, None, :]) ** 2).sum(1)
    ng_t = torch.as_tensor(float(ng), dtype=dtype, device=device)
    idm = (px_sub / (1.0 + k_diff[None, :, None] ** 2)).sum(1)
    idmn = (px_sub / (1.0 + (k_diff[None, :, None] ** 2) / (ng_t ** 2))).sum(1)
    inv_diff = (px_sub / (1.0 + k_diff[None, :, None])).sum(1)
    idn = (px_sub / (1.0 + k_diff[None, :, None] / ng_t)).sum(1)
    inv_var = (px_sub[:, 1:, :] / (k_diff[None, 1:, None] ** 2)).sum(1)
    sumavg = (k_sum[None, :, None] * px_add).sum(1)
    sument = -(px_add * torch.log2(px_add + eps)).sum(1)

    return {
        "Autocorrelation": _nanmean_angles(ac),
        "JointAverage": _mean_angles_propagate_nan(ux),
        "ClusterProminence": _nanmean_angles(cp),
        "ClusterShade": _nanmean_angles(cs),
        "ClusterTendency": _nanmean_angles(ct),
        "Contrast": _nanmean_angles(contrast),
        "Correlation": _nanmean_angles(corr),
        "DifferenceAverage": _nanmean_angles(diffavg),
        "DifferenceEntropy": _nanmean_angles(difent),
        "DifferenceVariance": _nanmean_angles(diffvar),
        "JointEnergy": _nanmean_angles(energy),
        "JointEntropy": _nanmean_angles(entropy),
        "Idm": _nanmean_angles(idm),
        "Idmn": _nanmean_angles(idmn),
        "Id": _nanmean_angles(inv_diff),
        "Idn": _nanmean_angles(idn),
        "InverseVariance": _nanmean_angles(inv_var),
        "MaximumProbability": _nanmean_angles(maxprob),
        "SumAverage": _nanmean_angles(sumavg),
        "SumEntropy": _nanmean_angles(sument),
        "SumSquares": _nanmean_angles(ss),
    }


def glrlm_features_from_coo(sparse: SparseGLRLM) -> Dict[str, np.ndarray]:
    """
    16 IBSI / HABIT GLRLM features from COO. No dense ``(B, Ng, Nr, Na)``.

    ``pr`` / ``pg`` are coefficient tensors (linear in Ng or Nr), not P.
    Empty angles become NaN and are ignored by ``nanmean``.

    Args:
        sparse: Raw run-count COO.

    Returns:
        Dict[str, np.ndarray]: Per-voxel feature vectors.
    """
    device = sparse.count.device
    dtype = sparse.count.dtype
    n_vox, ng, na = sparse.n_vox, sparse.ng, sparse.na
    nr_axis = max(int(sparse.nr_cap), 1)
    if sparse.count.numel() > 0:
        nr_axis = max(nr_axis, int(sparse.run.max().item()) + 1)
    eps = torch.finfo(dtype).eps

    nr_flat = torch.zeros(n_vox * na, dtype=dtype, device=device)
    pg = torch.zeros(n_vox, ng, na, dtype=dtype, device=device)
    pr = torch.zeros(n_vox, nr_axis, na, dtype=dtype, device=device)
    if sparse.count.numel() > 0:
        va = sparse.voxel * na + sparse.angle
        nr_flat.index_add_(0, va, sparse.count)
        pg.view(-1).index_add_(
            0, (sparse.voxel * ng + sparse.gray) * na + sparse.angle, sparse.count
        )
        pr.view(-1).index_add_(
            0,
            (sparse.voxel * nr_axis + sparse.run) * na + sparse.angle,
            sparse.count,
        )
        i_val = (sparse.gray + 1).to(dtype)
        j_val = (sparse.run + 1).to(dtype)
        i2 = i_val * i_val
        j2 = j_val * j_val
        # Joint reductions live on the COO rows, never on a dense P.
        joint_den = torch.zeros(n_vox * na, dtype=dtype, device=device)
        joint_srlgle = torch.zeros_like(joint_den)
        joint_srhgle = torch.zeros_like(joint_den)
        joint_lrlgle = torch.zeros_like(joint_den)
        joint_lrhgle = torch.zeros_like(joint_den)
        joint_ent = torch.zeros_like(joint_den)
        joint_den.index_add_(0, va, sparse.count)
        p_row = sparse.count / joint_den[va].clamp_min(eps)
        joint_srlgle.index_add_(0, va, sparse.count / (i2 * j2))
        joint_srhgle.index_add_(0, va, sparse.count * i2 / j2)
        joint_lrlgle.index_add_(0, va, sparse.count * j2 / i2)
        joint_lrhgle.index_add_(0, va, sparse.count * i2 * j2)
        joint_ent.index_add_(0, va, -p_row * torch.log2(p_row + eps))
        srlgle = joint_srlgle.view(n_vox, na)
        srhgle = joint_srhgle.view(n_vox, na)
        lrlgle = joint_lrlgle.view(n_vox, na)
        lrhgle = joint_lrhgle.view(n_vox, na)
        run_ent = joint_ent.view(n_vox, na)
    else:
        srlgle = srhgle = lrlgle = lrhgle = run_ent = torch.zeros(
            n_vox, na, dtype=dtype, device=device
        )

    nr = nr_flat.view(n_vox, na)
    nr = torch.where(nr == 0, torch.nan, nr)
    jvector = torch.arange(1, nr_axis + 1, dtype=dtype, device=device)
    ivector = torch.arange(1, ng + 1, dtype=dtype, device=device)
    i2v = ivector * ivector
    j2v = jvector * jvector
    sre = (pr / j2v[None, :, None]).sum(1) / nr
    lre = (pr * j2v[None, :, None]).sum(1) / nr
    gln = (pg ** 2).sum(1) / nr
    glnn = (pg ** 2).sum(1) / (nr ** 2)
    rln = (pr ** 2).sum(1) / nr
    rlnn = (pr ** 2).sum(1) / (nr ** 2)
    np_run = (pr * jvector[None, :, None]).sum(1)
    rp = nr / np_run
    pg_n = pg / nr[:, None, :]
    u_i = (pg_n * ivector[None, :, None]).sum(1, keepdim=True)
    glv = (pg_n * (ivector[None, :, None] - u_i) ** 2).sum(1)
    pr_n = pr / nr[:, None, :]
    u_j = (pr_n * jvector[None, :, None]).sum(1, keepdim=True)
    rv = (pr_n * (jvector[None, :, None] - u_j) ** 2).sum(1)
    lglre = (pg / i2v[None, :, None]).sum(1) / nr
    hglre = (pg * i2v[None, :, None]).sum(1) / nr
    srlgle = srlgle / nr
    srhgle = srhgle / nr
    lrlgle = lrlgle / nr
    lrhgle = lrhgle / nr
    run_ent = torch.where(nr == 0, torch.nan, run_ent)

    return {
        "ShortRunEmphasis": _nanmean_angles(sre),
        "LongRunEmphasis": _nanmean_angles(lre),
        "GrayLevelNonUniformity": _nanmean_angles(gln),
        "GrayLevelNonUniformityNormalized": _nanmean_angles(glnn),
        "RunLengthNonUniformity": _nanmean_angles(rln),
        "RunLengthNonUniformityNormalized": _nanmean_angles(rlnn),
        "RunPercentage": _nanmean_angles(rp),
        "GrayLevelVariance": _nanmean_angles(glv),
        "RunVariance": _nanmean_angles(rv),
        "RunEntropy": _nanmean_angles(run_ent),
        "LowGrayLevelRunEmphasis": _nanmean_angles(lglre),
        "HighGrayLevelRunEmphasis": _nanmean_angles(hglre),
        "ShortRunLowGrayLevelEmphasis": _nanmean_angles(srlgle),
        "ShortRunHighGrayLevelEmphasis": _nanmean_angles(srhgle),
        "LongRunLowGrayLevelEmphasis": _nanmean_angles(lrlgle),
        "LongRunHighGrayLevelEmphasis": _nanmean_angles(lrhgle),
    }


def glszm_features_from_coo(sparse: SparseGLSZM) -> Dict[str, np.ndarray]:
    """
    16 IBSI / HABIT GLSZM features from COO. No dense ``(B, Ng, Ns)``.

    Empty voxels: ``Nz`` is set to 1 (HABIT TorchRadiomicsGLSZM), so
    ratios become 0 rather than NaN.

    Args:
        sparse: Raw zone-count COO.

    Returns:
        Dict[str, np.ndarray]: Per-voxel feature vectors.
    """
    device = sparse.count.device
    dtype = sparse.count.dtype
    n_vox, ng = sparse.n_vox, sparse.ng
    ns_axis = max(int(sparse.ns_cap), 1)
    if sparse.count.numel() > 0:
        ns_axis = max(ns_axis, int(sparse.size.max().item()) + 1)
    eps = torch.finfo(dtype).eps

    nz = torch.zeros(n_vox, dtype=dtype, device=device)
    np_vox = torch.zeros(n_vox, dtype=dtype, device=device)
    pg = torch.zeros(n_vox, ng, dtype=dtype, device=device)
    ps = torch.zeros(n_vox, ns_axis, dtype=dtype, device=device)
    zone_ent = torch.zeros(n_vox, dtype=dtype, device=device)
    salgle = torch.zeros(n_vox, dtype=dtype, device=device)
    sahgle = torch.zeros(n_vox, dtype=dtype, device=device)
    lalgle = torch.zeros(n_vox, dtype=dtype, device=device)
    lahgle = torch.zeros(n_vox, dtype=dtype, device=device)
    if sparse.count.numel() > 0:
        i_val = (sparse.gray + 1).to(dtype)
        j_val = (sparse.size + 1).to(dtype)
        i2 = i_val * i_val
        j2 = j_val * j_val
        nz.index_add_(0, sparse.voxel, sparse.count)
        np_vox.index_add_(0, sparse.voxel, sparse.count * j_val)
        pg.view(-1).index_add_(0, sparse.voxel * ng + sparse.gray, sparse.count)
        ps.view(-1).index_add_(0, sparse.voxel * ns_axis + sparse.size, sparse.count)
        nz_row = nz[sparse.voxel].clamp_min(eps)
        p_row = sparse.count / nz_row
        zone_ent.index_add_(0, sparse.voxel, -p_row * torch.log2(p_row + eps))
        salgle.index_add_(0, sparse.voxel, sparse.count / (i2 * j2))
        sahgle.index_add_(0, sparse.voxel, sparse.count * i2 / j2)
        lalgle.index_add_(0, sparse.voxel, sparse.count * j2 / i2)
        lahgle.index_add_(0, sparse.voxel, sparse.count * i2 * j2)

    # HABIT dense: Nz==0 ? 1 so empty kernels yield 0, not NaN.
    nz_safe = torch.where(nz == 0, torch.ones_like(nz), nz)
    np_safe = torch.where(np_vox == 0, torch.ones_like(np_vox), np_vox)
    ivector = torch.arange(1, ng + 1, dtype=dtype, device=device)
    jvector = torch.arange(1, ns_axis + 1, dtype=dtype, device=device)
    i2v = ivector * ivector
    j2v = jvector * jvector
    pg_n = pg / nz_safe[:, None]
    u_i = (pg_n * ivector[None, :]).sum(1, keepdim=True)
    ps_n = ps / nz_safe[:, None]
    u_j = (ps_n * jvector[None, :]).sum(1, keepdim=True)

    def _host(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float64, copy=False)

    return {
        "SmallAreaEmphasis": _host((ps / j2v[None, :]).sum(1) / nz_safe),
        "LargeAreaEmphasis": _host((ps * j2v[None, :]).sum(1) / nz_safe),
        "GrayLevelNonUniformity": _host((pg ** 2).sum(1) / nz_safe),
        "GrayLevelNonUniformityNormalized": _host((pg ** 2).sum(1) / (nz_safe ** 2)),
        "SizeZoneNonUniformity": _host((ps ** 2).sum(1) / nz_safe),
        "SizeZoneNonUniformityNormalized": _host((ps ** 2).sum(1) / (nz_safe ** 2)),
        "ZonePercentage": _host(nz_safe / np_safe),
        "GrayLevelVariance": _host((pg_n * (ivector[None, :] - u_i) ** 2).sum(1)),
        "ZoneVariance": _host((ps_n * (jvector[None, :] - u_j) ** 2).sum(1)),
        "ZoneEntropy": _host(zone_ent),
        "LowGrayLevelZoneEmphasis": _host((pg / i2v[None, :]).sum(1) / nz_safe),
        "HighGrayLevelZoneEmphasis": _host((pg * i2v[None, :]).sum(1) / nz_safe),
        "SmallAreaLowGrayLevelEmphasis": _host(salgle / nz_safe),
        "SmallAreaHighGrayLevelEmphasis": _host(sahgle / nz_safe),
        "LargeAreaLowGrayLevelEmphasis": _host(lalgle / nz_safe),
        "LargeAreaHighGrayLevelEmphasis": _host(lahgle / nz_safe),
    }


def estimate_sparse_working_bytes(
    ng: int,
    batch: int,
    kernel_radius: int,
    n_angles: int = 13,
) -> int:
    """
    Conservative peak working-set for the sparse voxel GPU path.

    Does not include the input volume. Never used to clip HU or change
    ``binWidth`` / ``binCount``.

    Args:
        ng: Gray-level axis (max gray inside the ROI).
        batch: Listed voxels in one batch.
        kernel_radius: Voxel kernel radius ``r``.
        n_angles: Mono-directional angle count (13 in 3D).

    Returns:
        int: Estimated bytes.
    """
    window = (2 * int(kernel_radius) + 1) ** 3
    b = int(batch)
    ng_i = max(int(ng), 1)
    na = max(int(n_angles), 1)
    glcm_events = b * window * na
    glcm = glcm_events * 8 * 6
    glrlm_sort = window * b * na * 8 * 8
    glszm = window * b * 8 * 6
    marg = b * (2 * ng_i) * na * 8 * 2
    return int(max(glcm, glrlm_sort, glszm) + marg)


def estimate_dense_working_bytes(
    ng: int,
    batch: int,
    nr: int,
    n_angles: int = 13,
) -> int:
    """Peak bytes if the caller forced the dense gold matrices."""
    b = int(batch)
    ng_i = max(int(ng), 1)
    na = max(int(n_angles), 1)
    glcm = b * ng_i * ng_i * na * 8 * 2
    glrlm = b * ng_i * max(int(nr), 1) * na * 8
    return int(max(glcm, glrlm))


def voxel_nr_cap(kernel_radius: int) -> int:
    """Longest run that can exist inside a ``[-r, r]`` window."""
    return 2 * max(int(kernel_radius), 0) + 1


def voxel_ns_cap(kernel_radius: int, nd: int = 3) -> int:
    """Largest zone that can exist inside a ``[-r, r]^nd`` window."""
    side = 2 * max(int(kernel_radius), 0) + 1
    return int(side ** int(nd))
