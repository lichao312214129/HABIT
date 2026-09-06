# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Sparse COO features vs dense HABIT gold (float64)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from habit.kernels.radiomics.gpumatrices.glcm import calculate_glcm, calculate_glcm_coo
from habit.kernels.radiomics.gpumatrices.glrlm import calculate_glrlm, calculate_glrlm_coo
from habit.kernels.radiomics.gpumatrices.glszm import calculate_glszm, calculate_glszm_coo
from habit.kernels.radiomics.gpumatrices.sparse_coo import (
    glcm_features_from_coo,
    glrlm_features_from_coo,
    glszm_features_from_coo,
    voxel_nr_cap,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _case(rng: np.random.Generator, shape, ng: int, n_list: int, radius: int):
    image = rng.integers(1, ng + 1, size=shape).astype(np.int32)
    mask = rng.random(shape) > 0.25
    if int(mask.sum()) < n_list + 4:
        mask.ravel()[: n_list + 4] = True
    zyx = np.argwhere(mask)
    pick = rng.choice(len(zyx), size=min(n_list, len(zyx)), replace=False)
    coords = np.ascontiguousarray(zyx[pick].T.astype(np.int32))
    return image, mask, coords, radius


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if not np.any(ok):
        return 0.0
    denom = np.maximum(np.abs(a[ok]), 1e-12)
    return float(np.max(np.abs(a[ok] - b[ok]) / denom))


def _dense_glcm_feats(p: torch.Tensor) -> dict:
    from habit.kernels.radiomics.gpumatrices.sparse_coo import SparseGLCM

    # Gold = dense HABIT P converted to COO, then the same 21-feature
    # reduction (including P+P.T). Tests count parity + feature math.
    dtype = p.dtype
    device = p.device
    nz = torch.nonzero(p, as_tuple=False)
    if nz.numel() == 0:
        empty = SparseGLCM(
            voxel=torch.empty(0, dtype=torch.long, device=device),
            i_idx=torch.empty(0, dtype=torch.long, device=device),
            j_idx=torch.empty(0, dtype=torch.long, device=device),
            angle=torch.empty(0, dtype=torch.long, device=device),
            count=torch.empty(0, dtype=dtype, device=device),
            n_vox=int(p.shape[0]),
            ng=int(p.shape[1]),
            na=int(p.shape[3]),
        )
        return glcm_features_from_coo(empty, symmetrical=True)
    gold = SparseGLCM(
        voxel=nz[:, 0].to(torch.long),
        i_idx=nz[:, 1].to(torch.long),
        j_idx=nz[:, 2].to(torch.long),
        angle=nz[:, 3].to(torch.long),
        count=p[nz[:, 0], nz[:, 1], nz[:, 2], nz[:, 3]].to(dtype),
        n_vox=int(p.shape[0]),
        ng=int(p.shape[1]),
        na=int(p.shape[3]),
    )
    return glcm_features_from_coo(gold, symmetrical=True)


@pytest.mark.parametrize("device", DEVICES)
def test_glcm_coo_matches_dense(device: str) -> None:
    rng = np.random.default_rng(0)
    image, mask, coords, radius = _case(rng, (12, 11, 10), ng=6, n_list=8, radius=2)
    p, _ = calculate_glcm(
        image, mask, np.asarray([1], dtype=np.int32), 6,
        kernelRadius=radius, voxelCoordinates=coords, device=device,
        dtype=torch.float64,
    )
    coo, _ = calculate_glcm_coo(
        image, mask, np.asarray([1], dtype=np.int32), 6,
        kernelRadius=radius, voxelCoordinates=coords, device=device,
        dtype=torch.float64,
    )
    sparse = glcm_features_from_coo(coo, symmetrical=True)
    gold = _dense_glcm_feats(p)
    for name in sparse:
        assert _max_rel(gold[name], sparse[name]) < 1e-10, name


@pytest.mark.parametrize("device", DEVICES)
def test_glrlm_coo_matches_dense(device: str) -> None:
    rng = np.random.default_rng(1)
    image, mask, coords, radius = _case(rng, (10, 9, 8), ng=5, n_list=6, radius=2)
    nr = voxel_nr_cap(radius)
    p, _ = calculate_glrlm(
        image, mask, 5, int(max(image.shape)),
        kernelRadius=radius, voxelCoordinates=coords, device=device,
        dtype=torch.float64,
    )
    # Gold features from the protocol-Nr dense matrix; trailing zeros
    # must not change the 16 features versus window-capped COO.
    from habit.kernels.radiomics.gpumatrices.sparse_coo import SparseGLRLM

    nz = torch.nonzero(p, as_tuple=False)
    gold_coo = SparseGLRLM(
        voxel=nz[:, 0].to(torch.long),
        gray=nz[:, 1].to(torch.long),
        run=nz[:, 2].to(torch.long),
        angle=nz[:, 3].to(torch.long),
        count=p[nz[:, 0], nz[:, 1], nz[:, 2], nz[:, 3]].to(torch.float64),
        n_vox=int(p.shape[0]),
        ng=int(p.shape[1]),
        na=int(p.shape[3]),
        nr_cap=int(p.shape[2]),
    )
    gold = glrlm_features_from_coo(gold_coo)
    coo, _ = calculate_glrlm_coo(
        image, mask, 5, Nr=nr,
        kernelRadius=radius, voxelCoordinates=coords, device=device,
        dtype=torch.float64,
    )
    sparse = glrlm_features_from_coo(coo)
    for name in sparse:
        assert _max_rel(gold[name], sparse[name]) < 1e-8, name


@pytest.mark.parametrize("device", DEVICES)
def test_glszm_coo_matches_dense(device: str) -> None:
    rng = np.random.default_rng(2)
    image, mask, coords, radius = _case(rng, (10, 9, 8), ng=5, n_list=6, radius=2)
    p = calculate_glszm(
        image, mask, 5, int(mask.sum()),
        kernelRadius=radius, voxelCoordinates=coords, device=device,
        dtype=torch.float64,
    )
    from habit.kernels.radiomics.gpumatrices.sparse_coo import SparseGLSZM

    nz = torch.nonzero(p, as_tuple=False)
    gold_coo = SparseGLSZM(
        voxel=nz[:, 0].to(torch.long),
        gray=nz[:, 1].to(torch.long),
        size=nz[:, 2].to(torch.long),
        count=p[nz[:, 0], nz[:, 1], nz[:, 2]].to(torch.float64),
        n_vox=int(p.shape[0]),
        ng=int(p.shape[1]),
        ns_cap=int(p.shape[2]),
    )
    gold = glszm_features_from_coo(gold_coo)
    coo = calculate_glszm_coo(
        image, mask, 5, int(mask.sum()),
        kernelRadius=radius, voxelCoordinates=coords, device=device,
        dtype=torch.float64,
    )
    sparse = glszm_features_from_coo(coo)
    for name in sparse:
        assert _max_rel(gold[name], sparse[name]) < 1e-8, name
