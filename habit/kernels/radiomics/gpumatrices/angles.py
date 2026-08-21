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
Exact Python port of the angle generation in PyRadiomics' C extension.

The functions here mirror ``get_angle_count`` and ``build_angles`` from
``radiomics/src/cmatrices.c`` (PyRadiomics v3.0.1) so that GPU matrix
implementations produce the identical angle set, in the identical order,
as ``radiomics.cMatrices``. Any deviation in angle order would silently
permute the last axis of the texture matrices and break numerical parity.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def get_angle_count(
    size: Sequence[int],
    distances: Sequence[int],
    force2Ddimension: int = -1,
    bidirectional: bool = False,
) -> int:
    """
    Number of neighbour offsets ("angles") PyRadiomics generates.

    Port of ``get_angle_count`` in ``cmatrices.c``. For each requested
    infinity-norm distance ``d`` the count of offsets at exactly that
    distance is ``prod(2d+1) - prod(2d-1)`` over the in-plane dimensions,
    with each factor clamped when ``d`` exceeds the image size along that
    dimension. Mono-directional extraction (GLCM/GLRLM/NGTDM/GLDM) halves
    the total; GLSZM passes ``bidirectional=True`` and keeps all offsets.

    Args:
        size: Image shape per dimension (numpy order, e.g. ``(z, y, x)``).
        distances: Requested integer distances (infinity norm).
        force2Ddimension: Dimension excluded from angle generation when
            ``force2D`` is enabled; pass ``-1`` for full 3D.
        bidirectional: Keep both directions of every angle.

    Returns:
        int: Number of angles ``Na`` that :func:`build_angles` will emit.

    Raises:
        ValueError: If any distance is smaller than 1.
    """
    nd = len(size)
    na = 0
    for dist in distances:
        dist = int(dist)
        if dist < 1:
            raise ValueError(f"distances must be >= 1; got {dist}")
        na_d = 1  # offsets with infinity norm in [0, d]
        na_dd = 1  # offsets with infinity norm in [0, d - 1]
        for dim in range(nd):
            if dim == force2Ddimension:
                continue
            if dist < int(size[dim]):
                na_d *= 2 * dist + 1
                na_dd *= 2 * dist - 1
            else:
                # Distance covers the whole dimension: both bounds clamp to
                # the maximum offset the dimension supports.
                na_d *= 2 * (int(size[dim]) - 1) + 1
                na_dd *= 2 * (int(size[dim]) - 1) + 1
        na += na_d - na_dd
    if not bidirectional:
        na //= 2
    return na


def build_angles(
    size: Sequence[int],
    distances: Sequence[int],
    force2Ddimension: int = -1,
    bidirectional: bool = False,
) -> np.ndarray:
    """
    Generate the neighbour offsets exactly as PyRadiomics' C code does.

    Port of ``build_angles`` in ``cmatrices.c``. Offsets are enumerated with
    a mixed-radix counter where the LAST dimension cycles fastest and each
    dimension runs from ``+max_distance`` down to ``-max_distance``. An
    offset is kept when it is non-zero, fits inside the image, does not move
    along the ``force2Ddimension``, and its infinity norm is one of the
    requested ``distances``. Enumeration stops once ``Na`` angles are found,
    which for mono-directional extraction keeps the first half of the full
    symmetric set.

    Args:
        size: Image shape per dimension (numpy order, e.g. ``(z, y, x)``).
        distances: Requested integer distances (infinity norm).
        force2Ddimension: Dimension excluded from angle generation when
            ``force2D`` is enabled; pass ``-1`` for full 3D.
        bidirectional: Keep both directions of every angle.

    Returns:
        np.ndarray: ``int64`` array of shape ``(Na, Nd)`` with the offset
        vector of every angle, in PyRadiomics emission order.
    """
    size_i = [int(s) for s in size]
    distances_i = [int(d) for d in distances]
    nd = len(size_i)
    na = get_angle_count(size_i, distances_i, force2Ddimension, bidirectional)

    max_distance = max(distances_i)
    n_offsets = 2 * max_distance + 1

    # Mixed-radix strides: last dimension cycles fastest (stride 1), each
    # preceding dimension is slower by a factor of n_offsets.
    offset_stride = [1] * nd
    for dim in range(nd - 2, -1, -1):
        offset_stride[dim] = offset_stride[dim + 1] * n_offsets

    angles = []
    new_a_idx = 0
    max_iterations = n_offsets ** nd + 1
    while len(angles) < na:
        if new_a_idx >= max_iterations:
            raise RuntimeError(
                "build_angles exhausted all offset combinations before "
                f"reaching the expected angle count {na}; get_angle_count "
                "and build_angles are out of sync."
            )
        a_dist = 0
        offsets = []
        valid = True
        for dim in range(nd):
            offset = max_distance - (new_a_idx // offset_stride[dim]) % n_offsets
            if (
                (dim == force2Ddimension and offset != 0)
                or offset >= size_i[dim]
                or offset <= -size_i[dim]
            ):
                valid = False
                break
            offsets.append(offset)
            a_dist = max(a_dist, abs(offset))
        new_a_idx += 1
        if not valid or a_dist < 1:
            continue
        if a_dist in distances_i:
            angles.append(offsets)

    return np.asarray(angles, dtype=np.int64).reshape(na, nd)
