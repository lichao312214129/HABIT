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
# ---------------------------------------------------------------------------
# Third-party attribution
#
# This file is derived from PyTorchRadiomics and adapted for HABIT:
#
#     https://github.com/lyhyl/pytorchradiomics
#     Copyright (c) 2024 lyhyl
#     Licensed under the MIT License
#
# The original MIT copyright and permission notice are reproduced in
# LICENSE in this directory and in NOTICE at the project root.
# HABIT modifications remain under Apache-2.0.
# ---------------------------------------------------------------------------
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple, Union

import numpy
import torch
from radiomics import base

# Reuse one PyRadiomics discretisation across firstorder + texture classes
# in the same voxel execute(). Key is image/mask content + bin settings,
# so the returned matrix and Ng / grayLevels stay bit-identical to
# RadiomicsFeaturesBase._applyBinning. Bounded so a reused extractor
# cannot grow without limit across subjects.
_DISCRETIZE_CACHE: Dict[Tuple[Any, ...], Tuple[numpy.ndarray, numpy.ndarray, int]] = {}
_DISCRETIZE_CACHE_MAX = 4


def _array_md5(array: numpy.ndarray) -> str:
  """Stable fingerprint of array bytes for the discretisation cache."""
  contig = numpy.ascontiguousarray(array)
  return hashlib.md5(contig.tobytes()).hexdigest()


class TorchRadiomicsBase(base.RadiomicsFeaturesBase):
  """
  Shared PyTorch helpers for vendored torchradiomics feature classes.
  """
  def __init__(self, inputImage, inputMask, **kwargs):
    super(TorchRadiomicsBase, self).__init__(inputImage, inputMask, **kwargs)

    self.dtype = kwargs.get("dtype", torch.float64)
    self.device = kwargs.get("device", "cuda")

  def _applyBinning(self, matrix: numpy.ndarray) -> numpy.ndarray:
    """
    Discretise ``matrix`` once per unique image/mask/bin setting.

    Args:
        matrix: Image intensities before gray-level discretisation.

    Returns:
        numpy.ndarray: Same integer image ``RadiomicsFeaturesBase._applyBinning``
        would return. ``self.coefficients['grayLevels']`` and ``Ng`` are
        set to the cached values on a hit.
    """
    key = (
        tuple(int(size) for size in matrix.shape),
        str(matrix.dtype),
        _array_md5(matrix),
        _array_md5(numpy.asarray(self.maskArray, dtype=numpy.uint8)),
        self.settings.get("binWidth"),
        self.settings.get("binCount"),
    )
    hit = _DISCRETIZE_CACHE.get(key)
    if hit is None:
      binned = super(TorchRadiomicsBase, self)._applyBinning(matrix)
      hit = (
          binned,
          numpy.asarray(self.coefficients["grayLevels"]),
          int(self.coefficients["Ng"]),
      )
      if len(_DISCRETIZE_CACHE) >= _DISCRETIZE_CACHE_MAX:
        _DISCRETIZE_CACHE.clear()
      _DISCRETIZE_CACHE[key] = hit
      return binned
    binned, gray_levels, ng = hit
    # Same array object as the first class so GPU centre-grid reuse can
    # key on id() instead of hashing the volume again. Classes do not
    # write imageArray after discretisation.
    self.coefficients["grayLevels"] = gray_levels
    self.coefficients["Ng"] = ng
    return binned

  def tensor(self, array: numpy.ndarray) -> torch.Tensor:
    return torch.tensor(array, dtype=self.dtype, device=self.device)

  def gray_level_index(
      self,
      gray_levels: numpy.ndarray,
      reference: torch.Tensor,
  ) -> torch.Tensor:
    """
    Convert PyRadiomics 1-based gray levels to torch.long indices.

    NumPy int64 advanced indexing on CUDA tensors fails on recent PyTorch builds.
    """
    return torch.as_tensor(
        gray_levels - 1,
        dtype=torch.long,
        device=reference.device,
    )
  
  def delete(
      self,
      arr: torch.Tensor,
      ind: Union[int, Tuple, List],
      dim: int,
  ) -> torch.Tensor:
    """
    https://gist.github.com/velikodniy/6efef837e67aee2e7152eb5900eb0258
    """
    # skip = [i for i in range(arr.size(dim)) if i != ind]
    if isinstance(ind, int):
      skip = [i for i in range(arr.size(dim)) if i != ind]
    elif isinstance(ind, (tuple, list)): # torch.where
      lst: torch.Tensor = ind[0]
      lst = lst.cpu().numpy()
      skip = [i for i in range(arr.size(dim)) if i not in lst]
    else:
      raise TypeError("ind wrong type")
    indices = [slice(None) if i != dim else skip for i in range(arr.ndim)]
    return arr.__getitem__(indices)