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
from __future__ import annotations

from typing import List, Tuple, Union

import numpy
import torch
from radiomics import base


class TorchRadiomicsBase(base.RadiomicsFeaturesBase):
  """
  Shared PyTorch helpers for vendored torchradiomics feature classes.
  """
  def __init__(self, inputImage, inputMask, **kwargs):
    super(TorchRadiomicsBase, self).__init__(inputImage, inputMask, **kwargs)

    self.dtype = kwargs.get("dtype", torch.float64)
    self.device = kwargs.get("device", "cuda")
  
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