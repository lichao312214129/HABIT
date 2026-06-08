# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
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