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
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy
import six
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

  def _use_sparse_coo(self, voxelCoordinates: Optional[numpy.ndarray]) -> bool:
    """True when this class should compute features from sparse COO."""
    from habit.kernels.radiomics.gpumatrices import resolve_use_sparse_matrices

    if voxelCoordinates is None and not self.voxelBased:
      # Segment mode: COO is allowed (one matrix, actual events only).
      pass
    settings = dict(self.settings)
    settings["dtype"] = self.dtype
    settings["enabledFeatures"] = getattr(self, "enabledFeatures", {})
    return resolve_use_sparse_matrices(
      settings, device=self.device, voxel_based=bool(self.voxelBased)
    )

  def _calculateFeatures(self, voxelCoordinates=None):
    """Yield enabled features; prefer a sparse COO cache when present."""
    self._initCalculation(voxelCoordinates)
    cache = getattr(self, "_sparse_features", None)
    self.logger.debug("Calculating features")
    for feature, enabled in six.iteritems(self.enabledFeatures):
      if not enabled:
        continue
      try:
        if cache is not None and feature in cache:
          yield True, feature, cache[feature]
        else:
          yield True, feature, getattr(self, "get%sFeatureValue" % feature)()
      except DeprecationWarning as deprecatedFeature:
        self.logger.debug(
          "Feature %s is deprecated: %s", feature, deprecatedFeature.args[0]
        )
      except Exception:
        import traceback

        self.logger.error("FAILED: %s", traceback.format_exc())
        yield False, feature, numpy.nan

  def _calculateVoxels(self) -> None:
    """
    Voxel loop with one-line per-batch progress and OOM resume.

    A CUDA OOM retries the *failed* batch at half size. Completed
    batches are not recomputed. HU / binWidth are never changed.
    """
    import SimpleITK as sitk

    from habit.utils.torch_radiomics_utils import release_cuda_cache
    from habit.utils.voxel_batch_utils import preflight_texture_batch

    initValue = self.settings.get("initValue", 0)
    voxelBatch = int(self.settings.get("voxelBatch", -1))
    batch_progress = bool(self.settings.get("batch_progress", True))
    ng = int(self.coefficients.get("Ng", 1))
    radius = int(self.settings.get("kernelRadius", 1) if self.voxelBased else 0)
    voxelBatch = preflight_texture_batch(
      ng=ng,
      requested_batch=voxelBatch,
      kernel_radius=radius,
      torch_device=str(self.device),
      sparse=self._use_sparse_coo(
        self.labelledVoxelCoordinates[:, :1]
        if self.labelledVoxelCoordinates.shape[1]
        else None
      ),
      image_nr=int(max(self.imageArray.shape)),
    )

    for feature, enabled in six.iteritems(self.enabledFeatures):
      if enabled:
        self.featureValues[feature] = numpy.full(
          list(self.inputImage.GetSize())[::-1], initValue, dtype="float"
        )

    voxel_count = int(self.labelledVoxelCoordinates.shape[1])
    if voxelBatch < 0:
      voxelBatch = voxel_count
    voxel_batch_idx = 0
    batch_i = 0
    n_batches = int(numpy.ceil(float(voxel_count) / max(voxelBatch, 1)))
    class_name = self.__class__.__name__.replace("TorchRadiomics", "").lower()
    while voxel_batch_idx < voxel_count:
      batch_i += 1
      voxelCoords = self.labelledVoxelCoordinates[
        :, voxel_batch_idx : voxel_batch_idx + voxelBatch
      ]
      t0 = time.perf_counter()
      try:
        for success, featureName, featureValue in self._calculateFeatures(voxelCoords):
          if success:
            self.featureValues[featureName][tuple(voxelCoords)] = featureValue
      except RuntimeError as exc:
        if _is_cuda_oom(exc) and voxelBatch > 1:
          voxelBatch = max(1, voxelBatch // 2)
          self.logger.warning(
            "CUDA OOM at %s batch %d; shrinking voxelBatch to %d and "
            "retrying this batch (completed voxels are kept)",
            class_name,
            batch_i,
            voxelBatch,
          )
          release_cuda_cache()
          batch_i -= 1
          n_batches = batch_i + int(
            numpy.ceil(float(voxel_count - voxel_batch_idx) / voxelBatch)
          )
          continue
        raise
      elapsed = time.perf_counter() - t0
      if batch_progress:
        peak = _peak_allocated_mib()
        msg = (
          f"voxel_radiomics: {class_name} batch {batch_i}/{n_batches} "
          f"{elapsed:.3f}s peak={peak:.0f}MiB"
        )
        print(msg, flush=True)
        self.logger.info(msg)
      voxel_batch_idx += int(voxelCoords.shape[1])

    for feature, enabled in six.iteritems(self.enabledFeatures):
      if enabled:
        self.featureValues[feature] = sitk.GetImageFromArray(self.featureValues[feature])
        self.featureValues[feature].CopyInformation(self.inputImage)

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


def _is_cuda_oom(exc: BaseException) -> bool:
  """True when ``exc`` is a CUDA out-of-memory error."""
  text = str(exc).lower()
  return "out of memory" in text or "cuda oom" in text


def _peak_allocated_mib() -> float:
  """Current CUDA allocated MiB, or 0 when CUDA is unused."""
  try:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
      return float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0)
  except Exception:
    return 0.0
  return 0.0