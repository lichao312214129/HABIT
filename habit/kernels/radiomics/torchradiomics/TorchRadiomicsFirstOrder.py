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
import numpy
import torch
from radiomics import base, cMatrices, deprecated

from .TorchRadiomicsBase import TorchRadiomicsBase


class TorchRadiomicsFirstOrder(TorchRadiomicsBase):
  """
  RadiomicsFirstOrder PyTorch implement.

  All per-batch work (kernel-window gathering, the gray-level histogram and
  the feature math itself) runs on ``self.device`` as float64 tensors. The
  getters still return numpy arrays, so results stay drop-in compatible with
  the PyRadiomics reference implementation.
  """

  def __init__(self, inputImage, inputMask, **kwargs):
    # Device mirrors must exist before super().__init__(), because the base
    # constructor already triggers _initVoxelBasedCalculation().
    self._early_device = kwargs.get("device", "cuda")
    self._early_dtype = kwargs.get("dtype", torch.float64)
    self._image_t = None
    self._disc_t = None
    self._kernel_offsets_t = None
    self._target_t = None
    self._target_np = None

    super(TorchRadiomicsFirstOrder, self).__init__(inputImage, inputMask, **kwargs)

    self.dtype = kwargs.get("dtype", torch.float64)
    self.device = kwargs.get("device", "cuda")

    self.pixelSpacing = inputImage.GetSpacing()
    self.voxelArrayShift = kwargs.get('voxelArrayShift', 0)
    self.discretizedImageArray = self._applyBinning(self.imageArray.copy())

  def _dev(self):
    # self.device is assigned only after the base constructor ran; fall back
    # to the constructor kwarg during _initVoxelBasedCalculation().
    return getattr(self, "device", self._early_device)

  def _dt(self):
    return getattr(self, "dtype", self._early_dtype)

  @property
  def targetVoxelArray(self) -> numpy.ndarray:
    """
    numpy mirror of the per-voxel kernel windows, shape (Nvox, Nk).

    The canonical copy lives on the GPU (``_target_t``); the numpy view is
    rebuilt lazily so the voxel hot path never pays for a device-to-host
    copy. External consumers (e.g. supervoxel batching) may also assign a
    plain numpy array, which is uploaded to the device by the setter.
    """
    if self._target_np is None and self._target_t is not None:
      self._target_np = self._target_t.cpu().numpy()
    return self._target_np

  @targetVoxelArray.setter
  def targetVoxelArray(self, value: numpy.ndarray) -> None:
    if value is None:
      self._target_np = None
      self._target_t = None
      return
    arr = numpy.ascontiguousarray(value, dtype='float')
    self._target_np = arr
    self._target_t = torch.as_tensor(arr, dtype=self._dt(), device=self._dev())

  def _initVoxelBasedCalculation(self):
    super(TorchRadiomicsFirstOrder, self)._initVoxelBasedCalculation()

    kernelRadius = self.settings.get('kernelRadius', 1)

    # Get the size of the input, which depends on whether it is in masked mode or not
    if self.masked:
      size = numpy.max(self.labelledVoxelCoordinates, 1) - numpy.min(self.labelledVoxelCoordinates, 1) + 1
    else:
      size = numpy.array(self.imageArray.shape)

    # Take the minimum size along each dimension from either the size of the ROI or the kernel
    boundingBoxSize = numpy.minimum(size, kernelRadius * 2 + 1)

    # Calculate the offsets, which can be used to generate a list of kernel Coordinates. Shape (Nd, Nk)
    self.kernelOffsets = cMatrices.generate_angles(boundingBoxSize,
                                                   numpy.array(range(1, kernelRadius + 1)),
                                                   True,  # Bi-directional
                                                   self.settings.get('force2D', False),
                                                   self.settings.get('force2Ddimension', 0))
    self.kernelOffsets = numpy.append(self.kernelOffsets, [[0, 0, 0]], axis=0)  # add center voxel
    self.kernelOffsets = self.kernelOffsets.transpose((1, 0))

    self.imageArray = self.imageArray.astype('float')
    self.imageArray[~self.maskArray] = numpy.nan
    self.imageArray = numpy.pad(self.imageArray,
                                pad_width=self.settings.get('kernelRadius', 1),
                                mode='constant', constant_values=numpy.nan)
    self.maskArray = numpy.pad(self.maskArray,
                               pad_width=self.settings.get('kernelRadius', 1),
                               mode='constant', constant_values=False)

    # Upload the padded image and kernel offsets once; neither is modified
    # afterwards, so per-batch work is a pure on-device gather.
    self._image_t = torch.as_tensor(self.imageArray, dtype=self._dt(), device=self._dev())
    self._kernel_offsets_t = torch.as_tensor(
        numpy.ascontiguousarray(self.kernelOffsets), dtype=torch.long, device=self._dev())

  def _initCalculation(self, voxelCoordinates=None):
    device = self._dev()
    dtype = self._dt()

    if self._disc_t is None:
      # discretizedImageArray is created in __init__ (i.e. after the
      # voxel-based padding), so it can only be uploaded lazily here.
      self._disc_t = torch.as_tensor(
          numpy.ascontiguousarray(self.discretizedImageArray), dtype=dtype, device=device)
    if self._image_t is None:
      # Segment-based mode: imageArray was never padded or NaN-masked.
      self._image_t = torch.as_tensor(
          numpy.ascontiguousarray(self.imageArray.astype('float')), dtype=dtype, device=device)

    if voxelCoordinates is None:
      # maskArray may be reassigned between calls (supervoxel batching), so
      # it is uploaded fresh every time; it is a small boolean array.
      mask_t = torch.as_tensor(numpy.ascontiguousarray(self.maskArray), dtype=torch.bool, device=device)
      self._target_t = self._image_t[mask_t].reshape(1, -1)
      self._target_np = None

      # Segment mode histograms the gray levels actually present (like
      # numpy.unique(..., return_counts=True)), not all possible levels.
      disc_masked = self._disc_t[mask_t]
      _, counts = torch.unique(disc_masked, return_counts=True)
      p_i = counts.to(dtype).reshape(1, -1)
    else:
      # voxelCoordinates shape (Nd, Nvox)
      kernelRadius = self.settings.get('kernelRadius', 1)
      coords = numpy.ascontiguousarray(voxelCoordinates + kernelRadius)  # adjust for padding
      coords_t = torch.as_tensor(coords, dtype=torch.long, device=device)
      kernelCoords = self._kernel_offsets_t[:, None, :] + coords_t[:, :, None]  # Shape (Nd, Nvox, Nk)
      kernelCoords = tuple(kernelCoords)  # tuple of Nd tensors, each (Nvox, Nk)

      self._target_t = self._image_t[kernelCoords]  # shape (Nvox, Nk)
      self._target_np = None  # numpy mirror invalidated; rebuilt lazily

      disc_gathered = self._disc_t[kernelCoords]  # shape (Nvox, Nk)

      # Histogram over all possible gray levels. One broadcasted comparison
      # replaces the reference Python loop over gray levels; chunking over
      # the gray-level axis bounds the (Nvox, Nk, block) boolean tensor.
      gl_t = torch.as_tensor(
          numpy.ascontiguousarray(self.coefficients['grayLevels']), dtype=dtype, device=device)
      n_vox, n_k = disc_gathered.shape
      n_gl = int(gl_t.shape[0])
      p_i = torch.empty((n_vox, n_gl), dtype=dtype, device=device)
      block = max(1, (1 << 26) // max(1, n_vox * n_k))
      for g0 in range(0, n_gl, block):
        g1 = min(n_gl, g0 + block)
        # NaN entries compare False, matching numpy.nansum(... == gl, 1).
        p_i[:, g0:g1] = (disc_gathered.unsqueeze(-1) == gl_t[g0:g1]).sum(dim=1, dtype=dtype)

    sumBins = p_i.sum(dim=1, keepdim=True)
    sumBins = torch.where(sumBins == 0, torch.ones_like(sumBins), sumBins)  # Prevent division by 0 errors
    self.coefficients['p_i'] = p_i / sumBins

    self.logger.debug('First order feature class initialized')

  @staticmethod
  def _moment(a: torch.Tensor, moment: int = 1) -> torch.Tensor:
    r"""
    Calculate n-order moment of a tensor along axis 1 (NaN-aware).
    """
    if moment == 1:
      # By definition the first central moment is 0; kept for API parity.
      return torch.zeros(a.shape[0], dtype=a.dtype, device=a.device)
    mn = torch.nanmean(a, dim=1, keepdim=True)
    return torch.nanmean((a - mn) ** moment, dim=1)

  @staticmethod
  def _nanmin(x: torch.Tensor, dim: int) -> torch.Tensor:
    """torch.nanmin equivalent (not exposed by this torch build): NaN entries
    are ignored and all-NaN rows reduce to NaN, like numpy.nanmin."""
    valid = ~torch.isnan(x)
    out = torch.amin(torch.where(valid, x, torch.full_like(x, float('inf'))), dim=dim)
    return torch.where(valid.any(dim=dim), out, torch.full_like(out, float('nan')))

  @staticmethod
  def _nanmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """torch.nanmax equivalent: NaN entries are ignored and all-NaN rows
    reduce to NaN, like numpy.nanmax."""
    valid = ~torch.isnan(x)
    out = torch.amax(torch.where(valid, x, torch.full_like(x, float('-inf'))), dim=dim)
    return torch.where(valid.any(dim=dim), out, torch.full_like(out, float('nan')))

  @staticmethod
  def _nanmedian(x: torch.Tensor, dim: int) -> torch.Tensor:
    """numpy.nanmedian equivalent along ``dim``.

    torch.nanmedian returns the *lower* of the two middle values for an even
    number of valid entries, while numpy averages the two middle values, so
    the median is computed explicitly from the sorted row. torch.sort places
    NaN at the end, so the first ``n`` sorted entries are the valid ones.
    """
    xs, _ = torch.sort(x, dim=dim)
    n = (~torch.isnan(x)).sum(dim=dim, keepdim=True)  # valid count per row
    hi = (n // 2).clamp(min=0)  # upper middle index (middle for odd n)
    lo = ((n - 1) // 2).clamp(min=0)  # lower middle index (== hi for odd n)
    v_lo = xs.gather(dim, lo)
    v_hi = xs.gather(dim, hi)
    med = (v_lo + v_hi) / 2
    # All-NaN rows gathered element 0, which is NaN there anyway.
    return med.squeeze(dim)

  def getEnergyFeatureValue(self):
    r"""
    **1. Energy**

    .. math::
      \textit{energy} = \displaystyle\sum^{N_p}_{i=1}{(\textbf{X}(i) + c)^2}

    Here, :math:`c` is optional value, defined by ``voxelArrayShift``, which shifts the intensities to prevent negative
    values in :math:`\textbf{X}`. This ensures that voxels with the lowest gray values contribute the least to Energy,
    instead of voxels with gray level intensity closest to 0.

    Energy is a measure of the magnitude of voxel values in an image. A larger values implies a greater sum of the
    squares of these values.

    .. note::
      This feature is volume-confounded, a larger value of :math:`c` increases the effect of volume-confounding.
    """
    shiftedParameterArray = self._target_t + self.voxelArrayShift

    return torch.nansum(shiftedParameterArray ** 2, dim=1).cpu().numpy()

  def getTotalEnergyFeatureValue(self):
    r"""
    **2. Total Energy**

    .. math::
      \textit{total energy} = V_{voxel}\displaystyle\sum^{N_p}_{i=1}{(\textbf{X}(i) + c)^2}

    Here, :math:`c` is optional value, defined by ``voxelArrayShift``, which shifts the intensities to prevent negative
    values in :math:`\textbf{X}`. This ensures that voxels with the lowest gray values contribute the least to Energy,
    instead of voxels with gray level intensity closest to 0.

    Total Energy is the value of Energy feature scaled by the volume of the voxel in cubic mm.

    .. note::
      This feature is volume-confounded, a larger value of :math:`c` increases the effect of volume-confounding.

    .. note::
      Not present in IBSI feature definitions
    """
    cubicMMPerVoxel = numpy.multiply.reduce(self.pixelSpacing)

    return self.getEnergyFeatureValue() * cubicMMPerVoxel

  def getEntropyFeatureValue(self):
    r"""
    **3. Entropy**

    .. math::
      \textit{entropy} = -\displaystyle\sum^{N_g}_{i=1}{p(i)\log_2\big(p(i)+\epsilon\big)}

    Here, :math:`\epsilon` is an arbitrarily small positive number (:math:`\approx 2.2\times10^{-16}`).

    Entropy specifies the uncertainty/randomness in the image values. It measures the average amount of information
    required to encode the image values.

    .. note::
      Defined by IBSI as Intensity Histogram Entropy.
    """
    p_i = self.coefficients['p_i']

    eps = numpy.spacing(1)
    return (-1.0 * torch.sum(p_i * torch.log2(p_i + eps), dim=1)).cpu().numpy()

  def getMinimumFeatureValue(self):
    r"""
    **4. Minimum**

    .. math::
      \textit{minimum} = \min(\textbf{X})
    """
    return self._nanmin(self._target_t, 1).cpu().numpy()

  def get10PercentileFeatureValue(self):
    r"""
    **5. 10th percentile**

    The 10\ :sup:`th` percentile of :math:`\textbf{X}`
    """
    return torch.nanquantile(self._target_t, 0.1, 1).cpu().numpy()

  def get90PercentileFeatureValue(self):
    r"""
    **6. 90th percentile**

    The 90\ :sup:`th` percentile of :math:`\textbf{X}`
    """
    return torch.nanquantile(self._target_t, 0.9, 1).cpu().numpy()

  def getMaximumFeatureValue(self):
    r"""
    **7. Maximum**

    .. math::
      \textit{maximum} = \max(\textbf{X})

    The maximum gray level intensity within the ROI.
    """
    return self._nanmax(self._target_t, 1).cpu().numpy()

  def getMeanFeatureValue(self):
    r"""
    **8. Mean**

    .. math::
      \textit{mean} = \frac{1}{N_p}\displaystyle\sum^{N_p}_{i=1}{\textbf{X}(i)}

    The average gray level intensity within the ROI.
    """
    return torch.nanmean(self._target_t, dim=1).cpu().numpy()

  def getMedianFeatureValue(self):
    r"""
    **9. Median**

    The median gray level intensity within the ROI.
    """
    return self._nanmedian(self._target_t, 1).cpu().numpy()

  def getInterquartileRangeFeatureValue(self):
    r"""
    **10. Interquartile Range**

    .. math::
      \textit{interquartile range} = \textbf{P}_{75} - \textbf{P}_{25}

    Here :math:`\textbf{P}_{25}` and :math:`\textbf{P}_{75}` are the 25\ :sup:`th` and 75\ :sup:`th` percentile of the
    image array, respectively.
    """
    return (torch.nanquantile(self._target_t, 0.75, 1)
            - torch.nanquantile(self._target_t, 0.25, 1)).cpu().numpy()

  def getRangeFeatureValue(self):
    r"""
    **11. Range**

    .. math::
      \textit{range} = \max(\textbf{X}) - \min(\textbf{X})

    The range of gray values in the ROI.
    """
    return (self._nanmax(self._target_t, 1) - self._nanmin(self._target_t, 1)).cpu().numpy()

  def getMeanAbsoluteDeviationFeatureValue(self):
    r"""
    **12. Mean Absolute Deviation (MAD)**

    .. math::
      \textit{MAD} = \frac{1}{N_p}\displaystyle\sum^{N_p}_{i=1}{|\textbf{X}(i)-\bar{X}|}

    Mean Absolute Deviation is the mean distance of all intensity values from the Mean value of the image array.
    """
    u_x = torch.nanmean(self._target_t, dim=1, keepdim=True)
    return torch.nanmean(torch.abs(self._target_t - u_x), dim=1).cpu().numpy()

  def getRobustMeanAbsoluteDeviationFeatureValue(self):
    r"""
    **13. Robust Mean Absolute Deviation (rMAD)**

    .. math::
      \textit{rMAD} = \frac{1}{N_{10-90}}\displaystyle\sum^{N_{10-90}}_{i=1}
      {|\textbf{X}_{10-90}(i)-\bar{X}_{10-90}|}

    Robust Mean Absolute Deviation is the mean distance of all intensity values
    from the Mean Value calculated on the subset of image array with gray levels in between, or equal
    to the 10\ :sup:`th` and 90\ :sup:`th` percentile.
    """
    X = self._target_t
    prcnt10 = torch.nanquantile(X, 0.1, 1)
    prcnt90 = torch.nanquantile(X, 0.9, 1)

    # Keep only voxels inside the closed 10-90th percentile range; NaN
    # entries compare False and therefore stay NaN, matching the reference.
    in_range = (X >= prcnt10[:, None]) & (X <= prcnt90[:, None])
    percentileArray = torch.where(in_range, X, torch.full_like(X, float('nan')))

    mean = torch.nanmean(percentileArray, dim=1, keepdim=True)
    return torch.nanmean(torch.abs(percentileArray - mean), dim=1).cpu().numpy()

  def getRootMeanSquaredFeatureValue(self):
    r"""
    **14. Root Mean Squared (RMS)**

    .. math::
      \textit{RMS} = \sqrt{\frac{1}{N_p}\sum^{N_p}_{i=1}{(\textbf{X}(i) + c)^2}}

    Here, :math:`c` is optional value, defined by ``voxelArrayShift``, which shifts the intensities to prevent negative
    values in :math:`\textbf{X}`. This ensures that voxels with the lowest gray values contribute the least to RMS,
    instead of voxels with gray level intensity closest to 0.

    RMS is the square-root of the mean of all the squared intensity values. It is another measure of the magnitude of
    the image values. This feature is volume-confounded, a larger value of :math:`c` increases the effect of
    volume-confounding.
    """
    # If no voxels are segmented, prevent division by 0 and return 0
    if self._target_t.numel() == 0:
      return 0

    shiftedParameterArray = self._target_t + self.voxelArrayShift
    Nvox = torch.sum(~torch.isnan(self._target_t), dim=1).to(self._dt())
    return torch.sqrt(torch.nansum(shiftedParameterArray ** 2, dim=1) / Nvox).cpu().numpy()

  @deprecated
  def getStandardDeviationFeatureValue(self):
    r"""
    **15. Standard Deviation**

    .. math::
      \textit{standard deviation} = \sqrt{\frac{1}{N_p}\sum^{N_p}_{i=1}{(\textbf{X}(i)-\bar{X})^2}}

    Standard deviation measures the amount of variation or dispersion from the Mean value. By definition,
    :math:`\textit{standard deviation} = \sqrt{\textit{variance}}`

    .. note::
      As this feature is correlated with variance, it is marked so it is not enabled by default.
      To include this feature in the extraction, specify it by name in the enabled features
      (i.e. this feature will not be enabled if no individual features are specified (enabling 'all' features),
      but will be enabled when individual features are specified, including this feature).
      Not present in IBSI feature definitions (correlated with variance)
    """
    u_x = torch.nanmean(self._target_t, dim=1, keepdim=True)
    return torch.sqrt(torch.nanmean((self._target_t - u_x) ** 2, dim=1)).cpu().numpy()

  def getSkewnessFeatureValue(self):
    r"""
    **16. Skewness**

    .. math::
      \textit{skewness} = \displaystyle\frac{\mu_3}{\sigma^3} =
      \frac{\frac{1}{N_p}\sum^{N_p}_{i=1}{(\textbf{X}(i)-\bar{X})^3}}
      {\left(\sqrt{\frac{1}{N_p}\sum^{N_p}_{i=1}{(\textbf{X}(i)-\bar{X})^2}}\right)^3}

    Where :math:`\mu_3` is the 3\ :sup:`rd` central moment.

    Skewness measures the asymmetry of the distribution of values about the Mean value. Depending on where the tail is
    elongated and the mass of the distribution is concentrated, this value can be positive or negative.

    Related links:

    https://en.wikipedia.org/wiki/Skewness

    .. note::
      In case of a flat region, the standard deviation and 4\ :sup:`rd` central moment will be both 0. In this case, a
      value of 0 is returned.
    """
    m2 = self._moment(self._target_t, 2)
    m3 = self._moment(self._target_t, 3)

    # Flat Region: prevent division by 0 errors (m3 is 0 there, so the
    # feature becomes 0, exactly like the reference implementation).
    m2 = torch.where(m2 == 0, torch.ones_like(m2), m2)

    return (m3 / m2 ** 1.5).cpu().numpy()

  def getKurtosisFeatureValue(self):
    r"""
    **17. Kurtosis**

    .. math::
      \textit{kurtosis} = \displaystyle\frac{\mu_4}{\sigma^4} =
      \frac{\frac{1}{N_p}\sum^{N_p}_{i=1}{(\textbf{X}(i)-\bar{X})^4}}
      {\left(\frac{1}{N_p}\sum^{N_p}_{i=1}{(\textbf{X}(i)-\bar{X}})^2\right)^2}

    Where :math:`\mu_4` is the 4\ :sup:`th` central moment.

    Kurtosis is a measure of the 'peakedness' of the distribution of values in the image ROI. A higher kurtosis implies
    that the mass of the distribution is concentrated towards the tail(s) rather than towards the mean value. A lower
    kurtosis implies the reverse: that the mass of the distribution is concentrated towards a spike near the Mean value.

    Related links:

    https://en.wikipedia.org/wiki/Kurtosis

    .. note::
      In case of a flat region, the standard deviation and 4\ :sup:`rd` central moment will be both 0. In this case, a
      value of 0 is returned.

    .. note::
      The IBSI feature definition implements excess kurtosis, where kurtosis is corrected by -3, yielding 0 for normal
      distributions. The PyRadiomics kurtosis is not corrected, yielding a value 3 higher than the IBSI kurtosis.
    """
    m2 = self._moment(self._target_t, 2)
    m4 = self._moment(self._target_t, 4)

    # Flat Region: prevent division by 0 errors (m4 is 0 there).
    m2 = torch.where(m2 == 0, torch.ones_like(m2), m2)

    return (m4 / m2 ** 2.0).cpu().numpy()

  def getVarianceFeatureValue(self):
    r"""
    **18. Variance**

    .. math::
      \textit{variance} = \frac{1}{N_p}\displaystyle\sum^{N_p}_{i=1}{(\textbf{X}(i)-\bar{X})^2}

    Variance is the the mean of the squared distances of each intensity value from the Mean value. This is a measure of
    the spread of the distribution about the mean. By definition, :math:`\textit{variance} = \sigma^2`
    """
    u_x = torch.nanmean(self._target_t, dim=1, keepdim=True)
    return torch.nanmean((self._target_t - u_x) ** 2, dim=1).cpu().numpy()

  def getUniformityFeatureValue(self):
    r"""
    **19. Uniformity**

    .. math::
      \textit{uniformity} = \displaystyle\sum^{N_g}_{i=1}{p(i)^2}

    Uniformity is a measure of the sum of the squares of each intensity value. It is a measure of the homogeneity of
    the image array, where a greater uniformity implies a greater homogeneity or a smaller range of discrete intensity
    values.

    .. note::
      Defined by IBSI as Intensity Histogram Uniformity.
    """
    p_i = self.coefficients['p_i']
    return torch.nansum(p_i ** 2, dim=1).cpu().numpy()
