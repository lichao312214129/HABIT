MSI Features
============

Output
------

``msi_features.csv``

Definition
----------

**Multiregional spatial interaction (MSI)** matrix derived from the habitat
label map (intratumor partition map). On the basis of the multiregion map, the
MSI matrix characterizes and quantifies **intratumoral spatial heterogeneity**
(Wu et al., *Radiology* 2018). For each voxel in the ROI bounding box (padded
by one zero-voxel border), face-connected neighbors are probed and the
resulting label pair is added to :math:`M`. Surrounding tissue is included as
one distinct region (row/column 0 = background). Diagonal elements of
:math:`M` represent the connected size of individual subregions, whereas
off-diagonal elements relate to the size of borders where different subregions
meet. Matrix size is :math:`(K{+}1)\times(K{+}1)` where *K* = ``n_habitats``
from config (not per-map auto count). First-order features are raw or
normalized counts from :math:`M`; four second-order statistical features
(``contrast``, ``homogeneity``, ``correlation``, ``energy``) are computed on
the normalized matrix :math:`P`.

Formula
-------

HABIT uses 6-connectivity (face-adjacent neighbors only). The loop covers every
voxel inside the padded bounding box (including background voxels within the
box). For each voxel :math:`\mathbf{x}` and face-neighbor :math:`\mathbf{x}'`:

.. math::

   M_{i,j} \mathrel{+}= 1 \quad \text{when } L(\mathbf{x})=i,\ L(\mathbf{x}')=j

:math:`M` has size :math:`(K{+}1)\times(K{+}1)` (row/column 0 = background). Normalization:

.. math::

   D = \sum_{i \ge 1} \sum_{j \le i} M_{ij}, \quad P = M / D

Second-order features on :math:`P` (indices :math:`i,j = 0,\ldots,K`):

.. math::

   \mathrm{contrast} &= \sum_{i,j} (i-j)^2 P_{ij} \\
   \mathrm{homogeneity} &= \sum_{i,j} \frac{P_{ij}}{1+(i-j)^2} \\
   \mathrm{energy} &= \sum_{i,j} P_{ij}^2 \\
   \mathrm{correlation} &= \frac{\sum_{i,j} i j P_{ij} - \mu_x \mu_y}{\sigma_x \sigma_y}

If :math:`\sigma_x` or :math:`\sigma_y` is zero, ``correlation`` is set to ``1.0``.
If :math:`D=0` (no countable pairs after excluding background row/column from the
denominator), :math:`P` is all zeros and second-order features are zero.

Output columns
--------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Column pattern
     - Description
   * - ``firstorder_{i}_and_{j}`` (i < j, or i = j ≥ 1)
     - Raw MSI matrix entry :math:`M_{ij}` (diagonal i≥1: connected subregion size; off-diagonal: inter-subregion border size). ``firstorder_0_and_0`` is not exported.
   * - ``firstorder_normalized_{i}_and_{j}`` (same index rule)
     - Normalized MSI matrix entry :math:`P_{ij}`. ``firstorder_normalized_0_and_0`` is not exported.
   * - ``contrast``, ``homogeneity``, ``energy``, ``correlation``
     - Four second-order statistical features on :math:`P` (Wu et al., Fig. 2c)

Implementation
--------------

``habit/compat/engines/habitat_extraction/habitat_features/msi_features.py``

References
----------

Wu J et al., *Radiology* 2018 (`PubMed <https://pubmed.ncbi.nlm.nih.gov/29714680/>`__ · `DOI <https://doi.org/10.1148/radiol.2018172462>`__).
