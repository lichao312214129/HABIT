MSI Features
============

Output
------

``msi_features.csv``

Definition
----------

Multi-scale co-occurrence of habitat labels between face-adjacent voxels. A co-occurrence matrix :math:`M` is built over the ROI bounding box (padded by one zero-voxel border). First-order entries are raw or normalized counts; second-order entries follow GLCM-style statistics on the normalized matrix :math:`P`.

Formula
-------

For each voxel :math:`\mathbf{x}` and face-neighbor :math:`\mathbf{x}'`:

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

Output columns
--------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Column pattern
     - Description
   * - ``firstorder_{i}_and_{j}`` (i ≤ j)
     - Raw co-occurrence count :math:`M_{ij}`
   * - ``contrast``, ``homogeneity``, ``energy``, ``correlation``
     - Second-order statistics on :math:`P`

Implementation
--------------

``habit/core/habitat_analysis/habitat_features/msi_features.py``

References
----------

Wu J et al., *Radiology* 2018 (`PubMed <https://pubmed.ncbi.nlm.nih.gov/29714680/>`__ · `DOI <https://doi.org/10.1148/radiol.2018172462>`__).
