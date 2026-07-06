Non-radiomics Morphology
========================

Output
------

``habitat_basic_features.csv``

Definition
----------

Morphological statistics derived from the habitat label map ``L(\mathbf{x})`` (voxel label; 0 = background). No PyRadiomics intensity features are used.

Formula
-------

.. math::

   N_h &= \text{count of distinct labels } k > 0 \quad (\text{num_habitats}) \\
   V_{\mathrm{tumor}} &= \sum_{\mathbf{x}:\, L(\mathbf{x}) \neq 0} 1 \\
   V_k &= \sum_{\mathbf{x}:\, L(\mathbf{x}) = k} 1 \\
   \text{volume_ratio}_k &= V_k / V_{\mathrm{tumor}} \\
   \text{num_regions}_k &= \text{face-connected components with label } k

Connected components use SimpleITK ``ConnectedComponent`` with ``SetFullyConnected(False)`` (6-connectivity in 3D).

The exported CSV uses fixed column indices ``1 … K`` where *K* = ``n_habitats`` from
config (or auto-detected ``habitats.csv``), not necessarily the distinct labels
present in each subject's map. Per-subject ``num_habitats`` counts distinct labels
in that map and may differ from *K*.

Output columns
--------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Column
     - Description
   * - ``num_habitats``
     - Number of distinct habitat labels (> 0) in this subject's map
   * - ``{k}_num_regions`` (k = 1 … K)
     - Face-connected component count for label *k* (0 if absent)
   * - ``{k}_volume_ratio`` (k = 1 … K)
     - Fraction of tumor voxels with label *k* (0 if absent)

Implementation
--------------

``habit/core/habitat_analysis/habitat_features/basic_features.py``
