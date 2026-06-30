Habitats and clustering
=======================

**Habitat**: sub-regions within the tumor ROI with similar imaging features; label maps use distinct colors per habitat.

**Typical pipeline**: preprocessing → habitat segmentation → feature extraction → (optional) machine learning.

Clustering strategies
---------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Strategy
     - When to use
   * - Two-Step (demo default)
     - Multi-subject studies; supervoxels per subject, then population habitats
   * - One-Step
     - Single-tumor exploration
   * - Direct Pooling
     - Pool all voxels across subjects (special cases)

How-to: :doc:`../how_to/segment_habitat` . YAML: :doc:`../configuration/habitat` .
