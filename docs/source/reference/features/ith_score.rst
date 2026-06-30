ITH Score
=========

Output
------

``ith_scores.csv``

Definition
----------

Topological intra-tumor heterogeneity (ITH) score computed on the habitat label map. Each habitat label contributes connected-component statistics; the score increases when labels are fragmented into many small regions.

Formula
-------

.. math::

   \mathrm{ITHscore} = 1 - \frac{1}{S_{\mathrm{total}}} \sum_i \frac{S_{i,\max}}{n_i}

:math:`S_{\mathrm{total}}` = tumor voxel count; :math:`S_{i,\max}` = largest connected component of label *i*; :math:`n_i` = number of connected components of label *i*.

Output columns
--------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Column
     - Description
   * - ``ith_score``
     - ITH score (typically 0–1; higher = more fragmented)
   * - ``habitat_{i}_regions``
     - :math:`n_i` (component count for label *i*)
   * - ``habitat_{i}_largest_area``
     - :math:`S_{i,\max}`
   * - ``habitat_{i}_area_ratio``
     - :math:`S_{i,\max} / n_i`
   * - ``total_area``
     - :math:`S_{\mathrm{total}}`

Notes
-----

HABIT applies the ITHscore topology step to an existing habitat map (``*_habitats.nrrd``). It does **not** repeat the pixel-level radiomics clustering from the original ITHscore pipeline.

Implementation
--------------

``habit/core/habitat_analysis/habitat_features/ith_features.py``

References
----------

Li J et al., *European Radiology* 2023 (`PubMed <https://pubmed.ncbi.nlm.nih.gov/36001124/>`__ · `DOI <https://doi.org/10.1007/s00330-022-09055-0>`__).
