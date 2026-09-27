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

Connected components use SimpleITK ``ConnectedComponent`` (default 6-connectivity in 3D, same as ``non_radiomics``).

Output columns
--------------

Default (``Spec("ith_score")`` / ``IthHabitatFeatures()``) writes only
``ith_score``. Pass ``include_auxiliary=True`` to add the summaries below
(CSV export then remaps ``ith_num_habitats`` → ``num_habitats`` and
``ith_total_area`` → ``total_area``).

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Column
     - Description
   * - ``ith_score``
     - ITH score (higher = more fragmented; not clamped to [0, 1])
   * - ``ith_num_habitats``
     - Auxiliary. Number of distinct habitat labels (> 0) in the map
   * - ``habitat_{i}_regions``
     - Auxiliary. :math:`n_i` (component count for label *i*; *i* = actual label value)
   * - ``habitat_{i}_largest_area``
     - Auxiliary. :math:`S_{i,\max}`
   * - ``habitat_{i}_area_ratio``
     - Auxiliary. :math:`S_{i,\max} / n_i`
   * - ``ith_total_area``
     - Auxiliary. :math:`S_{\mathrm{total}}`

Notes
-----

HABIT applies the ITHscore topology step to an existing habitat map (``*_habitats.nrrd``). It does **not** repeat the pixel-level radiomics clustering from the original ITHscore pipeline.

Implementation
--------------

``habit/compat/engines/habitat_extraction/habitat_features/ith_features.py``

References
----------

Li J et al., *European Radiology* 2023 (`PubMed <https://pubmed.ncbi.nlm.nih.gov/36001124/>`__ · `DOI <https://doi.org/10.1007/s00330-022-09055-0>`__).
