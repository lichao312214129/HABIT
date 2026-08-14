Whole / Each Habitat Radiomics
==============================

whole_habitat
-------------

Output
~~~~~~

``whole_habitat_radiomics.csv``

Definition
~~~~~~~~~~

PyRadiomics run on the **multi-label habitat map** itself: the habitat label
image is passed as both ``image`` and ``mask`` to PyRadiomics, so voxel
**intensities are habitat label IDs** (not original MR/PET values). Parameters:
``params_file_of_habitat`` (optional; bundled ``habitat`` preset →
``habit/resources/radiomics/parameter_habitat.yaml`` when omitted).

Feature definitions follow `PyRadiomics Feature Reference <https://pyradiomics.readthedocs.io/en/latest/features.html>`_.

Output columns
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Column pattern
     - Description
   * - ``{pyradiomics_feature}``
     - PyRadiomics feature names (no modality suffix)
   * - (excluded)
     - Columns whose names contain ``diagnostic`` are dropped before export

Implementation
~~~~~~~~~~~~~~

``habit/compat/engines/habitat_extraction/habitat_features/builtin_plugins.py``
(``WholeHabitatPlugin``) → ``habitat_radiomics.py``

each_habitat
------------

Output
~~~~~~

- ``habitat_{k}_radiomics.csv`` — one file per habitat index *k* = 1 … *K*
  (*K* = ``n_habitats`` from config or ``habitats.csv``)
- ``habitat_count.csv`` — binary flags ``has_habitat_1`` … ``has_habitat_K``

Definition
~~~~~~~~~~

For each habitat index *k*, PyRadiomics is run on the **original preprocessed
image** with the **multi-label habitat map** as mask and ``label=k`` (voxels
outside label *k* are excluded). Uses ``params_file_of_non_habitat`` (roi
preset), **not** ``params_file_of_habitat``. Files are written for every
*k* in 1 … *K* even when a subject's map lacks that label (empty ROI → NaN /
error handling per subject).

Feature definitions follow `PyRadiomics Feature Reference <https://pyradiomics.readthedocs.io/en/latest/features.html>`_.

Output columns
~~~~~~~~~~~~~~

``habitat_{k}_radiomics.csv``:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Column pattern
     - Description
   * - ``{pyradiomics_feature}_of_{modality}``
     - One column per PyRadiomics feature × image modality
   * - (excluded)
     - Columns whose names contain ``diagnostic`` are dropped before export

``habitat_count.csv``:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Column
     - Description
   * - ``has_habitat_{k}``
     - ``1`` if subject map contains label *k*, else ``0``

Implementation
~~~~~~~~~~~~~~

``habit/compat/engines/habitat_extraction/habitat_features/builtin_plugins.py``
(``EachHabitatPlugin``) → ``habitat_radiomics.py``

Compare habitats (cohort or one subject)
----------------------------------------

The wide ``each_habitat`` table is one row per subject. To argue that
habitats are distinct -- the figure a reviewer expects -- melt it and
contrast habitats **across the cohort** (paired Cliff's delta + BH-FDR).
The same objects also describe one subject (differences without p-values).

Tens-to-hundreds of texture features: draw a habitat x feature heatmap
and a top-k effect-size forest. Violins / bars are for a shortlist, not
the full bank.

::

   from habit import (
       compare_habitat_features,
       plot_habitat_feature_bars,
       plot_habitat_feature_effect,
       plot_habitat_feature_heatmap,
       plot_habitat_feature_violin,
       to_habitat_feature_panel,
   )

   panel = to_habitat_feature_panel(table)          # wide each_habitat FeatureTable
   cmp = compare_habitat_features(panel)            # cohort if table has >= 2 subjects

   fig = plot_habitat_feature_heatmap(cmp)          # overview (z-scored)
   fig = plot_habitat_feature_effect(cmp, top_k=20) # ranked Cliff's delta
   fig = plot_habitat_feature_violin(cmp, max_features=6)
   fig = plot_habitat_feature_heatmap(cmp, subject_id="subj001")  # one subject
   fig = plot_habitat_feature_bars(cmp, subject_id="subj001", max_features=6)

``compare_habitat_features(..., subject_id=...)`` restricts the panel
first. Pairwise ``p_value`` / ``q_value`` are NaN when only one subject
remains.
