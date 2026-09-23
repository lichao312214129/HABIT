Features from habitat maps
==========================

HABIT-native definitions, formulas, and output CSV columns for features
extracted **on habitat maps** (after habitats exist). Intensity / texture /
shape radiomics follow IBSI definitions as implemented by PyRadiomics
(3-D averaged texture). ROI-level radiomics, voxel-level radiomics, and
3-D shape match ``FeatureExtractor.execute()`` — see **PyRadiomics
alignment** on :doc:`traditional`. Official digital-phantom numbers:
:doc:`traditional`. PyRadiomics catalogue:
`PyRadiomics Feature Reference <https://pyradiomics.readthedocs.io/en/latest/features.html>`_.

Each page: **Output** → **Definition** → **Formula** (if any) → **Output columns** → **Implementation** → **References** (if any).

Light built-in families (``volume``, ``msi``, ``ith_score``, ``graph``) are
peers in ``feature_types``. Topology walk-through:
:doc:`../../examples/graph_features`. CLI / YAML bookmark:
:doc:`../../how_to/graph_features`.

Voxel-level texture used as **habitat inputs** (``local_entropy``,
``voxel_radiomics``) is a different product surface — slice figures live under
:doc:`../../examples/voxel_texture`, not as a ``feature_types`` CSV family.
Registered extractor names (input side and map-side):
:doc:`../../how_to/habitat_components`.

.. list-table:: Feature types
   :header-rows: 1
   :widths: 22 28 50

   * - ``feature_types``
     - Output CSV
     - Page
   * - ``volume``
     - ``volume_features.csv``
     - Light family (voxel counts / volume fractions); see domain ``HabitatVolumeFeatures``
   * - ``msi``
     - ``msi_features.csv``
     - :doc:`msi`
   * - ``ith_score``
     - ``ith_scores.csv``
     - :doc:`ith_score`
   * - ``graph``
     - ``habitat_graph_features.csv``
     - :doc:`graph` (built-in topology family)
   * - ``non_radiomics``
     - ``habitat_basic_features.csv``
     - :doc:`non_radiomics`
   * - ``traditional``
     - ``raw_image_radiomics.csv``
     - :doc:`traditional`
   * - ``whole_habitat`` / ``each_habitat``
     - ``whole_habitat_radiomics.csv`` / ``habitat_{k}_radiomics.csv`` + ``habitat_count.csv``
     - :doc:`whole_each_habitat`

.. toctree::
   :maxdepth: 2
   :caption: Features from habitat maps

   msi
   ith_score
   graph
   non_radiomics
   traditional
   whole_each_habitat
