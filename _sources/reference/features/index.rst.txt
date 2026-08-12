Feature reference
=================

HABIT-native feature definitions, formulas, and output CSV columns. PyRadiomics features: `PyRadiomics Feature Reference <https://pyradiomics.readthedocs.io/en/latest/features.html>`_ .

Each page: **Output** → **Definition** → **Formula** (if any) → **Output columns** → **Implementation** → **References** (if any).

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
   :caption: Habitat features

   traditional
   non_radiomics
   whole_each_habitat
   msi
   ith_score
   graph
