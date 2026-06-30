Feature reference
=================

HABIT-native feature definitions, formulas, and output CSV columns. PyRadiomics features: `PyRadiomics Feature Reference <https://pyradiomics.readthedocs.io/en/latest/features.html>`_ .

Each page: **Output** → **Definition** → **Formula** → **Columns** → **Implementation** → **References** (if any).

.. list-table:: Feature types
   :header-rows: 1
   :widths: 22 28 50

   * - ``feature_types``
     - Output CSV
     - Page
   * - ``traditional``
     - ``raw_image_radiomics.csv``
     - :doc:`traditional`
   * - ``non_radiomics``
     - ``habitat_basic_features.csv``
     - :doc:`non_radiomics`
   * - ``whole_habitat`` / ``each_habitat``
     - ``whole_habitat_radiomics.csv`` / ``radiomics_of_habitat_*.csv``
     - :doc:`whole_each_habitat`
   * - ``msi``
     - ``msi_features.csv``
     - :doc:`msi`
   * - ``ith_score``
     - ``ith_scores.csv``
     - :doc:`ith_score`
   * - ``graph``
     - ``habitat_graph_features.csv``
     - :doc:`graph/index`

.. toctree::
   :maxdepth: 2
   :caption: Habitat features

   traditional
   non_radiomics
   whole_each_habitat
   msi
   ith_score
   graph/index
