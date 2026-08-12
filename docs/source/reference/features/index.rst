Feature reference
=================

HABIT-native feature definitions, formulas, and output CSV columns. PyRadiomics features: `PyRadiomics Feature Reference <https://pyradiomics.readthedocs.io/en/latest/features.html>`_ .

Each page: **Output** → **Definition** → **Formula** (if any) → **Output columns** → **Implementation** → **References** (if any).

Light built-in families (``volume``, ``msi``, ``ith_score``, ``graph``) are
peers in ``feature_types``. Dedicated how-to for topology:
:doc:`../../how_to/graph_features`.

Voxel-level texture used as **habitat inputs** (``local_entropy``,
``voxel_radiomics``) is a different product surface — slice figures live under
:doc:`../../how_to/voxel_texture` / :doc:`../../examples/voxel_texture`, not
as a ``feature_types`` CSV family.

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

   msi
   ith_score
   graph
   non_radiomics
   traditional
   whole_each_habitat
