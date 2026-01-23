Configuration
=============

This guide describes the configuration options for HABIT.

Basic Configuration
-------------------

.. code-block:: yaml

    data_dir: /path/to/input/images
    out_dir: /path/to/output

Habitat Analysis Configuration
------------------------------

.. code-block:: yaml

    HabitatAnalysis:
      FeatureConstruction:
        voxel_level:
          method: basic
          params:
            - type: intensity
      HabitatsSegmention:
        supervoxel:
          algorithm: kmeans
          n_clusters: 50
          random_state: 42
        habitat:
          algorithm: kmeans
          min_clusters: 2
          max_clusters: 5
          random_state: 42

Feature Extraction Configuration
--------------------------------

.. code-block:: yaml

    FeatureConstruction:
      voxel_level:
        method: radiomics
        params:
          - type: firstorder
          - type: glcm
          - type: glrlm

Preprocessing Configuration
---------------------------

.. code-block:: yaml

    preprocessing:
      - type: standardization
      - type: normalization
