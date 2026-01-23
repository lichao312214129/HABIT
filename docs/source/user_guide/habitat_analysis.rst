Habitat Analysis
================

Habitat analysis is the core feature of HABIT, which clusters tumor regions
into distinct habitats based on imaging features.

Overview
--------

The habitat analysis workflow consists of:

1. **Voxel Feature Extraction**: Extract features from each voxel
2. **Individual Clustering**: Cluster voxels into supervoxels (per subject)
3. **Population Clustering**: Cluster supervoxels into habitats (across subjects)

Clustering Strategies
---------------------

HABIT supports three clustering strategies:

Two-Step Strategy (Default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standard approach that first clusters voxels into supervoxels per subject,
then clusters supervoxels into habitats across all subjects.

.. code-block:: yaml

   HabitatsSegmention:
     clustering_mode: two_step

One-Step Strategy
~~~~~~~~~~~~~~~~~

Clusters voxels directly into habitats per subject, without a population-level step.

.. code-block:: yaml

   HabitatsSegmention:
     clustering_mode: one_step

Direct Pooling Strategy
~~~~~~~~~~~~~~~~~~~~~~~

Pools all voxels and clusters them in a single step.

.. code-block:: yaml

   HabitatsSegmention:
     clustering_mode: direct_pooling

Training and Prediction
------------------------

Training Mode
~~~~~~~~~~~~~

Train a new habitat analysis pipeline:

.. code-block:: yaml

   run_mode: train

.. code-block:: bash

   habit get-habitat --config config.yaml

Prediction Mode
~~~~~~~~~~~~~~~

Use a pre-trained pipeline for new data:

.. code-block:: yaml

   run_mode: predict
   pipeline_path: path/to/habitat_pipeline.pkl

.. code-block:: bash

   habit get-habitat --config config.yaml --mode predict

For more details, see :doc:`../architecture/index`.
