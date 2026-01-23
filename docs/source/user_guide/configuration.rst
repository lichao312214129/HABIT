Configuration Guide
===================

Configuration files in HABIT use YAML format and follow a hierarchical structure.

Configuration Structure
------------------------

.. code-block:: yaml

   # Run mode and pipeline settings
   run_mode: train  # or predict
   pipeline_path:   # Required for predict mode

   # Data paths
   data_dir: ./file_habitat.yaml
   out_dir: ./results/habitat

   # Feature extraction settings
   FeatureConstruction:
     voxel_level:
       method: concat(raw(delay2), raw(delay3), raw(delay5))
       params: {}

   # Habitat segmentation settings
   HabitatsSegmention:
     clustering_mode: two_step
     supervoxel:
       algorithm: kmeans
       n_clusters: 50

   # General settings
   processes: 2
   plot_curves: true
   save_results_csv: true
   random_state: 42

Configuration Options
---------------------

See the configuration templates in ``config_templates/`` for detailed examples
and annotations.
