Quick Start
===========

This guide will help you run your first habitat analysis.

Basic Workflow
--------------

1. **Prepare your data**: Organize medical images and masks
2. **Create configuration**: Write a YAML configuration file
3. **Run analysis**: Execute the habitat analysis command

Example Configuration
---------------------

Create a file ``config_habitat.yaml``:

.. code-block:: yaml

   # Run mode and pipeline settings
   run_mode: train
   pipeline_path:

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
       random_state: 42

   # General settings
   processes: 2
   plot_curves: true
   save_results_csv: true
   random_state: 42

Running Analysis
----------------

.. code-block:: bash

   habit get-habitat --config config_habitat.yaml

For prediction mode:

.. code-block:: bash

   habit get-habitat --config config_habitat.yaml --mode predict --pipeline path/to/pipeline.pkl

Next Steps
----------

- Read the :doc:`User Guide <../user_guide/index>` for detailed information
- Check out :doc:`Tutorials <../tutorials/index>` for examples
- Explore :doc:`API Reference <../api/index>` for programmatic usage
