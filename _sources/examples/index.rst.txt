Examples
========

Runnable scripts with captured output. Prefer the How-to chapter for
**YAML + demo_data**; use this gallery for **Python API** snippets.

Fastest start::

   python docs/source/examples/scripts/two_step_habitat_quickstart.py

Habitat examples end with a **napari eye-check** (see the habitats). Close
the window to continue; ``HABIT_NO_VIEW=1`` skips GUI. For 3D, load image +
``*_habitats`` in ITK-SNAP / 3D Slicer / SimpleITK.

YAML / CLI twin: :doc:`cli_yaml_workflows`. Your images: :doc:`../how_to/prepare_data`.

.. toctree::
   :maxdepth: 1

   two_step_habitat
   one_step_habitat
   direct_pooling_habitat
   habitat_preprocessing
   habitat_feature_routes
   feature_composition
   precise_features
   custom_voxel_features
   apply_saved_model
   image_preprocessing
   feature_extraction
   tabular_ml
   ml_advanced
   visualization
   persistence
   parallel_execution
   fault_tolerance
   run_from_yaml
   cli_yaml_workflows
   cohort_plugins_auxiliary
   image_preprocessing_api
   habitat_recipes_api
   habitat_preprocessing_api
   tabular_ml_api
   features_radiomics_api
   viz_parallel_extras_api

Scripts live in ``docs/source/examples/scripts/``.
