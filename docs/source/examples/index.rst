Habitat Guide
=============

One scientific task per page. Copy a script, swap ``DATA`` / ``MODALITIES``
/ ``ROI``. Figures on a page come from that page's script.

Imaging pages call :func:`~habit.datasets.fetch_demo` (downloads the official
pack once and prints the folder tree your own data must match). In-memory
pages use synthetic arrays so they run without a download.

Fastest start::

   python docs/source/examples/scripts/two_step_habitat_quickstart.py

Python first map: :doc:`../tutorial/quickstart_python`.
CLI first map: :doc:`../tutorial/quickstart`.
Registered ``Spec`` names (Reference): :doc:`../how_to/habitat_components`.
YAML schemas: :doc:`../configuration/index`.
CLI / YAML bookmarks: :doc:`../how_to/index`.

Scripts live in ``docs/source/examples/scripts/``.
``HABIT_NO_VIEW=1`` skips napari.

.. toctree::
   :maxdepth: 1
   :caption: 1. Data In

   data_from_arrays

.. toctree::
   :maxdepth: 1
   :caption: 2. Voxel Representation

   habitat_feature_routes
   voxel_texture
   custom_voxel_features
   feature_composition
   habitat_preprocessing
   precise_features

.. toctree::
   :maxdepth: 1
   :caption: 3. Habitat Maps

   habitat_analysis_overview
   two_step_habitat
   one_step_habitat
   direct_pooling_habitat
   habitat_atomic_ops
   habitat_fit_modes
   habitat_label_match
   habitat_custom_pipeline

.. toctree::
   :maxdepth: 1
   :caption: 4. Quantify and Paper Outputs

   feature_extraction
   graph_features
   habitat_feature_compare
   features_radiomics_api
   visualization
   provenance_methods

.. toctree::
   :maxdepth: 1
   :caption: 5. Models, Scale and Circulation

   apply_saved_model
   persistence
   parallel_execution
   fault_tolerance
   plugin_entry_points
   run_from_yaml
   cli_yaml_workflows

.. toctree::
   :maxdepth: 1
   :caption: Appendix: Supporting Tools

   image_preprocessing
   cohort_plugins_auxiliary
