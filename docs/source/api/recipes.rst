.. _api-recipes:

:mod:`habit.recipes`: named study designs
=========================================

.. automodule:: habit.recipes
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. currentmodule:: habit.recipes

**User guide:** Habitat Guide :doc:`../auto_examples/index` (especially
:doc:`../auto_examples/04_habitat_maps/plot_01_recipes`) ·
:doc:`python_api`. Component names:
:doc:`../how_to/habitat_components`.

Primary entry: :class:`~habit.recipes.Study` (sklearn-style
:meth:`~habit.recipes.Study.fit` / :meth:`~habit.recipes.Study.fit_predict` /
:meth:`~habit.recipes.Study.predict`). Factories
``two_step_habitat`` / ``one_step_habitat`` / ``direct_pooling_habitat``
build a :class:`~habit.recipes.Study` with a declared design.

Tabular ML helpers that also live in this package (``train_model``,
``cross_validate``, …) are on :doc:`table_ml`, not in the tables below.
Image-preprocessing recipes are on :doc:`image_preprocessing`.
:class:`~habit.report.Report` and figure atoms are on :doc:`report`.

Classes
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   Study
   StudyResult

Functions
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   two_step_habitat
   one_step_habitat
   direct_pooling_habitat
   extract_habitat_features
   traditional_radiomics
   identify_precise_voxel_features
   voxel_radiomics_factory
   prior2024_voxel_extract_params
   run_from_yaml

Supporting recipes
------------------

DICOM / table utilities still exported from ``habit.recipes``:

.. autosummary::
   :toctree: generated
   :nosignatures:

   icc_analysis
   sort_dicom
   dice
   dicom_info
   merge_tables
