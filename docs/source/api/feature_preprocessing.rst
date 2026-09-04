:mod:`habit.feature_preprocessing`: scale voxel-feature matrices
================================================================

.. automodule:: habit.feature_preprocessing
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. currentmodule:: habit.feature_preprocessing

**User guide:** Habitat Guide
:doc:`../auto_examples/02_voxel/plot_04_feature_preprocessing` ·
:doc:`domain_habitat`. Component names:
:doc:`../how_to/habitat_components`.

Subject-level and cohort-level preprocessing of voxel / supervoxel
feature matrices, composable into chains. This is **not** image
preprocessing (:doc:`image_preprocessing`) and **not** table-ML
preprocessing (:doc:`table_ml`).

Classes
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   SubjectFeaturePreprocessor
   CohortFeaturePreprocessor
   SubjectPreprocessingChain
   CohortPreprocessingChain
   ZScoreScaling
   MinMaxScaling
   RobustScaling
   Winsorizing
   LogTransform
   Binning
   Impute
   VarianceFilter
   CorrelationFilter
   PreciseCorrelationFilter
   MaxAbsScaling
   QuantileTransform
   L2Normalizer
   FeatureWhitelist
   FeaturePreprocessingMethodRegistry

Functions
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   build_methods
