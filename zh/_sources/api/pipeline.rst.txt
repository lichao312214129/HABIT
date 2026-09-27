:mod:`habit.pipeline`: compose subject / table pipelines
========================================================

.. automodule:: habit.pipeline
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. currentmodule:: habit.pipeline

**User guide:** :doc:`domain_habitat` · :doc:`python_api`.
Execution backends: :doc:`execution`.

:class:`~habit.pipeline.PoolMarker` is the subject↔cohort watershed
(``pool`` stage). :class:`~habit.pipeline.TablePipeline` is the tabular-ML
composer; see :doc:`table_ml`.

Classes
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   SubjectPipeline
   TablePipeline
   PoolMarker
   PooledUnits
   PoolingRegistry

Functions
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   voxel_units
   fan_in
