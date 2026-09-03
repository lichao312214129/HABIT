.. _api-contracts:

:mod:`habit.contracts`: in-memory data model
============================================

.. automodule:: habit.contracts
   :no-members:
   :no-inherited-members:
   :no-special-members:

.. currentmodule:: habit.contracts

**User guide:** :doc:`data_model` · Habitat Guide
:doc:`../auto_examples/01_data_in/plot_01_directory`.
Adapters that load these objects from disk: :doc:`adapters`.

Contracts are plain value objects with no IO and no YAML knowledge.

Classes
-------

Imaging
~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   Geometry
   ImageRef
   ArrayImageRef
   ImageVolume
   MaskVolume
   Subject
   Cohort
   CohortFingerprint

Provenance
~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   Provenance
   RunManifest
   StepRecord
   StepObserver

Habitat artefacts
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   VoxelFeatureField
   Supervoxelization
   HabitatMap
   HabitatModel

Tables and outcomes
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   FeatureTable
   Outcome
   BinaryOutcome
   MulticlassOutcome
   ContinuousOutcome
   SurvivalOutcome

Operator protocols
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   SubjectOperator
   CohortOperator
   SubjectResult
   ExecutionBackend
   DataSource
   ResultWriter

Functions
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   cohort_from_directory
   software_fingerprint
   outcome_from_dict
   outcome_to_dict
