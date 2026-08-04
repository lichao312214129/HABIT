Execution backends
==================

``habit.execution`` runs subject-level operators across a contracts ``Cohort``.

SerialBackend
-------------

.. code-block:: python

   from habit.execution import SerialBackend

   backend = SerialBackend()
   results = list(backend.map(pipeline, cohort))

   # Cohort.map defaults to SerialBackend
   results = cohort.map(pipeline)
   results = cohort.map(pipeline, backend=SerialBackend())

ProcessPoolBackend
------------------

.. code-block:: python

   from habit import RunPolicy
   from habit.execution import ProcessPoolBackend

   backend = ProcessPoolBackend(
       workers=4,
       subject_timeout_sec=900.0,
       on_subject_failure="continue",  # or "fail_fast"
   )
   results = list(backend.map(pipeline, cohort))

   # From RunPolicy
   backend = ProcessPoolBackend.from_policy(
       RunPolicy(workers=4, backend="process")
   )

Only path-backed lazy subjects cross the process boundary. Exceeding
``subject_timeout_sec`` raises ``SubjectTimeoutError``.

CheckpointStore
---------------

.. code-block:: python

   from habit.execution import CheckpointStore, SerialBackend

   store = CheckpointStore("out/run/.habitat_checkpoint")
   results = cohort.map(
       pipeline,
       backend=SerialBackend(),
       checkpoint=store,
   )
   # Re-running skips subjects already recorded as success

Exports: ``SerialBackend``, ``ProcessPoolBackend``, ``CheckpointStore``,
``SubjectTimeoutError``.
