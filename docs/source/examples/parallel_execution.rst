Parallel execution with RunPolicy
=================================

:class:`~habit.spec.HabitatSpec` declares *what* to compute;
:class:`~habit.spec.RunPolicy` declares *how* to schedule it. Pass a
:class:`~habit.execution.process_pool.ProcessPoolBackend` to any habitat recipe
to process subjects in parallel — with a fixed ``random_seed`` the scientific
result matches serial execution.

YAML equivalent: add a top-level ``policy:`` block (``workers``, ``backend``,
``parallel_mode``, …) — see ``config/habitat/config_habitat_two_step_wsl.yaml``.

Script
------

.. literalinclude:: scripts/parallel_execution_demo.py
   :language: python

Output
------

::

   Serial: 6 maps, 3 habitats
   RunPolicy: workers=2, backend='process', parallel_mode='isolated'
   Parallel: 6 maps, 3 habitats
   Label mismatches serial vs parallel: 0 / 6
   Atomic predict on subj001: 3 labels

What to read next
-----------------

* :doc:`two_step_habitat` — the serial baseline
* :doc:`run_from_yaml` — policy blocks in v1 YAML documents
* :class:`~habit.spec.RunPolicy` — every scheduling field
