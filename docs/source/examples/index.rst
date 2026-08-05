Examples
========

Runnable, end-to-end examples of the v1.0 Python API. Every script is
self-contained, deterministic (fixed seeds), and runs without downloaded
data — synthetic cohorts and tables stand in for real data so the examples
work anywhere. The console output shown on each page is the script's real
output, captured from an actual run.

.. note::

   These examples use :func:`~habit.datasets.make_synthetic_cohort` and
   :func:`~habit.datasets.make_synthetic_feature_table` so they run in
   seconds on any machine. To run them on your own images, swap the cohort
   construction for :func:`~habit.contracts.cohort_from_directory` (or a
   ``data:`` section in YAML) — everything downstream is identical.

.. toctree::
   :maxdepth: 1

   two_step_habitat
   apply_saved_model
   run_from_yaml
   tabular_ml

The scripts live in ``docs/source/examples/scripts/`` in the repository and
can be run directly::

   python docs/source/examples/scripts/two_step_habitat_quickstart.py

Why not sphinx-gallery
----------------------

sphinx-gallery is not part of this repository's toolchain; adding it would
execute every example on each documentation build and require a CI imaging
stack. These pages follow the same narrative-plus-runnable-code format with
plain ``literalinclude`` scripts and captured output instead — the scripts
remain directly runnable and testable without a gallery plugin.
