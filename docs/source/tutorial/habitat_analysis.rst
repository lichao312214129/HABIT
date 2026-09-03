Habitat analysis
================

This page is a bookmark. Strategy is inferred from stages, not from the
function name:

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Strategy
     - Stages
     - Use when
   * - **two_step**
     - ``partition`` + ``pool``
     - Shared cohort definition; supervoxels first (typical paper pipeline).
       :doc:`../examples/two_step_habitat`
   * - **direct_pooling**
     - ``pool`` only
     - Shared cohort definition; cluster voxels (no supervoxels).
       :doc:`../examples/direct_pooling_habitat`
   * - **one_step**
     - neither
     - Habitats defined **per subject**. Integer ids are permuted —
       align them before comparing patients
       (:doc:`../examples/apply_saved_model`).
       :doc:`../examples/one_step_habitat`

Atomic embedding (``op(subject)``, no recipe):
:doc:`../examples/habitat_atomic_ops`.

First map: :doc:`quickstart_python` (Python) or :doc:`quickstart` (CLI).
Data in: :doc:`../examples/data_from_arrays`.
