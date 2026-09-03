Recipes
=======

Three ``Study`` designs. Same scaffolding: load a cohort, declare
:class:`~habit.spec.HabitatSpec` stages, call
:meth:`~habit.recipes.Study.fit_predict`.

Two-step
--------

``partition`` + ``pool``: supervoxels first, then a shared cohort
definition.

.. literalinclude:: scripts/two_step_habitat_quickstart.py
   :language: python
   :start-after: # BEGIN example
   :end-before: # END example

.. literalinclude:: scripts/two_step_habitat_quickstart.py
   :language: python
   :start-after: # BEGIN figures
   :end-before: # END figures

.. figure:: ../_static/images/examples/two_step_overlay.png
   :alt: Two-step habitat overlay on anatomy
   :width: 720

   Habitat overlay (:func:`~habit.viz.plot_habitat_overlay`).

One-step
--------

Neither ``partition`` nor ``pool``: cluster voxels inside each subject
(no supervoxels). Integer ids are per-subject.

.. literalinclude:: scripts/one_step_habitat_demo.py
   :language: python
   :start-after: # BEGIN example
   :end-before: # END example

Direct-pooling
--------------

``pool`` only: skip the cluster partition and pool existing voxel units
across the cohort.

.. literalinclude:: scripts/direct_pooling_habitat_demo.py
   :language: python
   :start-after: # BEGIN example
   :end-before: # END example

Next: :doc:`habitat_atomic_ops` · :doc:`apply_saved_model`.
