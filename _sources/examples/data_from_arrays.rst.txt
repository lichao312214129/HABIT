Data from NumPy arrays (Subject / Cohort bridge)
================================================

**Level:** data in · **Data:** in-memory arrays · **Extras:** none · **Time:** <5 s

HABIT does not require its directory layout. If you already hold arrays
(from nibabel, SimpleITK, MONAI, …), wrap them as contracts and call domain
operators or recipes unchanged.

Key types
---------

* :class:`~habit.contracts.Geometry` — ``Geometry.from_array(shape, spacing=...)``
* :class:`~habit.contracts.ArrayImageRef` — lazy ``ImageRef`` over a NumPy array
* :class:`~habit.contracts.ImageVolume` / :class:`~habit.contracts.MaskVolume`
  — eager volumes (``from_geometry``)
* :class:`~habit.contracts.Subject` / :class:`~habit.contracts.Cohort`

Mask arrays must be **integer labels** (``0`` = background). Float masks in
``[0, 1)`` silently become empty ROIs.

Script
------

.. literalinclude:: scripts/data_from_arrays_demo.py
   :language: python

Run::

   python docs/source/examples/scripts/data_from_arrays_demo.py

What to read next
-----------------

* :doc:`habitat_atomic_ops` — run operators on these subjects
* :doc:`two_step_habitat` — or pass the ``Cohort`` to ``Study.fit_predict``
* :doc:`../how_to/prepare_data` — directory layout when you do use files
