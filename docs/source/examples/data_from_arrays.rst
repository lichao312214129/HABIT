Get data in (directory, SimpleITK, NumPy)
=========================================

**Level:** data in · **Data:** official demo pack · **Extras:** SimpleITK
for the sitk / NumPy wraps · **Time:** first ``fetch_demo()`` downloads
once; later calls reuse the cache

HABIT operators take a :class:`~habit.contracts.Subject` (or a
:class:`~habit.contracts.Cohort` of them). How you build that object is
your choice. Three atomic entries stop at ``Subject`` / ``Cohort``; the
next step is a voxel extractor or a map recipe
(:doc:`habitat_feature_routes`, :doc:`two_step_habitat`).

All three blocks call :func:`~habit.datasets.fetch_demo` (cache or local
``demo_data/preprocessed``, then download). Swap ``DATA`` to your own
tree when you are ready.

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Entry
     - When to use it
     - Figure on this page
   * - **Directory**
     - Files already on disk
     - None — this section only prints the layout
   * - **SimpleITK**
     - You already hold ``sitk.Image`` objects
     - Overlay from the sitk block (demo NRRD)
   * - **NumPy**
     - You already hold ``(z, y, x)`` arrays (nibabel / MONAI too)
     - Overlay from the NumPy block (same NRRD)

Mask arrays must be **integer labels** (``0`` = background). Float masks in
``[0, 1)`` silently become empty ROIs.

Key types: :class:`~habit.contracts.Geometry`,
:class:`~habit.contracts.ArrayImageRef`,
:class:`~habit.contracts.ImageVolume` /
:class:`~habit.contracts.MaskVolume`,
:class:`~habit.contracts.Subject` /
:class:`~habit.contracts.Cohort`.


Directory (``fetch_demo``)
--------------------------

``fetch_demo()`` prints the absolute path and an inventory. That printed
tree is what your own ``DATA`` must match. CLI: ``habit fetch-demo --work-dir .``.

.. literalinclude:: scripts/data_from_arrays_demo.py
   :language: python
   :start-after: # BEGIN disk
   :end-before: # END disk

Layout (no nested ``processed_images``)::

   DATA/
     images/<subject_id>/<modality>/<one image file>
     masks/<subject_id>/<roi>/<one mask file>

Demo subjects are ``subj001`` … ``subj005``. Series keys in the pack:
``pre_contrast``, ``LAP``, ``PVP``, ``delay_3min``. Backup share:
|download_demo_data| (code |demo_data_code|).

CLI / YAML can point ``data_dir`` / ``data.source`` at the same folder,
or at a path-list YAML when files are scattered (below). Never a bare
``.nii.gz``.

This section does **not** embed an MRI figure. After you have the pack,
run a Maps recipe and use that page's ``plot_*`` if you want an overlay.

CLI path-list YAML (scattered files)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Copy ``config/habitat/file_habitat_registered_single_roi.yaml``, edit
subject IDs / modality keys / paths, then point the workflow at **this
file** (not at a single ``.nii.gz``)::

   data_dir: ./file_habitat_registered_single_roi.yaml

   # or, native habitat YAML
   data:
     source: config/habitat/file_habitat_registered_single_roi.yaml

Example (absolute paths; relative paths resolve from **this YAML file's
directory**)::

   auto_select_first_file: false

   images:
     subj001:
       T1: D:/study/registered/subj001/T1.nii.gz
       T2: D:/study/registered/subj001/T2.nii.gz
   masks:
     subj001:
       T1: D:/study/registered/subj001/roi.nii.gz

Direct ``.nii.gz`` paths → keep ``auto_select_first_file: false``.
A folder that contains one NIfTI → ``true``. Check syntax::

   habit check-config --config config/habitat/file_habitat_registered_single_roi.yaml --syntax-only

Modality keys in that YAML must match later ``MODALITIES`` / habitat
config. Templates: :doc:`../configuration/recipe_catalog`.


SimpleITK
---------

Read a demo NRRD with SimpleITK, then wrap it. Geometry (spacing / origin /
direction) is kept. The figure is drawn from **this** block.

.. literalinclude:: scripts/data_from_arrays_demo.py
   :language: python
   :start-after: # BEGIN sitk
   :end-before: # END sitk

Draw the sitk figure (uses ``volume``, ``sitk_cohort``):

.. literalinclude:: scripts/data_from_arrays_demo.py
   :language: python
   :start-after: # BEGIN sitk_figures
   :end-before: # END sitk_figures

.. figure:: ../_static/images/examples/data_from_sitk_overlay.png
   :alt: Habitats from a SimpleITK-backed Subject
   :width: 420

   :func:`~habit.viz.plot_habitat_overlay` after
   :meth:`~habit.api.image.ImageVolume.from_sitk`.


NumPy arrays
------------

Same files, but you already hold arrays (nibabel / MONAI too): put them
into :class:`~habit.contracts.ArrayImageRef`.

.. literalinclude:: scripts/data_from_arrays_demo.py
   :language: python
   :start-after: # BEGIN example
   :end-before: # END example

Draw the NumPy figure (uses ``cohort``, ``t1``):

.. literalinclude:: scripts/data_from_arrays_demo.py
   :language: python
   :start-after: # BEGIN figures
   :end-before: # END figures

Run the whole script from the repository root::

   python docs/source/examples/scripts/data_from_arrays_demo.py

.. figure:: ../_static/images/examples/data_from_arrays_overlay.png
   :alt: Habitats from a NumPy-backed Subject
   :width: 420

   :func:`~habit.viz.plot_habitat_overlay` after wrapping arrays as
   :class:`~habit.contracts.Subject`.


What to read next
-----------------

* :doc:`habitat_feature_routes` — ``raw`` / ``concat`` on these subjects
* :doc:`habitat_atomic_ops` — ``op(subject)``
* :doc:`two_step_habitat` — or pass the ``Cohort`` to ``Study.fit_predict``
