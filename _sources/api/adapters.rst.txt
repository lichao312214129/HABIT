Adapters (``habit.adapters``)
=============================

L1 adapters turn external layouts into contracts objects without pulling
YAML or domain logic into the data layer.

DirectoryDataSource
-------------------

For on-disk studies. To explore the API without files, use
:func:`~habit.datasets.make_synthetic_cohort` instead (see :doc:`python_api`).

.. code-block:: python

   from habit.adapters import DirectoryDataSource

   source = DirectoryDataSource(
       "/path/to/processed_images",
       modalities=("T1", "T2"),
       roi="tumor",
       name="training",
   )
   cohort = source.load()  # habit.contracts.Cohort

``cohort_from_directory(...)`` is a thin convenience over this source
(see :doc:`data_model`).

FileImageRef
------------

Lazy on-disk image reference implementing ``ImageRef``:

.. code-block:: python

   from habit.adapters import FileImageRef
   from habit.contracts import Geometry

   # Usually produced by DirectoryDataSource.
   # Constructing manually:
   ref = FileImageRef(
       "data/subj001/T1.nii.gz",
       is_mask=False,
       role_name="T1",
   )
   volume = ref.load()  # ImageVolume with geometry from the file

DirectoryResultWriter
---------------------

The write-side counterpart of ``DirectoryDataSource``, implementing the
``ResultWriter`` protocol with the conventional directory layout:

.. code-block:: python

   from habit.adapters import DirectoryResultWriter

   writer = DirectoryResultWriter("out/study")   # creates nothing yet; maps as .nrrd
   result.write(writer)                          # a StudyResult from habit.recipes

   # Prefer StudyResult.save for the common case (same layout + units table):
   # result.save("out/study", map_format="nii.gz")

   # out/study/<subject>_habitats.nrrd     habitat label maps (geometry preserved)
   # out/study/habitat_model.habitatmodel  the population habitat definition
   # out/study/habitat_features.csv        the cohort feature table
   # out/study/run_manifest.json           provenance and methods text

``map_format`` on :class:`~habit.adapters.DirectoryResultWriter` (and on
:meth:`~habit.recipes.StudyResult.save`) selects the label-map container:
``nrrd`` (default), ``nii``, ``nii.gz``, ``mha``, or ``mhd``. The directory is
created on the first write, so constructing a writer you end up not using
leaves nothing behind. Implement the same four methods elsewhere to send
results to an object store or an in-memory sink instead.
