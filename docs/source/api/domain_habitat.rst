Habitat domain API
==================

Imaging-side protocols, built-in operators, registries, and
``SubjectPipeline``. Import registries from ``habit.domain`` (they are **not**
top-level ``habit`` exports).

For end-to-end habitat studies without hand-wiring each stage, prefer the L4
recipes (:func:`~habit.recipes.two_step`, :func:`~habit.recipes.direct_pooling`,
:func:`~habit.recipes.one_step`) documented in :doc:`python_api`.

.. code-block:: python

   import habit.domain  # registers all built-ins
   from habit.domain import (
       HabitatAssignerRegistry,
       HabitatFeatureExtractorRegistry,
       HabitatModelFitterRegistry,
       SubjectPipeline,
       SupervoxelFeatureExtractorRegistry,
       SupervoxelizerRegistry,
       VoxelFeatureExtractorRegistry,
   )

Protocols
---------

.. list-table::
   :header-rows: 1
   :widths: 28 42 30

   * - Protocol
     - Call shape
     - Level
   * - ``VoxelFeatureExtractor``
     - ``(Subject) -> VoxelFeatureField``
     - Subject
   * - ``Supervoxelizer``
     - ``(VoxelFeatureField) -> Supervoxelization``
     - Subject
   * - ``SupervoxelFeatureExtractor``
     - enriches supervoxel features
     - Subject
   * - ``HabitatModelFitter``
     - ``fit(units, cohort=...) -> HabitatModel``
     - Cohort
   * - ``HabitatAssigner``
     - ``(Supervoxelization) -> HabitatMap``
     - Subject
   * - ``HabitatFeatureExtractor``
     - ``(Subject, HabitatMap) -> FeatureTable``
     - Subject
   * - ``Seedable``
     - ``set_random_state(seed)``
     - mixin

Registry pattern (all domains)
------------------------------

.. code-block:: python

   names = SupervoxelizerRegistry.available()
   op = SupervoxelizerRegistry.create("slic", n_supervoxels=50)
   schema = SupervoxelizerRegistry.params_model("slic")

Voxel feature extractors
------------------------

Domain: ``voxel_feature_extractor``

.. code-block:: python

   from habit.domain import RawVoxelFeatures, VoxelFeatureExtractorRegistry

   voxel = VoxelFeatureExtractorRegistry.create(
       "raw",
       modalities=["T1", "T2"],
   )
   # Equivalent class form:
   voxel = RawVoxelFeatures(modalities=["T1", "T2"])
   field = voxel(subject)

* ``raw`` → ``RawVoxelFeatures``

Supervoxelizers
---------------

Domain: ``supervoxelizer``

.. code-block:: python

   from habit.domain import SupervoxelizerRegistry

   slic = SupervoxelizerRegistry.create("slic", n_supervoxels=50)
   km = SupervoxelizerRegistry.create("kmeans", n_clusters=40)
   gmm = SupervoxelizerRegistry.create("gmm", n_components=40)
   slic.set_random_state(42)
   unit = slic(field)

* ``slic`` → ``SlicSupervoxelizer``
* ``kmeans`` → ``KMeansSupervoxelizer``
* ``gmm`` → ``GmmSupervoxelizer``

Supervoxel feature extractors
-----------------------------

Domain: ``supervoxel_feature_extractor``

.. code-block:: python

   from habit.domain import SupervoxelFeatureExtractorRegistry

   mean_fx = SupervoxelFeatureExtractorRegistry.create("mean_voxel_features")
   rad_fx = SupervoxelFeatureExtractorRegistry.create("supervoxel_radiomics")

* ``mean_voxel_features`` → ``MeanVoxelFeatures``
* ``supervoxel_radiomics`` → ``SupervoxelRadiomicsFeatures``
* ``mean`` / ``std`` / ``percentile`` → per-supervoxel statistics (compose
  with ``concat``)

Step inspection (optional)
--------------------------

To observe every habitat pipeline boundary in memory, pass
``inspect=StepRecorder(...)`` to a recipe. See
:doc:`../examples/habitat_preprocessing_api` ("Inspect every step").

.. code-block:: python

   from habit import StepRecorder
   import habit.recipes as recipes

   rec = StepRecorder(steps=["supervoxels.described"], max_subjects=2)
   result = recipes.two_step(cohort, spec, inspect=rec)
   result.inspection.summary()

Habitat model fitters
---------------------

Domain: ``habitat_model_fitter`` (cohort-level)

.. code-block:: python

   from habit.domain import HabitatModelFitterRegistry

   fitter = HabitatModelFitterRegistry.create(
       "kmeans",
       n_habitats=4,
       n_init=10,
   )
   fitter.set_random_state(42)
   model = fitter.fit(units, cohort=cohort)

   gmm_fitter = HabitatModelFitterRegistry.create("gmm", n_habitats=4)

* ``kmeans`` → ``KMeansHabitatModelFitter``
* ``gmm`` → ``GmmHabitatModelFitter``

``n_habitats`` may be ``"auto"`` for elbow / model-selection paths supported by
the fitter implementation.

Habitat assigners
-----------------

Domain: ``habitat_assigner``

.. code-block:: python

   from habit.domain import HabitatAssignerRegistry

   assigner = HabitatAssignerRegistry.create("nearest_centroid")
   # Prefer the model factory after fit:
   assigner = model.assigner()
   habitat_map = assigner(unit)

* ``nearest_centroid`` → ``NearestCentroidAssigner``

Habitat feature extractors
--------------------------

Domain: ``habitat_feature_extractor``

.. code-block:: python

   from habit.domain import HabitatFeatureExtractorRegistry

   msi = HabitatFeatureExtractorRegistry.create("msi")
   ith = HabitatFeatureExtractorRegistry.create("ith_score")
   vol = HabitatFeatureExtractorRegistry.create("volume")
   non_rad = HabitatFeatureExtractorRegistry.create("non_radiomics")
   trad = HabitatFeatureExtractorRegistry.create("traditional")
   whole = HabitatFeatureExtractorRegistry.create("whole_habitat")
   each = HabitatFeatureExtractorRegistry.create("each_habitat")

   table = msi(subject, habitat_map)

* ``msi`` → ``MsiHabitatFeatures``
* ``ith_score`` → ``IthHabitatFeatures``
* ``volume`` → ``HabitatVolumeFeatures``
* ``non_radiomics`` → ``NonRadiomicsHabitatFeatures``
* ``traditional`` → ``TraditionalRadiomicsHabitatFeatures``
* ``whole_habitat`` → ``WholeHabitatRadiomicsFeatures``
* ``each_habitat`` → ``EachHabitatRadiomicsFeatures``

``SubjectPipeline``
-------------------

.. code-block:: python

   from habit.domain import SubjectPipeline

   pipe = SubjectPipeline(
       voxel_feature_extractor=voxel,
       supervoxelizer=svx,                 # or None for direct clustering
       habitat_assigner=model.assigner(),
       supervoxel_feature_extractor=None,  # optional
   )
   habitat_map = pipe(subject)
   table = pipe.extract_features(subject, [msi, ith, vol])
   maps = cohort.map(pipe)

Prefer :func:`~habit.domain.assembly.build_habitat_components` when starting
from a :class:`~habit.spec.HabitatSpec`: attribute names on
``HabitatComponents`` match Spec / pipeline fields
(``voxel_feature_extractor``, ``supervoxel_feature_extractor``,
``habitat_model_fitter``, ``habitat_features``, and the singular
``*_feature_preprocessor`` chains). See :doc:`domain`.

Direct (no-supervoxel) designs use ``voxel_units``:

.. code-block:: python

   from habit.domain.pipeline import voxel_units

   units = [voxel_units(voxel(s)) for s in cohort]
   model = fitter.fit(units, cohort=cohort)
   pipe = SubjectPipeline(voxel, None, model.assigner())

Hand-assembled two-step chain
-----------------------------

.. code-block:: python

   fields = [voxel(s) for s in cohort]
   units = [svx(f) for f in fields]
   model = fitter.fit(units, cohort=cohort)
   maps = [model.assigner()(u) for u in units]

Image preprocessing domain
--------------------------

Domain: ``preprocessor`` (image-space steps). Discover schemas the same way:

.. code-block:: python

   from habit import get_param_schema, list_plugins

   for info in list_plugins("preprocessor"):
       print(info.name, get_param_schema(info.name, "preprocessor"))

Registered names: ``resample``, ``reorientation``, ``registration``,
``n4_correction``, ``zscore_normalization``, ``histogram_standardization``,
``adaptive_histogram_equalization``, ``dcm2nii``, ``custom_preprocessor``.
