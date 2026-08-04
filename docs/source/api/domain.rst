Domain protocols and registries
================================

``habit.domain`` implements the v1 operator surface: subject-level callables,
one cohort-level habitat fitter, table-ML components, and registries that
construct them by name.

Import style
------------

* Protocols and pipelines: ``from habit import SubjectPipeline, ...`` **or**
  ``from habit.domain import ...``
* **Registries** are **not** top-level exports::

     from habit.domain import SupervoxelizerRegistry, HabitatModelFitterRegistry

Importing ``habit.domain`` registers built-in components.

Single-subject operators
------------------------

Each subject-level operator is a one-argument callable. No cohort, backend, or
YAML is required to process one subject:

.. code-block:: python

   from habit.domain import RawVoxelFeatures, SlicSupervoxelizer

   voxel_fx = RawVoxelFeatures(modalities=["T1", "T2"])
   svx = SlicSupervoxelizer(n_supervoxels=50)

   field = voxel_fx(subject)   # Subject -> VoxelFeatureField
   unit = svx(field)           # VoxelFeatureField -> Supervoxelization

Registry construction (same pattern for every domain):

.. code-block:: python

   from habit.domain import SupervoxelizerRegistry, HabitatModelFitterRegistry

   svx = SupervoxelizerRegistry.create("slic", n_supervoxels=50)
   fitter = HabitatModelFitterRegistry.create("kmeans", n_habitats=4, n_init=10)

Five habitat protocols
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Protocol
     - Call shape
     - Level
   * - ``VoxelFeatureExtractor``
     - ``(Subject) -> VoxelFeatureField``
     - Subject
   * - ``Supervoxelizer``
     - ``(VoxelFeatureField) -> Supervoxelization``
     - Subject
   * - ``HabitatModelFitter``
     - ``fit(Sequence[Supervoxelization]) -> HabitatModel``
     - Cohort
   * - ``HabitatAssigner``
     - ``(Supervoxelization) -> HabitatMap``
     - Subject
   * - ``HabitatFeatureExtractor``
     - ``(Subject, HabitatMap) -> FeatureTable``
     - Subject

Only fitting the habitat model is cohort-level. After that, mapping uses
``model.assigner()`` (or a registry-built assigner).

Hand-assembled two-step chain
-----------------------------

.. code-block:: python

   from habit.domain import (
       HabitatModelFitterRegistry,
       RawVoxelFeatures,
       SlicSupervoxelizer,
   )

   voxel_fx = RawVoxelFeatures(modalities=["T1", "T2"])
   svx = SlicSupervoxelizer(n_supervoxels=50)
   fields = [voxel_fx(s) for s in cohort]
   units = [svx(f) for f in fields]

   fitter = HabitatModelFitterRegistry.create("kmeans", n_habitats=4)
   fitter.set_random_state(42)
   model = fitter.fit(units, cohort=cohort)
   maps = [model.assigner()(u) for u in units]

``SubjectPipeline``
-------------------

Compose the subject-level chain into one callable (HABIT's typed Compose):

.. code-block:: python

   from habit.domain import (
       HabitatVolumeFeatures,
       IthHabitatFeatures,
       MsiHabitatFeatures,
       SubjectPipeline,
   )

   pipe = SubjectPipeline(
       voxel_feature_extractor=voxel_fx,
       supervoxelizer=svx,
       habitat_assigner=model.assigner(),
   )
   # Positional form also works: SubjectPipeline(voxel_fx, svx, model.assigner())

   habitat_map = pipe(subject)
   table = pipe.extract_features(
       subject,
       [MsiHabitatFeatures(), IthHabitatFeatures(), HabitatVolumeFeatures()],
   )
   maps = cohort.map(pipe)

Set ``supervoxelizer=None`` for direct voxel → habitat designs (no SLIC step).

Randomness (``Seedable``)
-------------------------

Components that are stochastic implement ``set_random_state(seed)``. Deterministic
components do not implement the protocol — that fact is itself provenance.

.. code-block:: python

   svx.set_random_state(42)
   fitter.set_random_state(42)

Built-in registry domains
-------------------------

Discover names at runtime with :doc:`plugins` (``list_plugins(domain)``).

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Registry
     - Domain string
     - Examples
   * - ``VoxelFeatureExtractorRegistry``
     - ``voxel_feature_extractor``
     - ``raw``
   * - ``SupervoxelizerRegistry``
     - ``supervoxelizer``
     - ``slic``, ``kmeans``, ``gmm``
   * - ``SupervoxelFeatureExtractorRegistry``
     - ``supervoxel_feature_extractor``
     - ``mean_voxel_features``, ``supervoxel_radiomics``
   * - ``HabitatModelFitterRegistry``
     - ``habitat_model_fitter``
     - ``kmeans``, ``gmm``
   * - ``HabitatAssignerRegistry``
     - ``habitat_assigner``
     - ``nearest_centroid``
   * - ``HabitatFeatureExtractorRegistry``
     - ``habitat_feature_extractor``
     - ``msi``, ``ith_score``, ``volume``, …
   * - ``TablePreprocessorRegistry``
     - ``table_preprocessor``
     - ``zscore``, ``minmax``, ``binning``, …
   * - ``FeatureSelectorRegistry``
     - ``feature_selector``
     - ``anova``, ``lasso``, ``mrmr``, …
   * - ``ClassifierRegistry``
     - ``classifier``
     - ``LogisticRegression``, ``SVM``, …
   * - ``MetricRegistry``
     - ``metric``
     - ``accuracy``, ``auc``, …

``TablePipeline`` composes table preprocessors / selectors / classifiers for
tabular ML outside the imaging path.

See also
--------

* Protocol and registry source: ``habit/domain/``
* Introspection: :doc:`plugins`
* In-memory types: :doc:`data_model`
