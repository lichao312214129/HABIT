Habitat domain API
==================

Imaging-side protocols, built-in operators, registries, and
``SubjectPipeline``. This is the **embedding API**: each protocol is
``op(subject)`` or ``op(field)`` and returns a typed contract. Import
registries from the v2 capability packages (they are **not** top-level ``habit``
exports).

Walkthrough (stop after any step, bring your own ``Subject``):
:doc:`../examples/habitat_atomic_ops`. Concept:
:doc:`../tutorial/habitat_analysis`. Arrays in:
:doc:`../examples/data_from_arrays`.

For a whole-cohort study without hand-wiring each operator, use
:meth:`~habit.recipes.Study.fit_predict` with
:attr:`~habit.spec.HabitatSpec.stages` (:doc:`python_api`, beginner
path :doc:`../tutorial/quickstart_python`). The mode-named aliases
(:func:`~habit.recipes.two_step_habitat`,
:func:`~habit.recipes.direct_pooling_habitat`,
:func:`~habit.recipes.one_step_habitat`) remain as thin validators.

A backend is optional. One subject is ``pipe(subject)``. Parallel /
checkpoints: :doc:`../tutorial/execution`.

.. code-block:: python

   import the v2 capability packages  # registers all built-ins
   from habit.habitat_model import HabitatAssignerRegistry, HabitatModelFitterRegistry
   from habit.habitat_features import HabitatFeatureExtractorRegistry
   from habit.pipeline import SubjectPipeline
   from habit.supervoxel import SupervoxelFeatureExtractorRegistry, SupervoxelizerRegistry
   from habit.voxel_features import VoxelFeatureExtractorRegistry

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
   * - ``PoolingMarker``
     - ``()`` marker (built-in ``pool``)
     - Dataflow watershed

Ordered stages and the shared executor
--------------------------------------

A :class:`~habit.spec.Stage` is a named component slot. Stage ``name`` values
are custom labels (not role keywords); scientific roles are inferred from
position + registry domain. Prefer declaring
:attr:`~habit.spec.HabitatSpec.stages` for new code; the classic named
fields (``voxel_feature_extractor``, ``supervoxelizer``, …, ``pooling``)
remain sugar that expands to the same sequence.

.. code-block:: python

   from habit.pipeline.stages import resolve_habitat_stages, run_subject_stage_prefix
   from habit.spec import HabitatSpec, Spec, Stage
   import habit.recipes as recipes

   stages = (
       Stage("extract_voxel_features", Spec("raw", {"modalities": ["T1", "T2"]})),
       Stage("pool", Spec("pool")),
       Stage("fit", Spec("kmeans", {
           "min_habitats": 2, "max_habitats": 10,
           "validation": "elbow", "n_init": 5,
       })),
       Stage("assign", Spec("nearest_centroid")),
       Stage("quantify", Spec("volume")),
   )
   spec = HabitatSpec(name="direct", stages=stages, random_seed=42)
   # Subject-level prefix only (no Cohort required):
   # units = run_subject_stage_prefix(subject, spec)
   result = recipes.Study(spec=spec).fit_predict(cohort)

Domain: ``pooling`` (entry-point group ``habit.pooling``). Built-in marker:
``PoolingRegistry.create("pool")``.

Registry pattern (all domains)
------------------------------

.. code-block:: python

   from habit.supervoxel import SlicSupervoxelizer, SupervoxelizerRegistry

   names = SupervoxelizerRegistry.available()
   print(SupervoxelizerRegistry.constructor_signature("slic"))
   op = SupervoxelizerRegistry.create("slic", n_supervoxels=50)
   op = SlicSupervoxelizer(n_supervoxels=50)

Voxel feature extractors
------------------------

Domain: ``voxel_feature_extractor``

.. code-block:: python

   from habit.voxel_features import (
       RawVoxelFeatures,
       VoxelFeatureExtractorRegistry,
       VoxelRadiomicsFeatures,
       extract_voxel_texture,
   )

   voxel = VoxelFeatureExtractorRegistry.create(
       "raw",
       modalities=["T1", "T2"],
   )
   # Equivalent class form:
   voxel = RawVoxelFeatures(modalities=["T1", "T2"])
   field = voxel(subject)

   texture = VoxelRadiomicsFeatures(modalities=["T2"], kernel_radius=3)
   # Same extractor, one ImageVolume + mask (no Subject):
   field = extract_voxel_texture(image, mask, kernel_radius=3, bin_width=12)

* ``raw`` → ``RawVoxelFeatures``
* ``local_entropy`` → ``LocalEntropyVoxelFeatures``
* ``voxel_radiomics`` → ``VoxelRadiomicsFeatures`` (per-voxel texture)
* ``concat`` / ``expression`` / ``kinetic`` → compose the families above
* :func:`~habit.voxel_features.extract_voxel_texture` — volume-level ``voxel_radiomics``

Full names and knobs: :doc:`../how_to/habitat_components`.

Supervoxelizers
---------------

Domain: ``supervoxelizer``

.. code-block:: python

   from habit.supervoxel import SupervoxelizerRegistry
   # All built-in supervoxelizers use n_supervoxels (not sklearn's
   # n_clusters / n_components). Confirm with the constructor:
   # SupervoxelizerRegistry.constructor_signature("kmeans")
   slic = SupervoxelizerRegistry.create("slic", n_supervoxels=50)
   km = SupervoxelizerRegistry.create("kmeans", n_supervoxels=40)
   gmm = SupervoxelizerRegistry.create("gmm", n_supervoxels=40)
   # All three are Seedable (default seed 0). Current skimage SLIC has no
   # RNG; set_random_state still records the seed for API uniformity.
   slic.set_random_state(42)
   km.set_random_state(42)
   unit = slic(field)

* ``slic`` → ``SlicSupervoxelizer`` (``Seedable``; backend currently deterministic)
* ``kmeans`` → ``KMeansSupervoxelizer`` (``Seedable``)
* ``gmm`` → ``GmmSupervoxelizer`` (``Seedable``)

Supervoxel feature extractors
-----------------------------

Domain: ``supervoxel_feature_extractor``

.. code-block:: python

   from habit.supervoxel import SupervoxelFeatureExtractorRegistry
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
:doc:`../examples/habitat_preprocessing` ("Inspect every step") and
:doc:`../examples/habitat_atomic_ops`.

.. code-block:: python

   from habit.inspection import StepRecorder
   import habit.recipes as recipes

   rec = StepRecorder(steps=["supervoxels.described"], max_subjects=2)
   result = recipes.Study(spec=spec).fit_predict(cohort, inspect=rec)
   result.inspection.summary()

Habitat model fitters
---------------------

Domain: ``habitat_model_fitter`` (cohort-level)

.. code-block:: python

   from habit.habitat_model import HabitatModelFitterRegistry
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

Omit ``n_habitats`` (or pass ``None``) to select the count over
``min_habitats..max_habitats`` via the fitter's ``validation`` score. There is
no string ``"auto"`` value — that was a documentation error.

Habitat assigners
-----------------

Domain: ``habitat_assigner``

.. code-block:: python

   from habit.habitat_model import HabitatAssignerRegistry
   # nearest_centroid requires the fitted HabitatModel (not a bare create()).
   assigner = HabitatAssignerRegistry.create("nearest_centroid", model=model)
   # Preferred after fit — same object, no registry call needed:
   assigner = model.assigner()
   habitat_map = assigner(unit)

* ``nearest_centroid`` → ``NearestCentroidAssigner`` (constructor arg: ``model``)

Habitat feature extractors
--------------------------

Domain: ``habitat_feature_extractor``

.. code-block:: python

   from habit.habitat_features import HabitatFeatureExtractorRegistry
   msi = HabitatFeatureExtractorRegistry.create("msi")
   ith = HabitatFeatureExtractorRegistry.create("ith_score")
   vol = HabitatFeatureExtractorRegistry.create("volume")
   # Built-in graph topology family (not a private plugin).
   graph = HabitatFeatureExtractorRegistry.create(
       "graph",
       edge_method="min_distance",
       node_method="uniform_grid",
       block_size=8,
   )
   non_rad = HabitatFeatureExtractorRegistry.create("non_radiomics")
   trad = HabitatFeatureExtractorRegistry.create("traditional")
   whole = HabitatFeatureExtractorRegistry.create("whole_habitat")
   each = HabitatFeatureExtractorRegistry.create("each_habitat")

   table = msi(subject, habitat_map)
   graph_table = graph(subject, habitat_map)

* ``msi`` → ``MsiHabitatFeatures``
* ``ith_score`` → ``IthHabitatFeatures``
* ``volume`` → ``HabitatVolumeFeatures``
* ``graph`` → ``GraphHabitatFeatures`` (built-in; see
  :doc:`../reference/features/graph`)
* ``non_radiomics`` → ``NonRadiomicsHabitatFeatures``
* ``traditional`` → ``TraditionalRadiomicsHabitatFeatures``
* ``whole_habitat`` → ``WholeHabitatRadiomicsFeatures``
* ``each_habitat`` → ``EachHabitatRadiomicsFeatures``

Compare features between habitats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After ``each_habitat`` (or any wide ``habitat_{id}_{feature}`` table),
:func:`~habit.habitat_features.to_habitat_feature_panel` / :func:`~habit.habitat_features.compare_habitat_features`
contrast habitats on the **cohort** (paired Cliff's delta, BH-FDR) or on
one subject. Figures: :func:`~habit.viz.plot_habitat_feature_heatmap`,
:func:`~habit.viz.plot_habitat_feature_effect`,
:func:`~habit.viz.plot_habitat_feature_components`,
:func:`~habit.viz.plot_habitat_feature_violin`,
:func:`~habit.viz.plot_habitat_feature_bars`. See
:doc:`../reference/features/whole_each_habitat`. Gallery:
:doc:`../examples/habitat_feature_compare`.

For array-only callers, use the public kernel helpers
:func:`~habit.kernels.extract_graph_features` /
:class:`~habit.kernels.HabitatGraphFeatureOptions` (same numeric definitions). Prefer
this domain / kernel path; the ``habit.compat.graph_plugin`` shim was removed
in v2.0.0.

``SubjectPipeline``
-------------------

.. code-block:: python

   from habit.pipeline import SubjectPipeline
   pipe = SubjectPipeline(
       voxel_feature_extractor=voxel,
       supervoxelizer=svx,                 # or None for direct clustering
       habitat_assigner=model.assigner(),
       supervoxel_feature_extractor=None,  # optional
   )
   habitat_map = pipe(subject)
   table = pipe.extract_features(subject, [msi, ith, vol])
   maps = cohort.map(pipe)

Prefer :func:`~habit.pipeline.assembly.build_habitat_components` when starting
from a :class:`~habit.spec.HabitatSpec`: attribute names on
``HabitatComponents`` match Spec / pipeline fields
(``voxel_feature_extractor``, ``supervoxel_feature_extractor``,
``habitat_model_fitter``, ``habitat_features``, and the singular
``*_feature_preprocessor`` chains). See :doc:`domain`.

Direct (no-supervoxel) designs use ``voxel_units``:

.. code-block:: python

   from habit.pipeline import voxel_units

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

Domain: ``preprocessor`` (image-space steps). Discover names the same way;
parameters live on each component constructor.

.. code-block:: python

   from habit.api.plugins import list_plugins
   from habit.image_preprocessing import PreprocessorRegistry

   for info in list_plugins("preprocessor"):
       print(info.name, PreprocessorRegistry.constructor_signature(info.name))

Registered names: ``resample``, ``reorientation``, ``registration``,
``n4_correction``, ``zscore_normalization``, ``histogram_standardization``,
``adaptive_histogram_equalization``, ``dcm2nii``, ``custom_preprocessor``.
