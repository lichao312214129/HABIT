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
       GraphHabitatFeatures,
       HabitatVolumeFeatures,
       IthHabitatFeatures,
       MsiHabitatFeatures,
       NonRadiomicsHabitatFeatures,
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
       [
           HabitatVolumeFeatures(),
           MsiHabitatFeatures(),
           IthHabitatFeatures(),
           NonRadiomicsHabitatFeatures(),
           GraphHabitatFeatures(),
           # Heavy PyRadiomics families (opt-in; require pyradiomics):
           # TraditionalRadiomicsHabitatFeatures(),
           # WholeHabitatRadiomicsFeatures(),
           # EachHabitatRadiomicsFeatures(),
       ],
   )
   maps = cohort.map(pipe)

Set ``supervoxelizer=None`` for direct voxel → habitat designs (no SLIC step).
Train / fit recipes that already hold clustering units should call
``pipe.assign(units)`` or ``pipe.label_and_describe(subject, units, extractors)``
instead of ``pipe(subject)`` / ``extract_features``, so Stage-1 voxel features
are not recomputed.


``HabitatComponents`` (spec → live objects)
-------------------------------------------

:func:`~habit.domain.assembly.build_habitat_components` is the single
construction site that turns a :class:`~habit.spec.HabitatSpec` into live
operators. Attribute names on
:class:`~habit.domain.assembly.HabitatComponents` match the corresponding
``HabitatSpec`` fields and :class:`~habit.domain.pipeline.SubjectPipeline`
parameters (assembled preprocessor chains use the singular form, as on the
pipeline):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ``HabitatComponents`` attribute
     - Spec / pipeline counterpart
   * - ``voxel_feature_extractor``
     - ``HabitatSpec.voxel_feature_extractor``
   * - ``supervoxelizer``
     - ``HabitatSpec.supervoxelizer``
   * - ``supervoxel_feature_extractor``
     - ``HabitatSpec.supervoxel_feature_extractor``
   * - ``voxel_feature_preprocessor``
     - assembled from ``voxel_feature_preprocessors``
   * - ``supervoxel_feature_preprocessor``
     - assembled from ``supervoxel_feature_preprocessors``
   * - ``cohort_feature_preprocessor``
     - assembled from ``cohort_feature_preprocessors``
   * - ``habitat_model_fitter``
     - ``HabitatSpec.habitat_model_fitter``
   * - ``habitat_features``
     - ``HabitatSpec.habitat_features``

.. code-block:: python

   from habit.domain.assembly import build_habitat_components

   components = build_habitat_components(spec)
   # Same name as on the Spec — the live VoxelFeatureExtractor instance:
   field = components.voxel_feature_extractor(subject)
   # Fit-time units (subject-level chains only; no assigner / cohort chain):
   units = components.pipeline(assigner=None).units(subject)

Factory helpers ``build_voxel_extractor`` / ``build_supervoxel_extractor`` /
``build_habitat_extractor`` build a single tree node; they are not
``HabitatComponents`` attributes.


Composing features: trees, combiners, and statistics
----------------------------------------------------

Every extraction stage accepts one **node** abstraction — a recursive
``Spec`` tree built from two shapes:

* **Leaf** — one extraction form over one or more modalities. The
  single-modality form takes ``modality="T1"``; the multi-modality stacking
  form keeps ``modalities=[...]`` (``raw`` only).
* **Combiner node** — a ``Combiner`` implementation with child nodes under
  ``params["children"]``. Combiners merge child blocks **column-wise** and
  know nothing about ``Subject`` or files, which keeps them trivially
  testable and reusable at any level.

.. code-block:: python

   from habit.domain import build_voxel_extractor, build_supervoxel_extractor
   from habit import parse_feature_expression

   voxel_fx = build_voxel_extractor(
       parse_feature_expression(
           'concat(raw("T1"), ratio(raw("T1"), raw("T2"), as_="t1_over_t2"))'
       )
   )
   field = voxel_fx(subject)  # still a one-argument atomic call

``build_voxel_extractor`` / ``build_supervoxel_extractor`` /
``build_habitat_extractor`` route a plain leaf ``Spec`` to the registry and
a tree ``Spec`` to a wrapper that implements the stage's protocol — the
pipeline consumes both transparently. ``SubjectPipeline`` binds the working
and original voxel fields to supervoxel extractors that declare
``bind_fields`` (the statistics extractors ``mean`` / ``std`` /
``percentile`` aggregate one modality's voxel signal per supervoxel;
``source="original"`` selects the pre-preprocessing signal).

Built-in combiners: ``concat``, ``weighted_concat`` / ``average``
(per-child ``weights``), ``ratio`` / ``difference`` (exactly two children),
``kinetic`` (DCE slope pairs), ``expression`` (dataframe arithmetic).
Column naming: single-column nodes keep their source label
(``modality`` > ``as_`` > name); ``as_`` renames only single-output nodes.
See :doc:`../examples/feature_composition` for a runnable tour.

Precision screen: perturbations and precise features
----------------------------------------------------

Two volume-level atoms match the combinatorial design of Prior et al.
(*Radiol Artif Intell* 2024;6(2):e230118). Neither needs a
:class:`~habit.contracts.subject.Cohort` or YAML.

.. code-block:: python

   import numpy as np
   from habit import extract_voxel_texture, perturb_image

   retest_rng = np.random.default_rng(7)
   noisy = perturb_image(image, method="gaussian_noise", rng=retest_rng)
   shifted = perturb_image(noisy, method="translation", shift_fraction=0.5, rng=retest_rng)
   perturbed = perturb_image(shifted, method="rotation", angle_degrees=0.5, rng=retest_rng)
   feat_r1 = extract_voxel_texture(image, mask, kernel_radius=1, bin_width=12)
   feat_r3 = extract_voxel_texture(image, mask, kernel_radius=3, bin_width=12)

``perturb_image`` wraps the registered ``image_perturbation`` methods
that return a perturbed **intensity** volume (``gaussian_noise``,
``translation``, ``rotation``, ``rigid``, ``bspline_deform``).
Mask-only contour methods (``morphological``, ``gradient_weighted``,
``slice_extent``) must be called on a
:class:`~habit.contracts.subject.Subject` via
:class:`~habit.domain.ImagePerturbationRegistry` — see
:doc:`../examples/precise_features`. ``extract_voxel_texture`` is one
voxel-radiomics pass; paper combinations are repeated calls (R1 vs R3,
B12 vs B25, original vs perturbed). Then compose ICC panels:

.. code-block:: python

   from habit import aggregate_panels, identify_precise_features, precision_panel

   panel = precision_panel(
       {"original": field, "perturbed": perturbed_field},
       agreement="absolute",      # ICC(3A,1); "consistency" is ICC(3C,1)
   )
   cohort_panel = aggregate_panels([panel_s1, panel_s2])  # median across subjects
   precise = identify_precise_features(
       {"repeatability": cohort_panel}, lcl_threshold=0.5
   )
   precise.save("precise_features.json")

The ninth protocol, ``ImagePerturbation``, is the Subject-level form of
the same methods (``component(subject, rng=...)``). The paper default
chain is :func:`~habit.domain.precision.prior2024_retest_perturbation`
(Appendix S2 / MIRP 1.2.0). Optional ``bspline_deform`` (MONAI
``Rand3DElastic``; extra ``monai``) is **not** in that chain.

``PreciseFeatureSet.preprocessor()`` returns a ``FeatureWhitelist``.
The one-call recipe is
:func:`habit.recipes.identify_precise_voxel_features`; the runnable tour
is :doc:`../examples/precise_features`.

Randomness (``Seedable``)
-------------------------

Components that participate in study seeding implement ``set_random_state``
(``Seedable``), including supervoxelizers. Default seed is ``0`` unless
``HabitatSpec.random_seed`` / an explicit call overrides it. A backend may
still be numerically deterministic (current skimage SLIC); the method remains
so every stage shares one seeding surface.

.. code-block:: python

   from habit.domain import HabitatModelFitterRegistry, SupervoxelizerRegistry

   slic = SupervoxelizerRegistry.create("slic", n_supervoxels=50)
   km = SupervoxelizerRegistry.create("kmeans", n_supervoxels=40)
   slic.set_random_state(42)
   km.set_random_state(42)
   fitter = HabitatModelFitterRegistry.create("kmeans", n_habitats=4)
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
     - ``mean_voxel_features``, ``supervoxel_radiomics``, ``mean``, ``std``, ``percentile``
   * - ``HabitatModelFitterRegistry``
     - ``habitat_model_fitter``
     - ``kmeans``, ``gmm``
   * - ``HabitatAssignerRegistry``
     - ``habitat_assigner``
     - ``nearest_centroid``
   * - ``CombinerRegistry``
     - ``combiner``
     - ``concat``, ``weighted_concat``, ``ratio``, …
   * - ``ImagePerturbationRegistry``
     - ``image_perturbation``
     - ``gaussian_noise``, ``translation``, ``rotation``, ``rigid``, ``bspline_deform``, ``morphological``, ``gradient_weighted``, ``slice_extent``
   * - ``HabitatFeatureExtractorRegistry``
     - ``habitat_feature_extractor``
     - ``volume``, ``msi``, ``ith_score``, ``graph``, ``non_radiomics`` (light); ``traditional``, ``whole_habitat``, ``each_habitat`` (heavy)
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
