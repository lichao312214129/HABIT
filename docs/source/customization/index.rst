Extending HABIT
===============

This page is the **v1 habitat plugin** path: implement a protocol, register
it, optionally ship it as an entry point. Third parties should not need to
fork HABIT.

For built-in ``Spec`` names and constructor parameters, see
:doc:`../how_to/habitat_components`. For discovery APIs, see
:doc:`../api/plugins`. For arithmetic formulas without a plugin, use the
built-in ``expression`` voxel extractor
(:doc:`../examples/custom_voxel_features`).

v1 registry and entry points
----------------------------

- Domain string = snake_case singular protocol name
  (e.g. ``voxel_feature_extractor``, ``habitat_model_fitter``)
- Entry-point group = ``habit.<domain>``
- Discover with ``list_plugins("voxel_feature_extractor")`` /
  ``load_plugins()``
- Create with ``Registry.create(name, **params)`` and reference the name
  from :class:`~habit.spec.HabitatSpec`

Habitat domains you typically extend:

- **Voxel feature extractors**: ``VoxelFeatureExtractorRegistry``
- **Supervoxelizers / supervoxel features**
- **Habitat model fitters / assigners / habitat feature extractors**

Three contracts apply to every plugin:

1. **Behavior** — implement the matching protocol (callable with
   ``spec``, usually ``__call__(subject)``).
2. **Registration** — ``@Registry.register("name")`` in-process, or an
   entry point that imports the module which registers.
3. **Parameters** — live on the component constructor (and therefore on
   ``Spec("name", {…})``). Look them up with
   ``Registry.constructor_signature("name")``.

.. important::

   Registration decorators run only when the Python module is imported.
   External plugins must be imported (or loaded via ``load_plugins()``)
   before the workflow starts. For in-tree development, import the module
   from the corresponding subpackage ``__init__.py``.

v1 custom voxel feature extractors
----------------------------------

Use this path for DIY formulas that need neighbourhoods, embeddings, or
logic beyond the built-in ``expression`` DSL. The runnable Custom
features gallery uses three liver DCE maps (arterial relative
enhancement, arterial-to-portal wash-out, arterial-to-delayed wash-out).

**Step 1: Implement the protocol and register**

.. code-block:: python

   import numpy as np
   from habit.contracts import VoxelFeatureField
   from habit.contracts.subject import Subject
   from habit.voxel_features import (
       VoxelFeatureExtractorRegistry,
       aligned_image,
       build_voxel_field,
       roi_voxels,
   )
   from habit.spec import Spec

   PHASES = ("pre_contrast", "LAP", "PVP", "delay_3min")
   NAMES = (
       "relative_enhancement_lap",
       "relative_washout_pvp",
       "relative_washout_delay",
   )

   @VoxelFeatureExtractorRegistry.register("dce_hemodynamics")
   class DCEHemodynamics:
       def __init__(self, phases=PHASES, roi=None, eps=1e-8):
           self.phases = tuple(phases)
           self.roi = roi
           self.eps = float(eps)

       @property
       def spec(self) -> Spec:
           return Spec(
               name="dce_hemodynamics",
               params={"phases": list(self.phases), "roi": self.roi, "eps": self.eps},
           )

       def __call__(self, subject: Subject) -> VoxelFeatureField:
           mask, inside, index = roi_voxels(subject, self.roi)
           pre, lap, pvp, delay = (
               aligned_image(subject, phase, mask, owner="dce_hemodynamics")[inside]
               for phase in self.phases
           )
           values = np.column_stack(
               [
                   (lap - pre) / (pre + self.eps),
                   (lap - pvp) / (lap - pre + self.eps),
                   (lap - delay) / (lap - pre + self.eps),
               ]
           )
           return build_voxel_field(
               subject, mask, index, NAMES, np.asarray(values), self.spec,
           )

**Step 2: Use it from a HabitatSpec**

.. code-block:: python

   from habit.spec import HabitatSpec, Spec, Stage
   import habit.recipes as recipes

   spec = HabitatSpec(
       name="diy",
       stages=(
           Stage(
               "extract_voxel_features",
               Spec(
                   "dce_hemodynamics",
                   {"phases": ["pre_contrast", "LAP", "PVP", "delay_3min"]},
               ),
           ),
           Stage("preprocess", Spec("zscore", {"across_features": False})),
           Stage(
               "fit",
               Spec(
                   "kmeans",
                   {
                       "min_habitats": 2,
                       "max_habitats": 5,
                       "validation": "elbow",
                       "n_init": 3,
                   },
               ),
           ),
           Stage("assign", Spec("nearest_centroid")),
           Stage("quantify", Spec("volume")),
       ),
       random_seed=42,
   )
   result = recipes.Study(spec=spec).fit_predict(cohort)

**Step 3 (optional): ship as a third-party package**

In the plugin's ``pyproject.toml``::

   [project.entry-points."habit.voxel_feature_extractor"]
   dce_hemodynamics = "my_package.features:register"

where ``register()`` performs the ``@VoxelFeatureExtractorRegistry.register``
call (or imports the module that does). Users then::

   from habit.api.plugins import load_plugins
   load_plugins()

See :doc:`../examples/custom_voxel_features` for a runnable demo covering both
``expression`` and a custom plugin, and :doc:`../api/plugins` for discovery.

Other habitat domains follow the same pattern: implement the protocol,
register on the matching registry, reference the name from
:class:`~habit.spec.HabitatSpec`. Catalogs and constructor signatures:
:doc:`../how_to/habitat_components`.

Extension principles
--------------------

**Follow the protocol.** Custom habitat components carry a ``spec`` and
accept the typed contract for that domain (``Subject``, feature field,
habitat map, …). Do not add I/O inside L0–L3 plugins.

**Register a stable name.** Use a descriptive snake_case name
(``dce_hemodynamics``, not ``dcehemo``). Keep it unique in that domain.

**Document parameters.** Constructor arguments are the public contract.
Give defaults and units in the docstring.

**Test.** Unit-test the plugin on a small ``Subject``; then run it through
``recipes.Study(spec=…).fit_predict`` on one demo case.

Image preprocessing and tabular ML
----------------------------------

Image-preprocessing steps and tabular-ML models / selectors are **not**
the public extension story. Those v0.1 factory tours
(``PreprocessorFactory``, ``ModelFactory``, ``SelectorRegistry``, …) are
frozen. Maintainers who still need them should read the archived pages in
the repository:

- ``developer/sphinx_archive/extension_points.rst``
- ``developer/sphinx_archive/customization_index.rst``

FAQ
---

**How do I debug a custom component?** Enable logging, write a unit test
against a synthetic ``Subject``, and call the plugin directly before
wiring it into a ``HabitatSpec``.

**How do I share a custom component?** Ship a small package with a
``habit.<domain>`` entry point, or contribute it upstream with tests.

**How do I handle extra dependencies?** Document them; put non-core stacks
in your plugin's extras, not in HABIT's required dependencies.

Next steps
----------

- :doc:`../development/architecture` — layers and invariants
- :doc:`../development/contributing` — tests and pull requests
- :doc:`../how_to/habitat_components` — built-in component catalog
- :doc:`../api/plugins` — ``list_plugins`` / ``load_plugins``
- :doc:`../reference/upstream_libraries` — third-party library notes
