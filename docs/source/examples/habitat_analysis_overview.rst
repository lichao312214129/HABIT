Habitat analysis overview (recipes → atoms → custom)
====================================================

**Level:** conceptual map · **Data:** none · **Extras:** none · **Time:** <1 min read

Habitat analysis is HABIT's core. Examples in this section are layered so you
can enter at the right depth:

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Layer
     - When to use it
     - Start here
   * - **Recipes**
     - Full cohort study, YAML twin, publish maps/features
     - :doc:`two_step_habitat`, :doc:`habitat_fit_modes`
   * - **Atomic operators**
     - Debug one subject; embed in a notebook; no YAML
     - :doc:`habitat_atomic_ops`
   * - **Custom pipeline**
     - Swap extractors / partition / fitter; design your own flow
     - :doc:`habitat_custom_pipeline`

Dataflow (classical two-step)
-----------------------------

::

   Subject
     │  voxel_feature_extractor
     ▼
   VoxelFeatureField
     │  supervoxelizer          ← omit for one_step / direct_pooling
     ▼
   Supervoxelization (units)
     │  pool across subjects    ← cohort watershed (``pool`` stage)
     ▼
   HabitatModel.fit(...)        ← only cohort-level step
     │  assigner bound to model
     ▼
   SubjectPipeline(subject) → HabitatMap
     │  quantify families
     ▼
   FeatureTable (+ RunManifest)

Three recipe shapes (same executor)
-----------------------------------

Strategy is **inferred from stages**, not from the function name:

* **two_step** — ``partition`` + ``pool`` (supervoxels, then cohort habitats)
* **direct_pooling** — ``pool`` without partition (voxels pooled across cohort)
* **one_step** — neither (habitats defined per subject)

Primary API: :func:`~habit.recipes.fit_habitat`. Thin aliases
``two_step`` / ``one_step`` / ``direct_pooling`` only validate the shape.

What must stay paired
---------------------

A published habitat definition is **two objects**:

1. :class:`~habit.contracts.HabitatModel` (centroids + cohort preprocessing state)
2. the :class:`~habit.domain.SubjectPipeline` that produced the fit-time units

Shipping the model without the matching procedure (or changing upstream
stages silently) is how labels look plausible but are wrong. See
:doc:`apply_saved_model` and :doc:`habitat_custom_pipeline`.

Reading order
-------------

1. :doc:`two_step_habitat` — fastest end-to-end recipe
2. :doc:`habitat_atomic_ops` — same science as callables
3. :doc:`habitat_custom_pipeline` — change components safely
4. :doc:`habitat_fit_modes` — all three modes + apply + persistence API
5. Feature design: :doc:`feature_composition`, :doc:`habitat_feature_routes`,
   :doc:`habitat_preprocessing`
