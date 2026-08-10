Custom habitat pipelines (operators + Spec stages)
==================================================

**Level:** custom / extension · **Data:** synthetic · **Extras:** none · **Time:** ~10–40 s

Use this when the **shape** of a recipe is right (two-step / one-step /
direct pooling) but you need different components: voxel formula, partition
algorithm, fitter, or quantify families.

Three equivalent customisation surfaces
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Surface
     - How you customise
   * - ``Registry.create(name, **params)``
     - Build operators by registered name. Params must match
       ``params_model`` field names (e.g. ``n_supervoxels``, not
       ``n_clusters``).
   * - Hand ``SubjectPipeline``
     - Swap Python callables; fit with a fitter; bind ``model.assigner()``.
   * - ``HabitatSpec.stages``
     - Change ``Stage(..., Spec("name", {...}))`` entries; run
       :func:`~habit.recipes.fit_habitat`. YAML-isomorphic.

Recommended customisation checklist
-----------------------------------

1. Keep the **subject↔cohort watershed** explicit (``pool`` stage or
   hand-pooled units before ``fit``).
2. After changing any upstream extractor / preprocessor, **re-fit** — do not
   reuse an old ``.habitatmodel``.
3. Publish **model + matching procedure** together
   (:meth:`~habit.contracts.HabitatModel.save` and the pipeline / Spec that
   produced fit-time units).
4. For brand-new algorithms, register a plugin
   (:doc:`custom_voxel_features`, :doc:`plugin_entry_points`) then reference
   it by Spec name.

Script
------

.. literalinclude:: scripts/habitat_custom_pipeline_demo.py
   :language: python

Run::

   python docs/source/examples/scripts/habitat_custom_pipeline_demo.py

Common swaps
------------

* Partition: ``Spec("kmeans", {"n_supervoxels": 50})`` ↔ ``Spec("slic", {...})``
* Voxel features: ``raw`` / ``concat(...)`` / expression / custom plugin name
* Insert subject preprocess: ``Stage("preprocess1", Spec("winsorize", {...}))``
  before ``partition``
* Quantify: add ``Stage("quantifyN", Spec("msi"|"ith_score"|...))``

What to read next
-----------------

* :doc:`habitat_atomic_ops` — operator-by-operator walkthrough
* :doc:`feature_composition` — feature trees and combiners
* :doc:`custom_voxel_features` — DIY voxel extractors
* :doc:`../customization/index` — full registry / entry-point guide
