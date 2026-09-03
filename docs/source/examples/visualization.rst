Figures and methods
===================

:func:`~habit.viz.plot_habitat_overlay` draws a habitat map on anatomy.
Pass an :class:`~habit.api.image.ImageVolume` and a
:class:`~habit.contracts.HabitatMap` so direction stays attached.
Do not pass ``.data``.

.. code-block:: python

   from pathlib import Path

   from habit.datasets import make_synthetic_cohort
   from habit.spec import HabitatSpec, Spec, Stage
   from habit.viz import plot_habitat_overlay
   import habit.recipes as recipes

   cohort = make_synthetic_cohort(n_subjects=2, shape=(16, 16, 16), rng=3)
   spec = HabitatSpec(
       name="viz_overlay",
       stages=(
           Stage("extract_voxel_features", Spec("raw", {"modalities": ["T1", "T2"]})),
           Stage("partition", Spec("kmeans", {"n_supervoxels": 8, "n_init": 3})),
           Stage("pool", Spec("pool")),
           Stage(
               "fit",
               Spec(
                   "kmeans",
                   {
                       "min_habitats": 2,
                       "max_habitats": 3,
                       "validation": "elbow",
                       "n_init": 3,
                   },
               ),
           ),
           Stage("assign", Spec("nearest_centroid")),
       ),
       random_seed=3,
   )
   result = recipes.Study(spec=spec).fit_predict(cohort)
   subject = cohort[0]
   habitat_map = result.habitat_maps[0]
   Path("out").mkdir(exist_ok=True)
   fig = plot_habitat_overlay(subject.image("T1"), habitat_map, title="habitats")
   fig.savefig("out/habitat_overlay.png", dpi=150, bbox_inches="tight")
   print("Wrote out/habitat_overlay.png")

Optional path check: ``habit view image.nrrd habitats.nrrd``.

.. figure:: ../_static/images/examples/habitat_core_overlay.png
   :alt: Habitat labels overlaid on anatomy
   :width: 420

   Overlay from ``plot_habitat_overlay`` (``ImageVolume`` + ``HabitatMap``).

Methods paragraph
-----------------

After a study, :class:`~habit.contracts.RunManifest` records what actually
ran. :meth:`~habit.contracts.RunManifest.describe_methods` drafts a
manuscript methods paragraph from that record.
:meth:`~habit.spec.HabitatSpec.fingerprint` changes when a stage param
changes, so two specs that look similar are not interchangeable.

The full fingerprint, IBSI, and golden-test story is on :doc:`rigor`.

What to read next
-----------------

* :doc:`feature_extraction` — volume, MSI, ITH, graph
* :doc:`rigor` — IBSI, provenance, numerical gates
