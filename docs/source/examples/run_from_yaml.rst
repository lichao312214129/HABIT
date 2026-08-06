Run a v1 YAML document with run_from_yaml
=========================================

:func:`~habit.recipes.run_from_yaml` is the programmatic twin of the CLI: it
reads a YAML configuration document, detects its version, and dispatches to
the same recipes the command line uses. **v1** documents
(``version: '1.0'``) are read directly; **v0.1** documents are translated
through :class:`~habit.spec.legacy.LegacyConfigAdapter` first.

To keep this example self-contained, the script first writes a tiny imaging
dataset in HABIT's conventional directory layout and a v1 YAML document,
then executes the document with ``save=True`` — persisting the same
artefacts the CLI would (NRRD habitat maps, the feature table, the
``.habitatmodel`` archive, and the run manifest).

The v1 document's ``spec:`` section mirrors
:class:`~habit.spec.HabitatSpec` field for field; what you write in YAML is
exactly what exists in Python. Scheduling lives under a sibling ``policy:``
block (:class:`~habit.spec.RunPolicy` field names). A complete annotated v1
document ships at ``config/habitat/config_habitat_two_step_v1.yaml``.

**Dual track with v0.1.** A v0.1 habitat YAML keeps parallel / checkpoint
knobs at the **top level** (``processes``, ``individual_subject_*``, …).
``run_from_yaml`` translates them into ``policy`` before selecting
SerialBackend or ProcessPoolBackend. Rename table and defaults:
:doc:`../api/spec`. When timeouts / OOM apply:
:doc:`../api/execution`. Habitat field reference:
:doc:`../configuration/habitat`. Parallel recipe example:
:doc:`parallel_execution`.

Script
------

.. literalinclude:: scripts/run_from_yaml_demo.py
   :language: python

Output
------

Real output of the script above (paths differ per run)::

   Wrote 4 synthetic subjects under /tmp/habit_yaml_demo_.../dataset
   Wrote v1 document /tmp/habit_yaml_demo_.../analysis.yaml

   Result type: StudyResult
   Habitats: 3
   Habitat maps: 4

   Artefacts under /tmp/habit_yaml_demo_.../out:
     P000_habitats.nrrd
     P000_supervoxel.nrrd
     P001_habitats.nrrd
     P001_supervoxel.nrrd
     P002_habitats.nrrd
     P002_supervoxel.nrrd
     P003_habitats.nrrd
     P003_supervoxel.nrrd
     habitat_features.csv
     habitat_model.habitatmodel
     habitats.csv
     run_manifest.json
     visualizations/habitat_clustering/habitat_clustering_2D.png
     visualizations/habitat_clustering/habitat_clustering_3D.png

Running a v0.1 document instead is the same call —
``recipes.run_from_yaml("config/habitat/config_habitat_two_step.yaml")`` —
with translation handled transparently.

What to read next
-----------------

* :doc:`../api/spec` — the spec/YAML relationship, ``RunPolicy`` mapping, v0.1 → v1 migration
* :doc:`../api/execution` — backend selection and knobs
* :doc:`parallel_execution` — policy blocks and ProcessPoolBackend
* :doc:`../configuration/index` — the YAML field reference
* :func:`~habit.recipes.run_from_yaml` — full parameter reference
