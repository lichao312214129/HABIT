Persisting study artefacts
============================

Writing to disk is a **separate, explicit act** in v1.0.
:meth:`~habit.recipes.StudyResult.save` writes the conventional layout:

* ``<subject>_habitats.<ext>`` — habitat label maps (default ``.nrrd``;
  set ``map_format`` to ``nii`` / ``nii.gz`` / ``mha`` / ``mhd``)
* ``<subject>_supervoxel.<ext>`` — supervoxel maps (two-step train; same
  ``map_format``)
* ``habitat_model.habitatmodel`` — self-describing model archive
* ``habitat_features.csv`` / parquet — per-subject habitat features
* ``habitats.parquet`` — clustering-unit table
* ``run_manifest.json`` — provenance + :meth:`~habit.contracts.RunManifest.describe_methods`
* optional ``visualizations/habitat_clustering/`` PCA figures

Script
------

.. literalinclude:: scripts/persistence_demo.py
   :language: python

Output
------

::

   Saved study to .../study_out

   Artefacts (13 files):
     habitat_features.csv
     habitat_model.habitatmodel
     habitats.parquet
     run_manifest.json
     subj001_habitats.nrrd
     subj001_supervoxel.nrrd
     ...
     visualizations/habitat_clustering/habitat_clustering_2D.png

   RunManifest keys: ['finished_at', 'provenance', 'spec_payload', ...]
   Reloaded HabitatModel: kmeans-041200a8a981d09e, 3 habitats
   Apply round-trip maps: 4

What to read next
-----------------

* :doc:`apply_saved_model` — publish and reuse ``.habitatmodel``
* :doc:`run_from_yaml` — CLI-equivalent persistence via ``save=True``
