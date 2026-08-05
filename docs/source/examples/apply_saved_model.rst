Apply a saved .habitatmodel to new subjects
===========================================

A fitted :class:`~habit.contracts.HabitatModel` is HABIT's primary
scientific artefact: a self-describing habitat definition that can be
published alongside a paper and applied by other groups to their own
cohorts. This example shows the publish-and-reuse workflow:

1. train a definition on a discovery cohort with
   :func:`~habit.recipes.two_step`,
2. round-trip it through a ``.habitatmodel`` archive
   (:meth:`~habit.contracts.HabitatModel.save` /
   :meth:`~habit.contracts.HabitatModel.load`),
3. project the reloaded definition onto **new, previously unseen subjects**
   with :func:`~habit.recipes.apply_habitat_model`.

No fitting happens after the reload: the model's stored cohort-level
preprocessing state is replayed, so the new supervoxels are scored in the
training feature space — the guarantee that train and predict stay
consistent.

Script
------

.. literalinclude:: scripts/apply_saved_model_demo.py
   :language: python

Output
------

Real output of the script above::

   Trained on 5 subjects: 3 habitats
   Saved habitat_model.habitatmodel (1523 bytes)
   Reloaded model kmeans-f4b52b2c3273c825 (3 habitats, features ['T1', 'T2'])

   New cohort: ['subj001', 'subj002', 'subj003']
     subj001: voxels per habitat {1: 345, 2: 729, 3: 345}
     subj002: voxels per habitat {1: 345, 2: 729, 3: 345}
     subj003: voxels per habitat {1: 345, 2: 729, 3: 345}

   Per-subject habitat features:
   subject  habitat_1_voxel_count  habitat_1_volume_fraction  habitat_2_voxel_count  habitat_2_volume_fraction  habitat_3_voxel_count  habitat_3_volume_fraction
   subj001                  345.0                   0.243129                  729.0                   0.513742                  345.0                   0.243129
   subj002                  345.0                   0.243129                  729.0                   0.513742                  345.0                   0.243129
   subj003                  345.0                   0.243129                  729.0                   0.513742                  345.0                   0.243129

The synthetic generator produces near-identical subjects, hence the repeated
voxel counts; on real data each subject's habitat composition differs.

What to read next
-----------------

* :doc:`two_step_habitat` — the training half of the workflow
* :class:`~habit.contracts.HabitatModel` — the model contract
* :doc:`run_from_yaml` — the same predict path driven by a YAML document
