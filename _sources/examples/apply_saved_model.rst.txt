Apply a saved .habitatmodel to new subjects
===========================================

A fitted :class:`~habit.contracts.HabitatModel` is HABIT's primary
scientific artefact: a self-describing habitat definition that can be
published alongside a paper and applied by other groups to their own
cohorts. This example shows the publish-and-reuse workflow:

1. train a definition on a discovery cohort with
   :func:`~habit.recipes.fit_habitat` (two-step stages),
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

Real output of the script above (abridged; the feature table also includes
``msi``, ``ith_score``, and ``non_radiomics`` columns)::

   Trained on 5 subjects: 3 habitats
   Saved habitat_model.habitatmodel (1531 bytes)
   Reloaded model kmeans-c8e4410e018281c1 (3 habitats, features ['T1', 'T2'])

   New cohort (batch apply): ['subj001', 'subj002', 'subj003']
     subj001: voxels per habitat {1: 513, 2: 394, 3: 563}
     subj002: voxels per habitat {1: 421, 2: 295, 3: 470}
     subj003: voxels per habitat {1: 279, 2: 222, 3: 395}

   Per-subject habitat features (first volume columns shown):
   subject  habitat_1_voxel_count  habitat_1_volume_fraction  ...  ith_score  ...  1_volume_ratio
   subj001                  513.0                   0.348980  ...        0.0  ...       0.348980
   subj002                  421.0                   0.354975  ...        0.0  ...       0.354975
   subj003                  279.0                   0.311384  ...        0.0  ...       0.311384

On real data each subject's habitat composition and feature values differ.

The script ends with a **napari eye-check** on the applied habitats.
``HABIT_NO_VIEW=1`` skips it.

What to read next
-----------------

* :doc:`two_step_habitat` — the training half of the workflow
* :class:`~habit.contracts.HabitatModel` — the model contract
* :doc:`run_from_yaml` — the same predict path driven by a YAML document
