Raw and concat voxel features
=============================

Clustering uses the voxel field you define — not a fixed T1 image.
This page is the intensity pair: ``raw`` and ``concat(raw(...), raw(...))``.
Texture: :doc:`voxel_texture`. Formulas / plugins:
:doc:`custom_voxel_features`. All registered names:
:doc:`../how_to/habitat_components`.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Route
     - Role
   * - ``raw``
     - Concatenate modality intensities inside the ROI
   * - ``concat``
     - Join families column-wise, e.g. ``concat(raw("T1"), raw("T2"))``
       (or later ``raw`` + ``voxel_radiomics``; see
       :doc:`../how_to/habitat_components` section 1B)

Every route supports **batch** (``recipes.Study(spec=spec).fit_predict(cohort)``;
two-step sugar or stages) and **atomic** inspection via
:func:`~habit.pipeline.assembly.build_habitat_components` — attribute names
match the Spec (``components.voxel_feature_extractor``,
``components.supervoxel_feature_extractor``, …) and
``components.pipeline(assigner=None).units(subject)``.

Script
------

.. literalinclude:: scripts/habitat_feature_routes_demo.py
   :language: python
   :start-after: # BEGIN example
   :end-before: # END example

Draw the figures
----------------

Paste this after the Script block (it uses ``subject``, ``m0``, and
``raw_result``). Writes ``out/habitat_feature_routes_overlay.png``.

.. literalinclude:: scripts/habitat_feature_routes_demo.py
   :language: python
   :start-after: # BEGIN figures
   :end-before: # END figures

Output (abbreviated)
--------------------

::

   === raw(modalities) ===
     atomic n_features: 2
     batch: 2 maps, ...

   === concat(raw, raw) per modality ===
     atomic n_features: 2
     batch: 2 maps

Run from the repository root::

   python docs/source/examples/scripts/habitat_feature_routes_demo.py

Figures
-------

Each route still ends in habitat maps. Overlay from the ``raw`` route in
this demo:

.. figure:: ../_static/images/examples/habitat_feature_routes_overlay.png
   :alt: Habitat overlay after a feature-route batch fit
   :width: 420

   Habitats after ``Study(...).fit_predict`` with ``raw`` intensities
   (:func:`~habit.viz.plot_habitat_overlay`).

What to read next
-----------------

* :doc:`voxel_texture` — local entropy / voxel radiomics as clustering inputs
* :doc:`custom_voxel_features` — ``expression`` and custom plugins
* :doc:`../how_to/habitat_components` — leaf vs tree; Python / YAML twins
* :doc:`two_step_habitat` — end-to-end two-step workflow
