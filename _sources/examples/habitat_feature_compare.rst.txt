Habitat feature contrast (cohort or one subject)
================================================

**Level:** recipe / atomic · **Data:** synthetic stand-in table (swap
``table``) · **Extras:** ``[tables,viz]`` · **Time:** < 1 min

After an ``each_habitat`` (or any wide ``habitat_{id}_{feature}``) table,
:func:`~habit.compare_habitat_features` melts the wide block so you can
ask: for each feature, do habitats differ, and by how much?

This page builds a **synthetic** stand-in table (two habitats; first-order
Mean / Median / Energy / Kurtosis plus ``volume_fraction``). Swap
``table`` for your ``each_habitat`` /
:func:`~habit.recipes.extract_habitat_features`
:class:`~habit.contracts.FeatureTable`. Then it draws the publication
figures. Bars are **one panel per feature** (independent y-axis) so Energy
and ``volume_fraction`` are not forced onto one linear scale.

``graph`` columns (``single_h*`` / ``pair_h*_h*``) can live on the same
joined :class:`~habit.contracts.FeatureTable`. They are subject-level
topology metrics and do **not** match ``habitat_{id}_{feature}``, so
:func:`~habit.to_habitat_feature_panel` ignores them. Contrast the wide
radiomics / volume block; inspect graph values as a separate table.

This is a software demo, not a clinical claim.

Script
------

Change the ``table`` construction (or load your extract). Figures land
under ``out/``.

.. literalinclude:: scripts/habitat_feature_compare_demo.py
   :language: python
   :start-after: # BEGIN example
   :end-before: # END example

Draw the figures
----------------

Paste this after the Script block (it uses ``cmp``). Writes
``out/habitat_feature_{heatmap_cohort,heatmap_subject,effect,violin,bars}.png``.

.. literalinclude:: scripts/habitat_feature_compare_demo.py
   :language: python
   :start-after: # BEGIN figures
   :end-before: # END figures

Run from the repository root (one line)::

   python docs/source/examples/scripts/habitat_feature_compare_demo.py

Output
------

Illustrative (12 synthetic subjects, two habitats, five features)::

   12 subjects; 5 features
   Wrote habitat-feature contrast figures under out/

Figures
-------

Cohort / single-subject figures from the taught script (not a clinical
claim).

.. figure:: ../_static/images/examples/habitat_feature_heatmap_cohort.png
   :alt: Cohort mean habitat-by-feature heatmap
   :width: 520

   Cohort mean habitat x feature
   (:func:`~habit.viz.plot_habitat_feature_heatmap`, z-scored).

.. figure:: ../_static/images/examples/habitat_feature_heatmap_subject.png
   :alt: One-subject habitat feature heatmap
   :width: 520

   One-subject profile
   (:func:`~habit.viz.plot_habitat_feature_heatmap` with
   ``subject_id``).

.. figure:: ../_static/images/examples/habitat_feature_effect.png
   :alt: Cliff's delta effect-size forest
   :width: 420

   Ranked Cliff's delta
   (:func:`~habit.viz.plot_habitat_feature_effect`).

.. figure:: ../_static/images/examples/habitat_feature_violin.png
   :alt: Per-feature habitat distributions
   :width: 520

   Top-k distributions
   (:func:`~habit.viz.plot_habitat_feature_violin`).

.. figure:: ../_static/images/examples/habitat_feature_bars.png
   :alt: Per-feature habitat means with independent y-axes
   :width: 620

   Faceted bars, one panel per feature
   (:func:`~habit.viz.plot_habitat_feature_bars`).

What to read next
-----------------

* :doc:`../reference/features/whole_each_habitat` — ``each_habitat`` /
  ``whole_habitat`` column layout
* :doc:`graph_features` — graph topology extract + 2D/3D figures
* :doc:`../how_to/extract_features` — YAML / CLI extract (``graph`` is
  now in the default light ``feature_types``)
* :doc:`../api/domain_habitat` — domain extractors and the contrast API
