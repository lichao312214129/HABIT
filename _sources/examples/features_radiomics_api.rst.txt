Feature extraction and traditional radiomics API
================================================

* :func:`~habit.recipes.extract_habitat_features` — habitat-wise tables
  (count / basic / MSI / ITH / optional radiomics).
* :func:`~habit.recipes.traditional_radiomics` — whole-ROI PyRadiomics
  without a habitat map.

In-memory habitat studies already return a feature
:class:`~habit.contracts.FeatureTable` from ``two_step`` /
``one_step`` / ``direct_pooling`` when ``habitat_features`` is set on the
spec — no directory layout required. The directory recipes are the CLI twins
for batch extraction over saved NRRD maps.

Script
------

.. literalinclude:: scripts/features_radiomics_api_demo.py
   :language: python

Coverage
--------

``demo_data/results/api/05_extract_features`` and
``06_traditional_radiomics``.

The script ends with a **napari eye-check**. ``HABIT_NO_VIEW=1`` skips it.

What to read next
-----------------

* :doc:`tabular_ml_api` — model the resulting tables
* :doc:`habitat_recipes_api` — produce habitat maps first
