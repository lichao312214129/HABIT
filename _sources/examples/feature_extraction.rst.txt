Habitat features
================

After habitat maps exist, :func:`~habit.recipes.extract_habitat_features`
computes light families — ``volume``, ``msi``, ``ith_score``,
``non_radiomics``, and built-in ``graph`` — from the label maps.
Change ``DATA`` / ``MODALITIES`` / ``ROI`` (and the ``out/`` paths) for
your project.

.. literalinclude:: scripts/feature_extraction_demo.py
   :language: python
   :start-after: # BEGIN example
   :end-before: # END example

.. figure:: ../_static/images/examples/feature_extract_overlay.png
   :alt: Habitat map used for feature extraction
   :width: 420

   Habitat map used before ``extract_habitat_features``
   (:func:`~habit.viz.plot_habitat_overlay`).

Traditional radiomics
---------------------

:func:`~habit.recipes.traditional_radiomics` extracts PyRadiomics features
from a whole-ROI mask without a habitat map. Habitat-wise radiomics
(``traditional``, ``whole_habitat``, ``each_habitat``) are optional heavy
families on the same extract call. Column definitions and the IBSI Phase 1
digital-phantom table are on :doc:`../reference/features/traditional`.
Family index: :doc:`../reference/features/index`.

What to read next
-----------------

* :doc:`visualization` — publication overlay
* :doc:`rigor` — IBSI, provenance, numerical gates
