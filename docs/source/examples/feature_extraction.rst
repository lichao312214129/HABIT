Habitat and radiomics feature extraction
========================================

Two extraction paths:

1. **After habitat maps exist** — :func:`~habit.recipes.extract_habitat_features`
   computes light families (``volume``, ``msi``, ``ith_score``, ``non_radiomics``,
   built-in ``graph``) and optional heavy radiomics (``traditional``,
   ``whole_habitat``, ``each_habitat``) from NRRD maps.
2. **Standalone ROI radiomics** — :func:`~habit.recipes.traditional_radiomics`
   extracts PyRadiomics features without habitat segmentation.

For a ``demo_data`` walkthrough of the ``graph`` family alone (subj001
habitats + gallery figures), see :doc:`graph_features`.

Habitat extraction example
----------------------------

The script trains a two-step model, saves maps + ``habitats.parquet``, then
calls the extract recipe. Pass ``n_habitats`` explicitly or rely on auto-detection
from ``habitats.parquet`` / ``habitats.csv``.

.. literalinclude:: scripts/feature_extraction_demo.py
   :language: python

The habitat half of the script ends with a **napari eye-check**.
``HABIT_NO_VIEW=1`` skips it.

Output (abbreviated)::

   Trained: 3 habitats, 3 maps
   Saved habitat maps to .../habitat_maps

   Extracting feature families: ['non_radiomics', 'whole_habitat', ...]
   Output: .../features
     output_dir: .../features
     run_manifest: .../features/habit_run_manifest.json

Traditional radiomics example
-----------------------------

Requires ``demo_data/preprocessed/`` and PyRadiomics.
Use ``--dry-run`` to validate the config dict without running extraction.

.. literalinclude:: scripts/traditional_radiomics_demo.py
   :language: python

What to read next
-----------------

* :doc:`two_step_habitat` — producing the habitat maps first
* :doc:`../reference/features/index` — feature definitions
