Configuration reference
=======================

Habitat analysis configs first (``habit get-habitat``, extract).
Preprocessing, DICOM sort, and machine learning are supporting
workflows.

Copy a template from ``config/`` → edit ★ fields (usually data + output paths)
→ run the matching ``habit`` command.

Catalog of templates: :doc:`recipe_catalog`.

Operator how-tos (shorter): :doc:`../how_to/index`.
Concept: :doc:`../tutorial/habitat_analysis`.

.. toctree::
   :maxdepth: 2

   recipe_catalog
   habitat
   feature_extraction
   radiomics
   preprocessing
   dicom_sort
   machine_learning
   model_comparison
   auxiliary
