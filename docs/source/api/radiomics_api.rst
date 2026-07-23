Low-level radiomics API
=======================

Geometry-aware feature extraction for notebooks and third-party pipelines.
This complements the YAML workflow ``run_radiomics``
(:doc:`../how_to/radiomics`).

Import from the top-level package::

   from habit import FeatureResult, FeatureTableResult, extract_features, extract_batch

Example:

.. code-block:: python

   from habit import extract_features, read_image, read_mask

   result = extract_features(
       read_image("data/subj001/T2.nii.gz"),
       read_mask("data/subj001/mask_T2.nii.gz"),
       params=None,  # bundled ROI defaults when omitted
       label=1,
   )
   print(dict(result.values))

Module reference
----------------

.. automodule:: habit.api.radiomics
   :members:
   :undoc-members:
   :show-inheritance:
