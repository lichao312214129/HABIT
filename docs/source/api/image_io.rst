Low-level image I/O helpers
===========================

File I/O helpers from ``habit.api.image``. Prefer contracts volumes inside
pipelines (:doc:`data_model`); use these when you need SimpleITK-backed read /
geometry checks outside a ``Subject``.

.. code-block:: python

   from habit import (
       GeometryPolicy,
       ImageMaskPair,
       align_image_mask,
       read_image,
       read_mask,
       validate_geometry,
   )

   image = read_image("data/subj001/T2.nii.gz", modality="T2")
   mask = read_mask("data/subj001/mask_T2.nii.gz")

   report = validate_geometry(image, mask)
   print(report.compatible, report.mismatches)

   pair = align_image_mask(
       ImageMaskPair(image, mask),
       policy=GeometryPolicy.RESAMPLE_MASK,  # STRICT | WARN | RESAMPLE_MASK
   )
   aligned_image, aligned_mask = pair.image, pair.mask
   print(pair.geometry_report)

Exports: ``GeometryPolicy``, ``GeometryReport``, ``ImageVolume``,
``MaskVolume``, ``ImageMaskPair``, ``read_image``, ``read_mask``,
``validate_geometry``, ``align_image_mask``.

.. warning::

   Top-level ``ImageVolume`` / ``MaskVolume`` here are the **API** types.
   Pipeline code should use ``habit.contracts.ImageVolume`` /
   ``habit.contracts.MaskVolume``.

Low-level radiomics extraction
------------------------------

Component API (not the YAML workflow)::

   from habit import extract_features, extract_batch

   result = extract_features(image, mask, params_file="params.yaml")
   batch = extract_batch(cases, params_file="params.yaml")

Returns ``FeatureResult`` / ``FeatureTableResult``.
