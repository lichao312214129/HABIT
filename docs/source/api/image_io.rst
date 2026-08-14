Low-level image I/O helpers
===========================

File I/O helpers (``read_image`` / ``read_mask`` / geometry checks). Prefer
contracts volumes inside pipelines (:doc:`data_model`); use these when you
need SimpleITK-backed read / geometry checks outside a ``Subject``.

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
       policy=GeometryPolicy.RESAMPLE_MASK,
   )
   aligned_image, aligned_mask = pair.image, pair.mask
   print(pair.geometry_report)

GeometryPolicy modes
--------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Policy
     - Behaviour on mismatch
   * - ``STRICT``
     - Raise :class:`~habit.exceptions.GeometryError` (default for this API
       and for ``extract_features`` / ``extract_batch``)
   * - ``WARN``
     - Emit ``RuntimeWarning``; leave arrays/metadata unchanged;
       ``geometry_report.compatible`` is ``False``, ``action="warn"``
   * - ``RESAMPLE_MASK``
     - Resample mask onto the image grid (nearest neighbour); report
       ``compatible=True``, ``action="resample_mask"``
   * - ``RESAMPLE_IMAGE``
     - Resample image onto the mask grid (linear); report
       ``compatible=True``, ``action="resample_image"``

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

   from habit import extract_features, extract_batch, GeometryPolicy

   result = extract_features(image, mask, params="params.yaml")
   batch = extract_batch(
       cases,
       params="params.yaml",
       geometry_policy=GeometryPolicy.STRICT,
       fail_fast=True,   # default: raise on first pair failure
   )

``fail_fast=False`` keeps successful rows and records per-subject errors in
``FeatureTableResult.failures`` (see :doc:`../examples/fault_tolerance`).

Returns ``FeatureResult`` / ``FeatureTableResult``.
