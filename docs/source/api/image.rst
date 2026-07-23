Image and geometry API
======================

Stable helpers for reading volumes and validating image/mask geometry before
radiomics or habitat workflows.

Import from the top-level package::

   from habit import (
       GeometryPolicy,
       ImageVolume,
       MaskVolume,
       read_image,
       read_mask,
       validate_geometry,
       align_image_mask,
   )

Typical notebook pattern:

.. code-block:: python

   from habit import GeometryPolicy, align_image_mask, read_image, read_mask

   image = read_image("data/subj001/T2.nii.gz")
   mask = read_mask("data/subj001/mask_T2.nii.gz")
   pair, report = align_image_mask(
       image,
       mask,
       policy=GeometryPolicy.RESAMPLE_MASK,
   )
   print(report.compatible, report.action)

Module reference
----------------

.. automodule:: habit.api.image
   :members:
   :undoc-members:
   :show-inheritance:
