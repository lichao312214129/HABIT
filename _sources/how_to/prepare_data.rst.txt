:orphan:

Prepare data
============

This page is a bookmark. Load images and masks into a ``Cohort``, then run a map recipe.
Which gallery page to copy depends on the files you already have
(:doc:`/auto_examples/01_data_in/index`):

* HABIT directory ``images/<subject>/<series>/`` and
  ``masks/<subject>/<roi>/`` —
  :doc:`/auto_examples/01_data_in/plot_01_directory`
* Loose NIfTI / NRRD / MetaImage, one subject or many, series name
  different from the ROI name, or grids that differ —
  :doc:`/auto_examples/01_data_in/plot_04_nifti_files`
* SimpleITK images already in memory —
  :doc:`/auto_examples/01_data_in/plot_02_simpleitk`
* NumPy arrays or tensors —
  :doc:`/auto_examples/01_data_in/plot_03_numpy_arrays`

Recipes take that cohort. ``cohort[:1]`` stays a cohort. A Python list
of subjects does not. DICOM still goes through
:doc:`preprocess` before any of the routes above.

Then run a map recipe: :doc:`../examples/two_step_habitat`,
:doc:`../examples/one_step_habitat`, or
:doc:`../examples/apply_saved_model`.
