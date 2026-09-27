:orphan:

Prepare data
============

This page is a bookmark. Load images and masks into a ``Cohort``, then run a map recipe.
Which gallery page to copy depends on the files you already have
(:doc:`/auto_examples/06_manipulating_images/index`):

* HABIT directory ``images/<subject>/<series>/`` and
  ``masks/<subject>/<roi>/`` —
  :doc:`/auto_examples/06_manipulating_images/plot_01_directory`
* Loose NIfTI / NRRD / MetaImage, one subject or many, series name
  different from the ROI name, or grids that differ —
  :doc:`/auto_examples/06_manipulating_images/plot_02_nifti`
* SimpleITK images already in memory —
  :doc:`/auto_examples/06_manipulating_images/plot_03_simpleitk`
* NumPy arrays or tensors —
  :doc:`/auto_examples/06_manipulating_images/plot_04_numpy`

Recipes take that cohort. ``cohort[:1]`` stays a cohort. A Python list
of subjects does not. DICOM still goes through
:doc:`preprocess` before any of the routes above.

Then run a map recipe: :doc:`/auto_examples/01_building_habitat_maps/plot_01_two_step_spec`,
:doc:`/auto_examples/00_introductory_tutorials/plot_02_inside_each_subject`, or
:doc:`/auto_examples/05_validation_and_reuse/plot_01_train_save_predict`.
