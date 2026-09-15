:orphan:

Prepare data
============

This page is a bookmark. Load images and masks from:

* a directory (``fetch_demo`` / ``cohort_from_directory``),
* one image file + one ROI file (``read_image`` / ``read_mask``, or
  ``FileImageRef`` + ``Subject``),
* SimpleITK, or
* NumPy arrays

on :doc:`../examples/data_from_arrays`. Path-list YAML for CLI (scattered
files) is on that same page. The NIfTI / NRRD file-pair route is
:doc:`/auto_examples/01_data_in/plot_04_nifti_files`.

Then run a map recipe: :doc:`../examples/two_step_habitat`,
:doc:`../examples/one_step_habitat`, or
:doc:`../examples/apply_saved_model`.
