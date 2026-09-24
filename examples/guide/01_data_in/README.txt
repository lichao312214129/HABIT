1. Data In
==========

The :doc:`complete analysis </auto_examples/00_full_pipeline/plot_01_full_pipeline>`
starts from a :class:`~habit.contracts.Cohort`. Every route on this
section ends in one. Each page shows
two assemblies: ``Cohort([one_subject])`` and ``Cohort([several_subjects])``.
Pass that cohort to ``fit`` / ``fit_predict``. A Python list is not a cohort.

Pick the page that matches the files you already have.

* **HABIT directory** — ``images/<subject>/<series>/<one file>`` and
  ``masks/<subject>/<roi>/<one file>``. One series or several. The mask
  folder name can differ from the series name.
  :doc:`/auto_examples/01_data_in/plot_01_directory`.
* **Loose NIfTI, NRRD, or MetaImage** — one pair or many. You choose the
  series name and the ROI name. Image and mask grids may differ.
  :doc:`/auto_examples/01_data_in/plot_04_nifti_files`.
* **SimpleITK images already in memory** —
  :doc:`/auto_examples/01_data_in/plot_02_simpleitk`.
* **NumPy arrays or deep-learning tensors** — axis order ``(z, y, x)``,
  integer mask, ``0`` = background.
  :doc:`/auto_examples/01_data_in/plot_03_numpy_arrays`.

DICOM, and series that still need resampling or bias correction, are not a
load route. Preprocess them first (:doc:`/how_to/preprocess`), then use the
directory page. The official demo pack is already preprocessed.

``read_image`` / ``read_mask`` accept what SimpleITK reads
(``.nii``, ``.nii.gz``, ``.nrrd``, ``.mha``, ``.mhd``).
