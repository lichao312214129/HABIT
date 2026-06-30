``habit.core.dicom_sort``
==========================

Standalone DICOM sorting (dcm2niix ``-r y``: rename/organize files only), CLI ``habit sort-dicom``, separate from the batch preprocessing pipeline (``habit preprocess`` / ``PreprocessingConfig``).

``DicomSortConfig.from_file`` does not use global ``resolve_config_paths`` value-guessing when loading YAML, so ``-f`` template strings (e.g. ``%n.../xxx.dcm``) are not turned into absolute paths. Only ``data_dir``, ``out_dir``, ``output_dir``, and ``dcm2niix_path`` resolve relative to the config file directory.

API exports
-----------

.. automodule:: habit.core.dicom_sort
   :members:
   :undoc-members:
   :show-inheritance:

Implementation modules
----------------------

.. automodule:: habit.core.dicom_sort.config_schema
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.dicom_sort.run
   :members:
   :undoc-members:
   :show-inheritance:
