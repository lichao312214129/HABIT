preprocessing module
====================

.. automodule:: habit.core.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

Core pipeline
-------------

``BatchProcessor`` is the main entry point for batch image processing.

.. automodule:: habit.core.preprocessing.configurator
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.preprocessing.image_processor_pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Correction
----------

.. automodule:: habit.core.preprocessing.n4_correction
   :members:
   :undoc-members:
   :show-inheritance:

Normalization
-------------

.. automodule:: habit.core.preprocessing.histogram_standardization
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.preprocessing.zscore_normalization
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.preprocessing.adaptive_histogram_equalization
   :members:
   :undoc-members:
   :show-inheritance:

Spatial transform
-----------------

.. automodule:: habit.core.preprocessing.resample
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.preprocessing.registration
   :members:
   :undoc-members:
   :show-inheritance:

Format conversion
-----------------

DICOM **sort/rename only** (not NIfTI conversion): see :doc:`dicom_sort` and CLI ``habit sort-dicom``.

.. automodule:: habit.core.preprocessing.dcm2niix_converter
   :members:
   :undoc-members:
   :show-inheritance:

Base and factory
----------------

.. automodule:: habit.core.preprocessing.base_preprocessor
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.preprocessing.preprocessor_factory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.preprocessing.custom_preprocessor_template
   :members:
   :undoc-members:
   :show-inheritance:
