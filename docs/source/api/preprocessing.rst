preprocessing 模块
====================

.. automodule:: habit.core.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

核心处理流程 (Core Pipeline)
-----------------------------

`BatchProcessor` 是预处理模块的核心入口，用于批量执行图像处理任务。

.. automodule:: habit.core.preprocessing.image_processor_pipeline
   :members:
   :undoc-members:
   :show-inheritance:

图像校正 (Correction)
----------------------

.. automodule:: habit.core.preprocessing.n4_correction
   :members:
   :undoc-members:
   :show-inheritance:

标准化与归一化 (Normalization)
-------------------------------

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

空间变换 (Spatial Transform)
-----------------------------

.. automodule:: habit.core.preprocessing.resample
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.preprocessing.registration
   :members:
   :undoc-members:
   :show-inheritance:

格式转换 (Format Conversion)
-----------------------------

.. automodule:: habit.core.preprocessing.dcm2niix_converter
   :members:
   :undoc-members:
   :show-inheritance:

基类与工厂 (Base & Factory)
----------------------------

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
