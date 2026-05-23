``habit.core.dicom_sort``
==========================

独立 DICOM 整理（dcm2niix ``-r y`` ：仅重命名/组织文件），对应 CLI ``habit sort-dicom`` ，
与批量图像预处理流水线（``habit preprocess`` / ``PreprocessingConfig``）分离。

配置类 ``DicomSortConfig.from_file`` 在加载 YAML 时不会启用全局 ``resolve_config_paths``
的「按值猜路径」逻辑，因此不会把 ``-f`` 模板字符串（如 ``%n.../xxx.dcm``）误加成绝对路径；
仅 ``data_dir`` 、``out_dir`` 、``output_dir`` 、``dcm2niix_path`` 相对于配置文件所在目录解析。

API 导出
--------

.. automodule:: habit.core.dicom_sort
   :members:
   :undoc-members:
   :show-inheritance:

实现子模块（按需展开）
----------------------

.. automodule:: habit.core.dicom_sort.config_schema
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: habit.core.dicom_sort.run
   :members:
   :undoc-members:
   :show-inheritance:
