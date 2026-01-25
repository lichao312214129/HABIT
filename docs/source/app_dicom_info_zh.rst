DICOM 信息查看器 (habit dicom-info)
===================================

概述
----

医生在处理影像数据时，经常需要查看 DICOM 文件的元信息，例如：
*   这个序列的层厚是多少？
*   回波时间 (TE) 和重复时间 (TR) 是多少？
*   这是哪个厂商的机器扫描的？

**HABIT DICOM-Info** 是一个轻量级的命令行工具，可以快速扫描文件夹，提取并汇总这些信息到 Excel 表格中，省去了逐个打开查看的麻烦。

使用方法
--------

**基本用法：**

.. code-block:: bash

   # 扫描 "data/dicom" 文件夹，提取信息并保存为 info.csv
   habit dicom-info -i ./data/dicom -o info.csv

**常用参数：**

*   `-i, --input`: 输入文件夹路径（必需）。
*   `-o, --output`: 输出文件路径（支持 .csv 或 .xlsx）。
*   `-t, --tags`: 指定要提取的标签（用逗号分隔）。如果不指定，默认提取常用标签（如 PatientID, StudyDate, Modality 等）。
*   `-r, --recursive`: 是否递归扫描子文件夹（默认开启）。

**高级用法示例：**

如果您只想查看特定的几个标签（例如层厚和厂商），并保存为 Excel：

.. code-block:: bash

   habit dicom-info -i ./data/dicom -o result.xlsx -t "SliceThickness,Manufacturer,ModelName" -f excel

输出结果
--------

生成的表格将包含以下内容：

.. csv-table::
   :header: "FilePath", "PatientID", "Modality", "SliceThickness", "Manufacturer"
   :widths: 30, 15, 10, 15, 20

   "./data/dicom/sub1/1.dcm", "sub-001", "MR", "5.0", "GE MEDICAL SYSTEMS"
   "./data/dicom/sub1/2.dcm", "sub-001", "MR", "5.0", "GE MEDICAL SYSTEMS"
   "...", "...", "...", "...", "..."

这对于整理数据集、筛选符合入组标准的病例非常有用。
