CSV 合并工具 (habit merge-csv)
==============================

概述
----

`habit merge-csv` 工具用于横向合并多个 CSV 或 Excel 文件，基于公共索引列进行连接。

**主要用途：**

- 合并不同来源的特征数据
- 整合多个时间点的测量结果
- 合并不同模态的数据

与纵向分析工具的区别：

- **merge-csv**: 横向合并（增加列），用于整合不同来源的数据
- **test-retest**: 评估重测一致性，用于质量控制

使用方法
--------

**基本用法**

.. code-block:: bash

   habit merge-csv file1.csv file2.csv file3.csv -o merged.csv

**指定索引列**

.. code-block:: bash

   habit merge-csv file1.csv file2.csv -o merged.csv --index-cols PatientID

**指定不同索引列**

.. code-block:: bash

   habit merge-csv file1.csv file2.csv -o merged.csv --index-cols ID1 ID2

**使用 Excel 文件**

.. code-block:: bash

   habit merge-csv file1.xlsx file2.xlsx -o merged.csv

**指定分隔符**

.. code-block:: bash

   habit merge-csv file1.csv file2.csv -o merged.csv --separator ";"

参数说明
--------

- **input_files**: 要合并的输入文件 (至少2个)
- **-o, --output**: 输出文件路径 (必需)
- **--index-cols**: 索引列名
  - 不指定：使用第一个文件的第一个列
  - 单个列名：所有文件使用相同列名
  - 多个列名：按顺序匹配到每个文件
- **--separator**: CSV 分隔符 (默认逗号)
- **--encoding**: 文件编码 (默认 UTF-8)
- **--join-type**: 连接类型
  - inner: 只保留所有文件中都有索引的行
  - outer: 保留所有索引 (默认)

使用场景示例
------------

**场景1：合并两个扫描的特征**

.. code-block:: bash

   habit merge-csv ./data/scan1_features.csv ./data/scan2_features.csv -o ./data/merged_features.csv --index-cols PatientID

**场景2：合并临床数据和影像特征**

.. code-block:: bash

   habit merge-csv ./data/clinical.csv ./data/radiomics.csv -o ./data/combined.csv --index-cols SubjectID

**场景3：合并多个时间点数据**

.. code-block:: bash

   habit merge-csv baseline.csv month3.csv month6.csv month12.csv -o longitudinal.csv --index-cols PatientID

**场景4：处理不同列名的情况**

.. code-block:: bash

   habit merge-csv file1.csv file2.csv -o merged.csv --index-cols ID ID_number

输出文件说明
------------

合并后的文件包含：

- 索引列 (PatientID 或指定的列名)
- 第一个文件的所有其他列 (加上前缀区分来源)
- 第二个文件的所有其他列 (加上前缀区分来源)
- ...

**输入文件1 (scan1.csv):**

.. csv-table::
   :header: "PatientID", "Feature_A", "Feature_B"
   :widths: 15, 20, 20

   "sub-001", 12.5, 0.45
   "sub-002", 14.2, 0.67

**输入文件2 (scan2.csv):**

.. csv-table::
   :header: "PatientID", "Feature_C", "Feature_D"
   :widths: 15, 20, 20

   "sub-001", 102.3, 0.85
   "sub-002", 98.1, 0.92

**输出文件 (merged.csv):**

.. csv-table::
   :header: "PatientID", "scan1_Feature_A", "scan1_Feature_B", "scan2_Feature_C", "scan2_Feature_D"
   :widths: 15, 20, 20, 20, 20

   "sub-001", 12.5, 0.45, 102.3, 0.85
   "sub-002", 14.2, 0.67, 98.1, 0.92

注意事项
--------

1. **索引列必须唯一**: 每个文件中索引列不能有重复值
2. **索引列数据类型**: 确保所有文件的索引列数据类型一致
3. **内存限制**: 大文件可能需要较长时间合并
4. **列名冲突**: 同名列会自动添加前缀区分
5. **编码问题**: 确保所有文件使用相同编码

常见问题
--------

**Q: 合并后某些行丢失？**

A: 默认使用 inner join，只有所有文件中都存在的索引才会保留。使用 `--join-type outer` 保留所有行。

**Q: 合并后列名重复？**

A: 工具会自动为不同来源的同名列添加前缀区分。

**Q: 可以合并多少个文件？**

A: 没有硬性限制，但建议不超过 20 个文件，以免内存不足。
