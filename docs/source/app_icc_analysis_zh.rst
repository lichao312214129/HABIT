ICC 分析工具 (habit icc)
========================

概述
----

ICC (Intraclass Correlation Coefficient，组内相关系数) 是评估特征可重复性的重要指标。

**主要用途：**

- 评估特征在不同扫描条件下的一致性
- 筛选稳定的特征用于后续分析
- 评估测量方法的可靠性

ICC 值解释：

- **ICC > 0.9**: 优秀 (Excellent)
- **0.75 < ICC ≤ 0.9**: 良好 (Good)
- **0.5 < ICC ≤ 0.75**: 中等 (Moderate)
- **ICC ≤ 0.5**: 较差 (Poor)

使用方法
--------

**1. 创建配置文件**

.. code-block:: yaml

   # ICC分析配置

   # 输入配置
   input:
     type: files  # 输入类型：files 或 directories
     file_groups:
       # 文件组列表，每组包含要比较的文件路径
       # 同一组内的文件将进行 ICC 分析
       - [./ml_data/dataset1.csv, ./ml_data/dataset2.csv]

   # 评估指标配置
   metrics:
     - icc2    # ICC(2,1): 双向随机效应，绝对一致性
     - icc3    # ICC(3,1): 双向混合效应，一致性
     - cohen    # Cohen's Kappa: 分类一致性
     - fleiss   # Fleiss' Kappa: 多评估者一致性
     - krippendorff  # Krippendorff's Alpha: 多评估者一致性

   # 输出配置
   output:
     path: ./ml_data/icc_results.json  # 输出文件路径

   # 处理配置
   processes: 1  # 并行进程数，null 表示使用所有可用 CPU 核心

   debug: false  # 调试模式

**2. 运行命令**

.. code-block:: bash

   habit icc --config config_icc.yaml

配置说明
--------

**input**: 输入数据配置

- **type**: 输入类型
  - ``files``: 指定文件列表进行配对分析
  - ``directories``: 指定目录列表，自动匹配同名文件

- **file_groups**: 文件组列表（二维数组）
  - 每一行是一个文件组
  - 同一组内的所有文件将进行 ICC 分析
  - 文件数量应一致（相同数量的特征）

**output**: 输出配置

- **path**: 结果输出文件路径 (JSON格式)

**metrics**: ICC 计算配置

- 支持的指标类型：
  - **icc2**: ICC(2,1) - 双向随机效应模型，绝对一致性
  - **icc3**: ICC(3,1) - 双向混合效应模型，一致性
  - **cohen**: Cohen's Kappa - 分类变量一致性
  - **fleiss**: Fleiss' Kappa - 多评估者分类一致性
  - **krippendorff**: Krippendorff's Alpha - 多评估者任意类型一致性

**processes**: 并行处理配置

- 并行进程数量
- ``null`` 或不指定：使用所有可用 CPU 核心
- ``1``: 单进程

**debug**: 调试模式

- ``true``: 启用详细日志
- ``false``: 普通日志级别

输出文件说明
------------

运行完成后，会生成 `icc_results.json` 文件：

.. code-block:: json

   {
     "features": {
       "Feature_1": {
         "icc2": 0.95,
         "icc3": 0.94,
         "ci_icc2": [0.89, 0.98],
         "ci_icc3": [0.88, 0.97],
         "interpretation": "Excellent"
       },
       "Feature_2": {
         "icc2": 0.87,
         "icc3": 0.86,
         "ci_icc2": [0.75, 0.94],
         "ci_icc3": [0.74, 0.93],
         "interpretation": "Good"
       }
     },
     "summary": {
       "excellent_count": 15,
       "good_count": 23,
       "moderate_count": 8,
       "poor_count": 3
     }
   }

**输出字段说明：**

- **icc2**: ICC(2,1) 值
- **icc3**: ICC(3,1) 值
- **ci_icc2**: ICC(2,1) 的 95% 置信区间
- **ci_icc3**: ICC(3,1) 的 95% 置信区间
- **interpretation**: ICC 解释（Excellent/Good/Moderate/Poor）

特征筛选建议
------------

基于 ICC 值进行特征筛选：

1. **选择 ICC > 0.9 的特征**: 用于关键临床决策
2. **选择 ICC > 0.75 的特征**: 用于一般临床研究
3. **排除 ICC < 0.5 的特征**: 不可靠，不建议使用

注意事项
--------

1. **标签列排除**: 确保标签列不参与 ICC 计算
2. **文件配对**: 确保配对文件的样本顺序一致
3. **样本量**: 建议至少 10 对样本进行可靠分析
4. **特征维度**: 确保配对文件具有相同的特征列
5. **数据类型**: ICC 适用于连续变量，Cohen's Kappa 适用于分类变量

常见问题
--------

**Q: ICC 和 Cohen's Kappa 有什么区别？**

A:
- ICC: 适用于连续变量，评估测量值的一致性
- Cohen's Kappa: 适用于分类变量，校正了随机一致性后的 agreement

**Q: 为什么要用 file_groups 而不是直接列出文件？**

A: file_groups 支持多组配对分析：
- 第1组：dataset1 vs dataset2
- 第2组：dataset3 vs dataset4
- 每组独立计算 ICC

**Q: ICC 值很低怎么办？**

A:
- 检查数据配对是否正确
- 确认特征计算方法一致
- 考虑使用更稳定的预处理方法
- 排除异常值后重新计算
