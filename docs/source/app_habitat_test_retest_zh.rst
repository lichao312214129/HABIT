Test-Retest 分析工具 (habit retest)
===================================

概述
----

Test-Retest 分析用于评估生境映射在不同扫描条件下的可重复性，特别适用于：

- 评估测试-重测扫描中生境标签的映射质量
- 识别生境分割的稳定性
- 为后续分析筛选可靠的生境映射结果

使用方法
--------

Test-Retest 工具使用配置文件：

.. code-block:: bash

   habit retest --config config_test_retest.yaml

参数说明
--------

- **--config, -c**: 配置文件路径

配置方式
--------

配置文件示例：

.. code-block:: yaml

   # config_test_retest.yaml

   out_dir: ./ml_data/test_retest
   test_habitat_table: ./ml_data/habitats_test.csv
   retest_habitat_table: ./ml_data/habitats_retest.csv
   similarity_method: pearson
   input_dir: ./demo_data/habitat_maps/retest
   output_dir: ./ml_data/test_retest/remapped
   processes: 4
   debug: false

然后运行：

.. code-block:: bash

   habit retest --config config_test_retest.yaml

配置文件字段说明：

- **out_dir**: 结果输出根目录
- **test_habitat_table**: 测试扫描生境表路径
- **retest_habitat_table**: 重测扫描生境表路径
- **similarity_method**: 相似性计算方法
- **input_dir**: 重测 NRRD 文件目录
- **output_dir**: 重新映射文件输出目录
- **processes**: 并行进程数
- **debug**: 调试模式

生境表文件格式
--------------

生境表文件应包含以下列：

.. csv-table::
   :header: "subject_id", "habitat_1", "habitat_2", "habitat_3", "..."
   :widths: 15, 15, 15, 15, 15

   "sub-001", 0.12, 0.45, 0.33, ...
   "sub-002", 0.23, 0.56, 0.21, ...
   "sub-003", 0.15, 0.38, 0.47, ...

- **subject_id**: 患者ID
- **habitat_X**: 每个生境的特征值

输出文件说明
------------

运行完成后，在输出目录下生成：

**数据文件：**

- ``remapped_habitats.csv``: 重新映射后的生境表
- ``mapping_quality.csv``: 映射质量评估结果

**映射质量报告示例：**

.. csv-table::
   :header: "subject_id", "mapping_score", "features_used", "status"
   :widths: 20, 15, 20, 15

   "sub-001", 0.95, 45, "success"
   "sub-002", 0.88, 45, "success"
   "sub-003", 0.72, 42, "partial"

相似性方法说明
--------------

**相关系数方法（适用于连续特征）：**

- **pearson**: 皮尔逊相关系数（线性关系）
- **spearman**: 斯皮尔曼相关系数（单调关系）
- **kendall**: 肯德尔相关系数（秩相关）

**距离方法（适用于生境概率图）：**

- **euclidean**: 欧氏距离
- **cosine**: 余弦相似度
- **manhattan**: 曼哈顿距离
- **chebyshev**: 切比雪夫距离

使用建议：

- 连续特征值推荐使用 ``pearson`` 或 ``spearman``
- 生境概率图推荐使用 ``cosine`` 或 ``euclidean``
- 不确定性较高时尝试多种方法比较

与 ICC 分析的结合
-----------------

Test-Retest 映射后，可以使用 ICC 分析评估特征稳定性：

.. code-block:: bash

   # 1. 先进行 Test-Retest 映射
   habit retest --config config_test_retest.yaml

   # 2. 使用 ICC 分析评估特征稳定性
   habit icc --config config_icc.yaml

注意事项
--------

1. **样本配对**: 确保测试和重测扫描来自同一患者
2. **特征一致性**: 两个生境表应具有相同的特征列
3. **文件格式**: 支持 CSV 和 Excel 格式
4. **相似性阈值**: 根据相似性方法设置合适的阈值
5. **并行处理**: 大数据量时增加进程数以加速

常见问题
--------

**Q: 映射失败怎么办？**

A: 可能原因：
- 生境表格式不一致
- 患者ID不匹配
- 相似性阈值设置过高
- 检查日志文件获取详细错误信息

**Q: 选择哪个相似性方法？**

A: 建议：
- 特征值比较：``pearson``
- 生境概率图：``cosine`` 或 ``euclidean``
- 不确定时：尝试多种方法

**Q: 可以只映射部分特征吗？**

A: 可以，在配置文件的 ``features`` 字段中指定特征列表。
