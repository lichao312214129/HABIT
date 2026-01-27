生境特征提取
============

本节介绍如何使用 HABIT 从生境图中提取特征。

概述
----

生境特征提取是生境分析的重要步骤，从生成的生境图中提取各种类型的特征，为后续的机器学习建模做准备。

HABIT 支持多种特征类型：

- **传统影像组学特征**: 使用 pyradiomics 提取标准影像组学特征
- **非影像组学特征**: 提取非标准的影像组学特征
- **整体生境特征**: 提取整个生境图谱的特征
- **每个生境特征**: 提取每个生境区域的特征
- **MSI 特征**: 空间交互矩阵（Spatial Interaction Matrix）特征
- **ITH 特征**: 肿瘤内异质性特征

CLI 使用方法
------------

**基本语法：**

.. code-block:: bash

   habit extract --config <config_file>

**参数说明：**

- `--config`, `-c`: 配置文件路径（必需）

**使用示例：**

.. code-block:: bash

   habit extract --config ./demo_data/config_extract_features.yaml

**输出：**

特征文件将保存在配置文件中指定的输出目录。

Python API 使用方法
------------------

**基本用法：**

.. code-block:: python

   from habit.core.habitat_analysis.analyzers.habitat_analyzer import HabitatMapAnalyzer

   # 创建特征提取器
   analyzer = HabitatMapAnalyzer(
       params_file_of_non_habitat='./parameter.yaml',
       params_file_of_habitat='./parameter_habitat.yaml',
       raw_img_folder='./preprocessed/processed_images',
       habitats_map_folder='./results/habitat',
       out_dir='./results/features',
       n_processes=3,
       habitat_pattern='*_habitats.nrrd'
   )

   # 运行特征提取
   analyzer.run(
       feature_types=['traditional', 'non_radiomics', 'whole_habitat', 'each_habitat', 'msi', 'ith_score'],
       n_habitats=3
   )

**详细示例：**

.. code-block:: python

   import logging
   from habit.core.habitat_analysis.analyzers.habitat_analyzer import HabitatMapAnalyzer
   from habit.utils.log_utils import setup_logger
   from pathlib import Path

   # 设置日志
   output_dir = Path('./results/features')
   output_dir.mkdir(parents=True, exist_ok=True)
   logger = setup_logger(
       name='feature_extraction',
       output_dir=output_dir,
       log_filename='feature_extraction.log',
       level=logging.INFO
   )

   # 创建特征提取器
   analyzer = HabitatMapAnalyzer(
       params_file_of_non_habitat='./parameter.yaml',
       params_file_of_habitat='./parameter_habitat.yaml',
       raw_img_folder='./preprocessed/processed_images',
       habitats_map_folder='./results/habitat',
       out_dir='./results/features',
       n_processes=3,
       habitat_pattern='*_habitats.nrrd',
       voxel_cutoff=10
   )

   # 运行特征提取
   logger.info("开始特征提取")
   analyzer.run(
       feature_types=['traditional', 'non_radiomics', 'whole_habitat', 'each_habitat', 'msi', 'ith_score'],
       n_habitats=3
   )
   logger.info("特征提取完成！")

YAML 配置详解
--------------

**配置文件结构：**

.. code-block:: yaml

   # 参数文件
   params_file_of_non_habitat: ./parameter.yaml  # 从原始图像提取特征的参数文件
   params_file_of_habitat: ./parameter_habitat.yaml  # 从生境图提取特征的参数文件

   # 数据目录
   raw_img_folder: ./preprocessed/processed_images  # 原始图像根目录
   habitats_map_folder: ./results/habitat  # 生境图根目录
   out_dir: ./results/features  # 输出目录

   # 特征提取参数
   n_processes: 3  # 并行进程数
   habitat_pattern: '*_habitats.nrrd'  # 生境文件匹配模式

   # 特征类型
   feature_types:
     - traditional      # 传统影像组学特征
     - non_radiomics   # 非影像组学特征
     - whole_habitat   # 整体生境特征
     - each_habitat    # 每个生境特征
    - msi             # MSI（Spatial Interaction Matrix）特征
     - ith_score       # ITH 特征

   n_habitats:   # 生境数量（空表示自动检测）

   # 调试参数
   debug: false  # 启用调试模式

**字段说明：**

**params_file_of_non_habitat**: 从原始图像提取特征的参数文件

- 使用 pyradiomics 提取传统影像组学特征
- 参考 pyradiomics 文档了解参数说明

**params_file_of_habitat**: 从生境图提取特征的参数文件

- 使用 pyradiomics 从生境图中提取特征
- 参考 pyradiomics 文档了解参数说明

**raw_img_folder**: 原始图像根目录

- 包含预处理后的图像
- 用于提取传统影像组学特征

**habitats_map_folder**: 生境图根目录

- 包含生成的生境图
- 用于提取生境相关特征

**out_dir**: 输出目录

- 特征文件将保存在此目录

**n_processes**: 并行进程数

- 用于并行处理多个受试者
- 根据您的硬件配置调整

**habitat_pattern**: 生境文件匹配模式

- 用于匹配生境图文件
- 支持通配符（`*`）

**feature_types**: 特征类型列表

- `traditional`: 传统影像组学特征
- `non_radiomics`: 非影像组学特征
- `whole_habitat`: 整体生境特征
- `each_habitat`: 每个生境特征
- `msi`: MSI（Spatial Interaction Matrix）特征
- `ith_score`: ITH 特征

**n_habitats**: 生境数量

- 空表示自动检测
- 可以手动指定生境数量

**debug**: 调试模式

- 启用详细日志
- 便于调试和问题排查

特征类型详解
----------------

**传统影像组学特征 (traditional)**

使用 pyradiomics 提取标准影像组学特征。

**适用场景：**
- 与传统影像组学研究对比
- 需要标准的影像组学特征

**特征类别：**
- 一阶统计特征（First Order Statistics）
- 形状特征（Shape）
- 灰度共生矩阵特征（GLCM）
- 灰度游程矩阵特征（GLRLM）
- 灰度区域矩阵特征（GLSZM）
- 邻域灰度差分矩阵特征（NGTDM）

**参数文件示例：**

.. code-block:: yaml

   # parameter.yaml
   imageType: original
   resampledPixelSpacing: None
   interpolator: sitkBSpline
   force2D: false
   force2Ddimension: 0
   resampling: None
   preCrop: false
   binWidth: 25
   binCount: 32
   normalize: false
   normalizeScale: 1
   removeOutliers: false
   voxelBased: true

   # 特征类别
   shape: enabled
   firstorder: enabled
   glcm: enabled
   glrlm: enabled
   glszm: enabled
   ngtdm: enabled

**非影像组学特征 (non_radiomics)**

提取非标准的影像组学特征。

**适用场景：**
- 探索新的特征类型
- 研究特定的生物学特征

**特征类型：**
- 空间分布特征
- 强度分布特征
- 形态学特征
- 纹理特征

**整体生境特征 (whole_habitat)**

提取整个生境的特征。

**适用场景：**
- 研究生境的整体特性
- 分析生境之间的关系

**特征类型：**
- 生境数量
- 生境分布
- 生境大小
- 生境形状

**每个生境特征 (each_habitat)**

提取每个生境的特征。

**适用场景：**
- 研究每个生境的独立特性
- 分析生境之间的差异

**特征类型：**
- 每个生境的强度特征
- 每个生境的纹理特征
- 每个生境的形状特征

**MSI 特征 (msi)**

空间交互矩阵（Spatial Interaction Matrix）特征，描述不同生境区域之间的空间交互关系。

**适用场景：**
- 研究生境之间的空间交互模式
- 量化肿瘤内空间异质性

**特征类型：**
- 生境间空间邻接/共现相关指标
- 生境间相互作用矩阵派生统计量

**参考：**
Intratumoral Spatial Heterogeneity at Perfusion MR Imaging Predicts Recurrence-free Survival in Locally Advanced Breast Cancer Treated with Neoadjuvant Chemotherapy

**ITH 特征 (ith_score)**

肿瘤内异质性特征。

**适用场景：**
- 量化肿瘤内异质性
- 研究异质性与临床结果的关系

**特征类型：**
- 异质性指数
- 多样性指数
- 复杂性指数

参数文件说明
------------

HABIT 使用 pyradiomics 进行特征提取，需要提供参数文件。

**parameter.yaml（原始图像特征提取）：**

.. code-block:: yaml

   # 图像设置
   imageType: original
   resampledPixelSpacing: None
   interpolator: sitkBSpline
   force2D: false
   force2Ddimension: 0
   resampling: None
   preCrop: false

   # 分箱设置
   binWidth: 25
   binCount: 32
   normalize: false
   normalizeScale: 1
   removeOutliers: false
   voxelBased: true

   # 特征类别
   shape: enabled
   firstorder: enabled
   glcm: enabled
   glrlm: enabled
   glszm: enabled
   ngtdm: enabled

**parameter_habitat.yaml（生境图特征提取）：**

.. code-block:: yaml

   # 图像设置
   imageType: original
   resampledPixelSpacing: None
   interpolator: sitkBSpline
   force2D: false
   force2Ddimension: 0
   resampling: None
   preCrop: false

   # 分箱设置
   binWidth: 25
   binCount: 32
   normalize: false
   normalizeScale: 1
   removeOutliers: false
   voxelBased: true

   # 特征类别
   shape: enabled
   firstorder: enabled
   glcm: enabled
   glrlm: enabled
   glszm: enabled
   ngtdm: enabled

实际示例
--------

**示例 1: 基本特征提取**

.. code-block:: yaml

   params_file_of_non_habitat: ./parameter.yaml
   params_file_of_habitat: ./parameter_habitat.yaml

   raw_img_folder: ./preprocessed/processed_images
   habitats_map_folder: ./results/habitat/predict
   out_dir: ./results/features

   n_processes: 3
   habitat_pattern: '*_habitats.nrrd'

   feature_types:
     - traditional
     - non_radiomics
     - whole_habitat

   n_habitats:

   debug: false

**示例 2: 完整特征提取**

.. code-block:: yaml

   params_file_of_non_habitat: ./parameter.yaml
   params_file_of_habitat: ./parameter_habitat.yaml

   raw_img_folder: ./preprocessed/processed_images
   habitats_map_folder: ./results/habitat/predict
   out_dir: ./results/features

   n_processes: 4
   habitat_pattern: '*_habitats.nrrd'

   feature_types:
     - traditional
     - non_radiomics
     - whole_habitat
     - each_habitat
     - msi
     - ith_score

   n_habitats:

   debug: false

输出结构
--------

特征提取的输出结构：

.. code-block:: text

   results/features/
   ├── traditional/              # 传统影像组学特征
   │   ├── features.csv
   │   └── ...
   ├── non_radiomics/           # 非影像组学特征
   │   ├── features.csv
   │   └── ...
   ├── whole_habitat/           # 整体生境特征
   │   ├── features.csv
   │   └── ...
   ├── each_habitat/            # 每个生境特征
   │   ├── features.csv
   │   └── ...
   ├── msi/                    # MSI（Spatial Interaction Matrix）特征
   │   ├── features.csv
   │   └── ...
   ├── ith_score/               # ITH 特征
   │   ├── features.csv
   │   └── ...
   └── feature_extraction.log    # 日志文件

常见问题
--------

**Q1: 特征提取失败怎么办？**

A: 检查以下几点：

1. 配置文件路径是否正确。
2. 参数文件是否存在。
3. 生境图文件是否存在。
4. 原始图像文件是否存在。
5. pyradiomics 是否正确安装。
6. 查看日志文件了解详细错误信息。

**Q2: 如何选择特征类型？**

A: 根据您的研究需求选择：
- **传统影像组学**: 与传统研究对比
- **非影像组学**: 探索新的特征
- **整体生境**: 研究生境整体特性
- **每个生境**: 研究每个生境独立特性
- **MSI**: 研究生境间空间交互特征
- **ITH**: 量化肿瘤内异质性

**Q3: 如何调整特征提取参数？**

A: 修改参数文件：

- `binWidth`: 分箱宽度。
- `binCount`: 分箱数量。
- `normalize`: 是否标准化。
- 特征类别：启用或禁用特定特征类别。

**Q4: 特征提取速度慢怎么办？**

A: 可以尝试以下方法：

1. 增加 `n_processes` 参数，使用更多并行进程。
2. 减少特征类型，只提取必要的特征。
3. 使用更快的硬件（如 SSD）。
4. 对于大数据集，考虑分批处理。

**Q5: 如何验证提取的特征？**

A: 可以通过以下方法验证：

1. 检查特征文件的完整性。
2. 检查特征值的合理性。
3. 可视化特征分布。
4. 统计特征数量和质量。

下一步
-------

特征提取完成后，您可以：

- :doc:`machine_learning_modeling_zh`: 进行机器学习建模
- :doc:`../customization/index_zh`: 了解如何自定义特征提取器
- :doc:`../configuration_zh`: 了解配置文件的详细说明
