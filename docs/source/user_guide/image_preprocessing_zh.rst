图像预处理
========

本节介绍如何使用 HABIT 进行医学图像预处理。

概述
----

图像预处理是生境分析的第一步，目的是提高图像质量，统一图像格式和空间分辨率，为后续的生境分割和特征提取做好准备。

HABIT 提供了丰富的预处理方法，包括：

- **DICOM 转换**: 将 DICOM 格式转换为 NIfTI 格式
- **N4 偏置场校正**: 校正 MRI 图像的偏置场
- **重采样**: 统一图像的空间分辨率
- **配准**: 将多时相图像配准到同一空间
- **标准化**: 对图像强度进行标准化处理
- **自适应直方图均衡化**: 增强图像对比度

CLI 使用方法
------------

**基本语法：**

.. code-block:: bash

   habit preprocess --config <config_file>

**参数说明：**

- `--config`, `-c`: 配置文件路径（必需）

**使用示例：**

.. code-block:: bash

   habit preprocess --config ./demo_data/config_preprocessing.yaml

**输出：**

预处理后的图像将保存在配置文件中指定的输出目录。

Python API 使用方法
------------------

**基本用法：**

.. code-block:: python

   from habit.core.common.service_configurator import ServiceConfigurator
   from habit.core.preprocessing.config_schemas import PreprocessingConfig

   # 加载配置
   config = PreprocessingConfig.from_file('./config_preprocessing.yaml')

   # 创建配置器
   configurator = ServiceConfigurator(config=config)

   # 创建预处理器
   processor = configurator.create_batch_processor()

   # 运行预处理
   processor.process_batch()

**详细示例：**

.. code-block:: python

   import logging
   from habit.core.common.service_configurator import ServiceConfigurator
   from habit.core.preprocessing.config_schemas import PreprocessingConfig
   from habit.utils.log_utils import setup_logger
   from pathlib import Path

   # 设置日志
   output_dir = Path('./preprocessed')
   output_dir.mkdir(parents=True, exist_ok=True)
   logger = setup_logger(
       name='preprocessing',
       output_dir=output_dir,
       log_filename='preprocessing.log',
       level=logging.INFO
   )

   # 加载配置
   config = PreprocessingConfig.from_file('./config_preprocessing.yaml')

   # 创建配置器
   configurator = ServiceConfigurator(config=config, logger=logger, output_dir=str(output_dir))

   # 创建预处理器
   processor = configurator.create_batch_processor()

   # 运行预处理
   logger.info("开始图像预处理")
   processor.process_batch()
   logger.info("预处理完成！")

YAML 配置详解
--------------

**配置文件结构：**

.. code-block:: yaml

   # 数据路径
   data_dir: ./files_preprocessing.yaml
   out_dir: ./preprocessed

   # 预处理设置
   Preprocessing:
     dcm2nii:
       images: [delay2, delay3, delay5]
       dcm2niix_path: ./dcm2niix.exe
       compress: true
       anonymize: true

     n4_correction:
       images: [delay2, delay3, delay5]
       num_fitting_levels: 4

     resample:
       images: [delay2, delay3, delay5]
       target_spacing: [1.0, 1.0, 1.0]

     registration:
       images: [delay2, delay3, delay5]
       fixed_image: delay2
       moving_images: [delay3, delay5]
       type_of_transform: SyNRA
       use_mask: false

     zscore_normalization:
       images: [delay2, delay3, delay5]
       only_inmask: false
       mask_key: mask

     adaptive_histogram_equalization:
       images: [delay2, delay3, delay5]
       alpha: 0.3
       beta: 0.3
       radius: 5

   # 保存选项
   save_options:
     save_intermediate: true
     intermediate_steps: [dcm2nii, n4_correction, resample]

   # 通用设置
   processes: 2
   random_state: 42

**字段说明：**

**data_dir**: 数据目录路径，可以是文件夹或 YAML 配置文件

**out_dir**: 输出目录路径

**Preprocessing**: 预处理设置

- **dcm2nii**: DICOM 转换设置
  - `images`: 要转换的图像列表
  - `dcm2niix_path`: dcm2niix 可执行文件路径
  - `compress`: 是否压缩输出文件（true/false）
  - `anonymize`: 是否匿名化（true/false）

- **n4_correction**: N4 偏置场校正设置
  - `images`: 要校正的图像列表
  - `num_fitting_levels`: 拟合级别数（2-4）

- **resample**: 重采样设置
  - `images`: 要重采样的图像列表
  - `target_spacing`: 目标间距 [x, y, z]（单位：mm）

- **registration**: 配准设置
  - `images`: 所有涉及的图像列表
  - `fixed_image`: 固定图像（参考图像）
  - `moving_images`: 要配准的图像列表
  - `type_of_transform`: 变换类型（SyNRA、SyN、Affine 等）
  - `use_mask`: 是否使用掩码引导配准（true/false）
  - `mask_key`: 掩码键名（当 use_mask 为 true 时）

- **zscore_normalization**: Z-Score 标准化设置
  - `images`: 要标准化的图像列表
  - `only_inmask`: 是否仅在掩码内计算统计量（true/false）
  - `mask_key`: 掩码键名（当 only_inmask 为 true 时）

- **adaptive_histogram_equalization**: 自适应直方图均衡化设置
  - `images`: 要均衡化的图像列表
  - `alpha`: 全局对比度增强因子 [0, 1]
  - `beta`: 局部对比度增强因子 [0, 1]
  - `radius`: 局部窗口半径（像素）

**save_options**: 保存选项

- `save_intermediate`: 是否保存中间结果（true/false）
- `intermediate_steps`: 要保存的中间步骤列表（空列表表示保存所有步骤）

**processes**: 并行进程数

**random_state**: 随机种子

预处理方法详解
----------------

**DICOM 转换 (dcm2nii)**

将 DICOM 格式转换为 NIfTI 格式。

**适用场景：**
- 原始数据为 DICOM 格式
- 需要将 DICOM 转换为 NIfTI 格式

**参数说明：**
- `dcm2niix_path`: dcm2niix 可执行文件路径
- `compress`: 是否压缩输出文件
- `anonymize`: 是否匿名化（移除患者信息）

**注意事项：**
- 确保 dcm2niix 可执行文件有执行权限
- 对于大量 DICOM 文件，转换可能需要较长时间

**N4 偏置场校正 (n4_correction)**

校正 MRI 图像的偏置场，改善图像质量。

**适用场景：**
- MRI 图像存在偏置场不均匀
- 需要改善图像质量

**参数说明：**
- `num_fitting_levels`: 拟合级别数（2-4，级别越高，校正越精细，但计算时间越长）

**注意事项：**
- 主要用于 MRI 图像
- 拟合级别数越高，计算时间越长

**重采样 (resample)**

统一图像的空间分辨率。

**适用场景：**
- 不同图像的空间分辨率不一致
- 需要统一图像分辨率

**参数说明：**
- `target_spacing`: 目标间距 [x, y, z]（单位：mm）

**注意事项：**
- 重采样会改变图像的物理尺寸
- 选择合适的目标间距，避免过度重采样

**配准 (registration)**

将多时相图像配准到同一空间。

**适用场景：**
- 多时相图像存在空间不匹配
- 需要将图像配准到同一空间

**参数说明：**
- `fixed_image`: 固定图像（参考图像）
- `moving_images`: 要配准的图像列表
- `type_of_transform`: 变换类型
  - `SyNRA`: 对称归一化互相关（推荐）
  - `SyN`: 对称归一化
  - `Affine`: 仿射变换
- `use_mask`: 是否使用掩码引导配准
- `mask_key`: 掩码键名

**注意事项：**
- 配准是一个计算密集型操作，可能需要较长时间
- 选择合适的变换类型，平衡精度和计算时间

**Z-Score 标准化 (zscore_normalization)**

将图像强度标准化为均值 0、标准差 1 的分布。

**适用场景：**
- 需要统一图像强度分布
- 为机器学习准备数据

**参数说明：**
- `only_inmask`: 是否仅在掩码内计算统计量
- `mask_key`: 掩码键名（当 only_inmask 为 true 时）

**注意事项：**
- 标准化会改变图像的绝对强度值
- 对于仅掩码内标准化，确保掩码正确

**自适应直方图均衡化 (adaptive_histogram_equalization)**

增强图像对比度。

**适用场景：**
- 图像对比度不足
- 需要增强图像细节

**参数说明：**
- `alpha`: 全局对比度增强因子 [0, 1]
  - 0: 不进行全局对比度增强
  - 1: 完全进行全局对比度增强
- `beta`: 局部对比度增强因子 [0, 1]
  - 0: 不进行局对比度增强
  - 1: 完全进行局对比度增强
- `radius`: 局部窗口半径（像素）

**注意事项：**
- 增强因子不宜过大，避免过度增强
- 半径值不宜过大，避免平滑细节

自定义预处理器
------------

HABIT 支持自定义预处理器，您可以添加自己的预处理方法。

**步骤 1: 创建自定义预处理器**

从模板文件开始：

.. code-block:: python

   from habit.core.preprocessing.preprocessor_factory import PreprocessorFactory
   from habit.core.preprocessing.base_preprocessor import BasePreprocessor

   @PreprocessorFactory.register("my_preprocessor")
   class MyPreprocessor(BasePreprocessor):
       def __init__(self, keys, allow_missing_keys=False, **kwargs):
           super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
           # 初始化参数
           self.param1 = kwargs.get('param1', default_value)
           self.param2 = kwargs.get('param2', default_value)

       def __call__(self, data):
           self._check_keys(data)
           for key in self.keys:
               data[key] = self._process_item(data[key])
           return data

       def _process_item(self, item):
           # 实现您的预处理逻辑
           return processed_item

**步骤 2: 在配置文件中使用**

.. code-block:: yaml

   Preprocessing:
     my_preprocessor:
       images: [T1, T2]
       param1: value1
       param2: value2

**步骤 3: 运行预处理**

.. code-block:: bash

   habit preprocess --config config_with_custom_preprocessor.yaml

实际示例
--------

**示例 1: 基本预处理**

基于 `demo_data/config_preprocessing.yaml`：

.. code-block:: yaml

   data_dir: ./files_preprocessing.yaml
   out_dir: ./preprocessed

   Preprocessing:
     dcm2nii:
       images: [delay2, delay3, delay5]
       dcm2niix_path: ./dcm2niix.exe
       compress: true
       anonymize: true

     resample:
       images: [delay2, delay3, delay5]
       target_spacing: [1.0, 1.0, 1.0]

     zscore_normalization:
       images: [delay2, delay3, delay5]
       only_inmask: false

   processes: 2
   random_state: 42

**示例 2: 完整预处理流程**

包含多个预处理步骤：

.. code-block:: yaml

   data_dir: ./files_preprocessing.yaml
   out_dir: ./preprocessed

   Preprocessing:
     dcm2nii:
       images: [delay2, delay3, delay5]
       dcm2niix_path: ./dcm2niix.exe
       compress: true

     n4_correction:
       images: [delay2, delay3, delay5]
       num_fitting_levels: 4

     resample:
       images: [delay2, delay3, delay5]
       target_spacing: [1.0, 1.0, 1.0]

     registration:
       images: [delay2, delay3, delay5]
       fixed_image: delay2
       moving_images: [delay3, delay5]
       type_of_transform: SyNRA
       use_mask: false

     zscore_normalization:
       images: [delay2, delay3, delay5]
       only_inmask: false

   save_options:
     save_intermediate: true
     intermediate_steps: [dcm2nii, n4_correction, resample, registration]

   processes: 4
   random_state: 42

输出结构
--------

预处理后的输出结构：

.. code-block:: text

   preprocessed/
   ├── dcm2nii_01/              # DICOM 转换结果
   │   ├── images/
   │   │   ├── subj001/
   │   │   │   ├── delay2/
   │   │   │   │   └── delay2.nii.gz
   │   │   │   ├── delay3/
   │   │   │   │   └── delay3.nii.gz
   │   │   │   └── delay5/
   │   │   │       └── delay5.nii.gz
   │   │   └── subj002/
   │   │       └── ...
   │   └── masks/
   │       └── ...
   ├── n4_correction_02/          # N4 校正结果
   ├── resample_03/               # 重采样结果
   ├── registration_04/             # 配准结果
   └── processed_images/           # 最终结果（总是保存）
       ├── images/
       │   ├── subj001/
       │   │   ├── delay2/
       │   │   │   └── delay2.nii.gz
       │   │   ├── delay3/
       │   │   │   └── delay3.nii.gz
       │   │   └── delay5/
       │   │       └── delay5.nii.gz
       │   └── subj002/
       │       └── ...
       └── masks/
           └── ...

常见问题
--------

**Q1: 预处理失败怎么办？**

A: 检查以下几点：

- 确认配置文件路径正确。
- 确认数据文件存在且格式正确。
- 检查日志文件了解详细错误信息。
- 确保有足够的磁盘空间。

**Q2: 如何选择预处理步骤？**

A: 根据数据特点选择：

- DICOM 数据：需要 dcm2nii。
- MRI 图像：建议 N4 校正。
- 多时相图像：需要配准。
- 不同分辨率：需要重采样。
- 机器学习：建议标准化。

**Q3: 预处理时间太长怎么办？**

A: 可以尝试：

- 增加并行进程数（`processes` 参数）。
- 跳过不必要的预处理步骤。
- 使用更简单的变换类型（如 Affine 而非 SyNRA）。
- 分批处理数据。

**Q4: 如何验证预处理结果？**

A: 使用医学图像查看器（如 ITK-SNAP）检查：

- 图像质量是否改善。
- 空间对齐是否正确。
- 强度分布是否合理。
- 掩码是否正确。

**Q5: 中间结果占用太多空间怎么办？**

A: 可以：

- 设置 `save_intermediate: false` 只保存最终结果。
- 定期清理中间结果目录。
- 使用压缩格式（.nii.gz）。

下一步
-------

图像预处理完成后，您可以：

- :doc:`habitat_segmentation_zh`: 进行生境分割
- :doc:`habitat_feature_extraction_zh`: 提取生境特征
- :doc:`machine_learning_modeling_zh`: 进行机器学习建模
