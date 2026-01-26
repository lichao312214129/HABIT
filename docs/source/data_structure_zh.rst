数据结构说明
============

本节详细说明 HABIT 支持的数据输入方式，包括文件夹结构和 YAML 配置文件格式。

**重要提示**: 使用前需要先解压 `demo_data` 目录中的 `demo_data.rar` 压缩包。

解压后会得到以下 demo 数据：

- **DICOM 原始数据**: ``demo_data/dicom/``
- **预处理后的数据**: ``demo_data/preprocessed/``（包含 processed_images 子目录）
- **配置文件**: ``demo_data/config_preprocessing.yaml``、``demo_data/files_preprocessing.yaml`` 等

**解压后的目录结构**：

.. code-block:: text

   demo_data/
   ├── dicom/                              # DICOM 原始数据
   │   ├── sub001/
   │   │   └── WATER_BHAxLAVA-Flex-2min_Series0012/
   │   │       └── Image (0001).dcm
   │   └── sub002/
   ├── preprocessed/                         # 预处理后的数据
   │   └── processed_images/                # 预处理输出目录
   │       ├── images/                     # 图像
   │       │   ├── subj001/
   │       │   │   ├── delay2/
   │       │   │   ├── delay3/
   │       │   │   └── delay5/
   │       │   └── subj002/
   │       └── masks/                      # 掩码
   │           ├── subj001/
   │           │   ├── delay2/
   │           │   ├── delay3/
   │           │   └── delay5/
   │           └── subj002/
   ├── config_preprocessing.yaml              # 预处理配置
   ├── files_preprocessing.yaml               # 文件列表
   ├── file_habitat.yaml                   # 生境分析数据配置
   ├── config_habitat_one_step.yaml          # 生境分析配置（一步法）
   └── ...                                # 其他配置文件

数据输入方式概述
----------------

HABIT 支持两种数据输入方式：

1. **文件夹方式**: 按照固定的文件夹结构组织数据
2. **YAML 配置文件方式**: 通过 YAML 文件指定数据路径（推荐）

**推荐使用 YAML 配置文件方式**，因为它更加灵活，适合复杂的数据组织。

文件夹结构方式
---------------

标准文件夹结构
~~~~~~~~~~~~~~

使用文件夹方式时，数据必须按照以下结构组织：

.. code-block:: text

   data_root/
   ├── images/           # 图像文件夹（固定名称）
   │   ├── subject1/     # 受试者1
   │   │   ├── T1/       # T1图像
   │   │   │   └── T1.nii.gz
   │   │   ├── T2/       # T2图像
   │   │   │   └── T2.nii.gz
   │   │   └── FLAIR/    # FLAIR图像
   │   │       └── FLAIR.nii.gz
   │   └── subject2/
   │       ├── T1/
   │       │   └── T1.nii.gz
   │       ├── T2/
   │       │   └── T2.nii.gz
   │       └── FLAIR/
   │           └── FLAIR.nii.gz
   └── masks/            # 掩码文件夹（固定名称）
       ├── subject1/
       │   ├── T1/
       │   │   └── mask_T1.nii.gz
       │   ├── T2/
       │   │   └── mask_T2.nii.gz
       │   └── FLAIR/
       │       └── mask_FLAIR.nii.gz
       └── subject2/
           ├── T1/
           │   └── mask_T1.nii.gz
           ├── T2/
           │   └── mask_T2.nii.gz
           └── FLAIR/
               └── mask_FLAIR.nii.gz

**关键要点：**

- `images/` 和 `masks/` 是固定的文件夹名称
- 每个受试者（subject）有独立的文件夹
- 每个图像类型（T1、T2、FLAIR 等）有独立的文件夹
- **文件选择规则**：
    - 如果文件夹中包含 DICOM 序列（多个 .dcm 文件），系统会将其作为一个整体读取。
    - 如果文件夹中包含多个 NIfTI (.nii.gz) 或 NRRD 文件，系统**只会自动选择第一个**。
    - 在进行 `dcm2nii` 转换后，建议每个文件夹只存放一个对应的 NIfTI 文件。

**使用示例：**

在配置文件中指定根目录：

.. code-block:: yaml

   data_dir: ./data_root

系统会自动扫描 `images/` 和 `masks/` 文件夹，读取所有受试者的数据。

文件夹命名规则
~~~~~~~~~~~~~~

**受试者命名：**

- 可以使用任何名称，但建议使用有意义的标识符
- 示例：`subject001`、`patient_01`、`subj001` 等
- 避免使用空格和特殊字符

**图像类型命名：**

- 可以使用任何名称，但建议使用标准的医学图像命名
- 示例：`T1`、`T2`、`FLAIR`、`DWI`、`ADC` 等
- 避免使用空格和特殊字符

**文件命名：**

- 对于非 DICOM 数据，系统默认选择文件夹中的第一个文件。
- 文件格式支持：`.nii.gz`、`.nrrd`、`.dcm` 等。
- 建议使用有意义的文件名，且一个文件夹下只放一个目标文件。

**注意事项：**

- 文件夹和文件名不能包含空格和特殊字符
- 推荐使用英文命名，避免编码问题
- 保持命名的一致性，便于管理和维护

YAML 配置文件方式（推荐）
-------------------------

YAML 配置文件结构
~~~~~~~~~~~~~~~~~~

使用 YAML 配置文件方式时，数据路径通过 YAML 文件指定：

.. code-block:: yaml

   # 控制是否自动读取目录中的第一个文件
   # true: 自动读取目录中的第一个文件 (适用于已转换的nii文件等场景)
   # false: 保持目录路径不变 (适用于dcm2nii等需要整个文件夹的任务)
   auto_select_first_file: true

   images:
     subject1:
       T1: /path/to/subject1/T1/T1.nii.gz
       T2: /path/to/subject1/T2/T2.nii.gz
       FLAIR: /path/to/subject1/FLAIR/FLAIR.nii.gz
     subject2:
       T1: /path/to/subject2/T1/T1.nii.gz
       T2: /path/to/subject2/T2/T2.nii.gz
       FLAIR: /path/to/subject2/FLAIR/FLAIR.nii.gz

   masks:
     subject1:
       T1: /path/to/subject1/T1/mask_T1.nii.gz
       T2: /path/to/subject2/T2/mask_T2.nii.gz
       FLAIR: /path/to/subject1/FLAIR/mask_FLAIR.nii.gz
     subject2:
       T1: /path/to/subject2/T1/mask_T1.nii.gz
       T2: /path/to/subject2/T2/mask_T2.nii.gz
       FLAIR: /path/to/subject2/FLAIR/mask_FLAIR.nii.gz

**字段说明：**

- `auto_select_first_file`: 控制是否自动读取目录中的第一个文件
- `images`: 图像路径配置
- `masks`: 掩码路径配置（可选）

路径格式
~~~~~~~~

**完整文件路径：**

.. code-block:: yaml

   images:
     subject1:
       T1: /path/to/subject1/T1/T1.nii.gz

**文件夹路径（推荐）：**

.. code-block:: yaml

   images:
     subject1:
       T1: /path/to/subject1/T1/

系统会自动选择文件夹中的第一个文件。

**相对路径：**

.. code-block:: yaml

   images:
     subject1:
       T1: ./data/subject1/T1/T1.nii.gz

路径相对于配置文件的位置。

**混合使用：**

可以在同一个配置文件中混合使用完整文件路径和文件夹路径：

.. code-block:: yaml

   images:
     subject1:
       T1: /path/to/subject1/T1/           # 文件夹路径
       T2: /path/to/subject1/T2/T2.nii.gz  # 完整文件路径
       FLAIR: /path/to/subject1/FLAIR/      # 文件夹路径

auto_select_first_file 参数详解
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**参数作用：**

- `true`: 自动读取目录中的第一个文件（适用于已转换的 nii 文件等场景）
- `false`: 保持目录路径不变（适用于 dcm2nii 等需要整个文件夹的任务）

**使用场景：**

**场景 1: 预处理阶段（dcm2nii）**

对于 dcm2nii 等需要整个文件夹的任务，必须设置为 `false`：

.. code-block:: yaml

   # files_preprocessing.yaml
   auto_select_first_file: false  # 必须为false，因为dcm2nii需要整个文件夹

   images:
     subj001:
       delay2: ./dicom/sub001/WATER_BHAxLAVA-Flex-2min_Series0012  # 文件夹路径
       delay3: ./dicom/sub001/WATER_BHAxLAVA-Flex-3min_Series0014  # 文件夹路径
       delay5: ./dicom/sub001/WATER_BHAxLAVA-Flex-5min_Series0016  # 文件夹路径

**场景 2: 生境分析阶段（已转换的 nii 文件）**

对于已转换的 nii 文件，可以设置为 `true`：

.. code-block:: yaml

   # file_habitat.yaml
   auto_select_first_file: true  # 可以为true，自动读取第一个nii文件

   images:
     subj001:
       delay2: ./preprocessed/processed_images/images/subj001/delay2  # 文件夹路径
       delay3: ./preprocessed/processed_images/images/subj001/delay3  # 文件夹路径
       delay5: ./preprocessed/processed_images/images/subj001/delay5  # 文件夹路径

   masks:
     subj001:
       delay2: ./preprocessed/processed_images/masks/subj001/delay2  # 文件夹路径
       delay3: ./preprocessed/processed_images/masks/subj001/delay3  # 文件夹路径
       delay5: ./preprocessed/processed_images/masks/subj001/delay5  # 文件夹路径

路径和文件名规则
~~~~~~~~~~~~~~~

**重要规则：**

1. **不能有空格**: 路径和文件名不能包含空格
2. **不能有特殊字符**: 避免使用 `!@#$%^&*()` 等特殊字符
3. **推荐使用英文**: 避免中文路径，防止编码问题
4. **使用下划线**: 如果需要分隔单词，使用下划线 `_` 而不是空格

**正确示例：**

.. code-block:: yaml

   images:
     subject_001:
       T1: ./data/subject_001/T1/T1.nii.gz
       T2: ./data/subject_001/T2/T2.nii.gz

**错误示例：**

.. code-block:: yaml

   images:
     subject 001:           # 错误：包含空格
       T1: ./data/subject 001/T1/T1.nii.gz  # 错误：包含空格
       T2: ./data/subject_001/T2/T2@image.nii.gz  # 错误：包含特殊字符@

实际示例
~~~~~~~~

**示例 1: 预处理数据配置**

基于 `demo_data/files_preprocessing.yaml`：

.. code-block:: yaml

   # 控制是否自动读取目录中的第一个文件
   auto_select_first_file: false

   images:
     subj001:
       delay2: ./dicom/sub001/WATER_BHAxLAVA-Flex-2min_Series0012
       delay3: ./dicom/sub001/WATER_BHAxLAVA-Flex-3min_Series0014
       delay5: ./dicom/sub001/WATER_BHAxLAVA-Flex-5min_Series0016
     subj002:
       delay2: ./dicom/sub002/013_WATERBHAxLAVAFlex2min
       delay3: ./dicom/sub002/015_WATERBHAxLAVAFlex3min
       delay5: ./dicom/sub002/016_WATERWATERBHAxLAVAFlex5min

**示例 2: 生境分析数据配置**

基于 `demo_data/file_habitat.yaml`：

.. code-block:: yaml

   # 控制是否自动读取目录中的第一个文件
   auto_select_first_file: true

   images:
     subj001:
       delay2: .\preprocessed\processed_images\images\subj001\delay2
       delay3: .\preprocessed\processed_images\images\subj001\delay3
       delay5: .\preprocessed\processed_images\images\subj001\delay5
     subj002:
       delay2: .\preprocessed\processed_images\images\subj002\delay2
       delay3: .\preprocessed\processed_images\images\subj002\delay3
       delay5: .\preprocessed\processed_images\images\subj002\delay5

   masks:
     subj001:
       delay2: .\preprocessed\processed_images\masks\subj001\delay2
       delay3: .\preprocessed\processed_images\masks\subj001\delay3
       delay5: .\preprocessed\processed_images\masks\subj001\delay5
     subj002:
       delay2: .\preprocessed\processed_images\masks\subj002\delay2
       delay3: .\preprocessed\processed_images\masks\subj002\delay3
       delay5: .\preprocessed\processed_images\masks\subj002\delay5

两种方式对比
------------

.. list-table:: 两种方式对比
   :widths: 20 40 40
   :header-rows: 1

   * - 特性
     - 文件夹方式
     - YAML 配置文件方式
   * - **灵活性**
     - 低（必须遵循固定结构）
     - 高（可以自由组织）
   * - **适用场景**
     - 简单项目、快速原型
     - 复杂项目、生产环境
   * - **维护性**
     - 低（需要手动管理文件夹）
     - 高（配置文件易于管理）
   * - **可读性**
     - 中（需要查看文件夹结构）
     - 高（配置文件清晰明了）
   * - **版本控制**
     - 困难（文件夹不便于版本控制）
     - 容易（配置文件易于版本控制）
   * - **共享性**
     - 困难（需要共享整个文件夹）
     - 容易（只需共享配置文件）
   * - **推荐度**
     - 低
     - **高**

**推荐使用 YAML 配置文件方式**，原因如下：

1. **更灵活**: 可以自由组织数据，不受固定结构限制
2. **更易维护**: 配置文件易于管理和修改
3. **更易共享**: 只需分享配置文件，不需要分享整个数据集
4. **更易版本控制**: 配置文件可以纳入版本控制，便于追踪变更
5. **更清晰**: 配置文件结构清晰，易于理解

转换方法
--------

**从文件夹方式转换为 YAML 配置文件方式：**

1. 创建一个新的 YAML 文件
2. 按照上述 YAML 配置文件结构填写路径
3. 在配置文件中指定 YAML 文件路径

**示例：**

假设您有以下文件夹结构：

.. code-block:: text

   data_root/
   ├── images/
   │   ├── subject1/
   │   │   ├── T1/
   │   │   │   └── T1.nii.gz
   │   │   └── T2/
   │   │       └── T2.nii.gz
   │   └── subject2/
   │       ├── T1/
   │       │   └── T1.nii.gz
   │       └── T2/
   │           └── T2.nii.gz
   └── masks/
       ├── subject1/
       │   ├── T1/
       │   │   └── mask_T1.nii.gz
       │   └── T2/
       │       └── mask_T2.nii.gz
       └── subject2/
           ├── T1/
           │   └── mask_T1.nii.gz
           └── T2/
               └── mask_T2.nii.gz

转换为 YAML 配置文件：

.. code-block:: yaml

   auto_select_first_file: true

   images:
     subject1:
       T1: ./data_root/images/subject1/T1/
       T2: ./data_root/images/subject1/T2/
     subject2:
       T1: ./data_root/images/subject2/T1/
       T2: ./data_root/images/subject2/T2/

   masks:
     subject1:
       T1: ./data_root/masks/subject1/T1/
       T2: ./data_root/masks/subject1/T2/
     subject2:
       T1: ./data_root/masks/subject2/T1/
       T2: ./data_root/masks/subject2/T2/

然后在配置文件中指定：

.. code-block:: yaml

   data_dir: ./data_config.yaml

数据验证
--------

HABIT 会自动验证数据路径的有效性：

1. **检查路径是否存在**: 如果路径不存在，会发出警告
2. **检查文件格式**: 支持的格式包括 `.nii.gz`、`.nrrd`、`.dcm` 等
3. **检查文件完整性**: 确保文件可以正常读取

**错误处理：**

如果数据路径有问题，HABIT 会：

- 在日志中记录警告信息
- 在控制台输出警告信息
- 跳过有问题的文件，继续处理其他文件

**建议：**

- 在运行前检查所有路径是否正确
- 确保文件格式正确
- 检查文件权限，确保可以正常读取

下一步
-------

现在您已经了解了 HABIT 的数据结构，可以：

- 阅读 :doc:`../user_guide/index_zh` 了解详细的使用指南
- 查看 :doc:`../tutorials/index_zh` 学习完整的教程
- 参考 :doc:`configuration_zh` 了解配置文件的详细说明
