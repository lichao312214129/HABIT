数据结构说明
============

本节详细说明 HABIT 支持的数据输入方式，包括文件夹结构和 YAML 配置文件格式。

**工作根目录**

Demo 与多数 YAML 假定 **工作根目录** 下同时有 ``config/`` 与 ``demo_data/`` ：

- **Windows 便携包**：pack 根目录（如 ``D:\habit-cpu\`` ）；``config.zip`` 、``demo_data.rar`` 从网盘解压到此（见 :doc:`getting_started/installation_zh`）
- **源码安装**：GitHub 仓库 / ZIP 根目录；``demo_data.rar`` 从网盘解压到 ``demo_data/``

**重要提示**: 使用前需要先解压网盘中的 ``demo_data.rar`` 到工作根目录下的 ``demo_data/`` ；``config.zip`` 解压得到 ``config/`` 。

解压后会得到以下 demo 数据：

- **DICOM 原始数据**: ``demo_data/dicom/``
- **预处理后的数据**: ``demo_data/preprocessed/`` （包含 processed_images 子目录）
- **配置文件**: ``config/preprocessing/config_preprocessing_demo.yaml`` 、``config/preprocessing/files_preprocessing.yaml`` 等

**解压后的目录结构** ：

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
   ├── ml_data/                            # 机器学习演示 CSV（输入表、train/test ID，非运行输出）
   ├── results/                            # 文档 Demo YAML 的运行输出（预处理、生境、特征、ML、对比等）
   │   ├── preprocessed/processed_images/  # 步骤 1 输出（可选；包内亦可直接用 preprocessed/ 下数据）
   │   ├── habitat_two_step/               # 步骤 2
   │   ├── features/                       # 步骤 3
   │   ├── ml/                             # 步骤 4（radiomics、clinical、kfold 等）
   │   └── model_comparison/               # 步骤 5
   ├── configs/                            # 按流程分类的 YAML（推荐从这里选用）
   │   ├── preprocess/
   │   ├── habitat/
   │   ├── extract/
   │   ├── machine_learning/
   │   ├── model_comparison/
   │   ├── auxiliary/
   │   ├── parameters/
   │   └── manifests/
   └── ...                                 # 其他数据与输出目录（results 等）

数据输入方式概述
----------------

各流程在 **对应模块的 YAML** 里指定数据位置，最常见为顶层 ``data_dir`` （与 ``out_dir`` 同级），例如：

- **预处理** （``habit preprocess``）： ``config/preprocessing/*.yaml`` 中的 ``data_dir``
- **生境分析** （``habit get-habitat``）： ``config/habitat/*.yaml`` 中的 ``data_dir``
- **DICOM 整理** （``habit sort-dicom``）： ``config/dicom_sort/*.yaml`` 中的 ``data_dir``

特征提取（``habit extract``）使用 ``raw_img_folder`` 、``habitats_map_folder`` 等字段，含义与 ``data_dir`` 类似，详见 :doc:`configuration_zh` 中特征提取小节。

``data_dir`` 可填 **目录** 或 **路径清单 YAML** （二者解析规则相同；下文仍称「文件夹方式」「YAML 清单方式」）：

1. **文件夹方式** ： ``data_dir`` 指向含固定 ``images/`` 、``masks/`` 结构的根目录
2. **YAML 清单方式** ： ``data_dir`` 指向列出各受试者/序列路径的 YAML（如 ``files_preprocessing.yaml`` 、``file_habitat.yaml``；**推荐** ，更灵活）

**推荐使用 YAML 清单方式** ，适合复杂目录与非标准命名；字段说明与 Demo 路径见 :doc:`configuration_zh`。

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
- **文件选择规则** ：
    - 如果文件夹中包含 DICOM 序列（多个 .dcm 文件），系统会将其作为一个整体读取。
    - 如果文件夹中包含多个 NIfTI (.nii.gz) 或 NRRD 文件，系统 **只会自动选择第一个** 。
    - 在进行 `dcm2nii` 转换后，建议每个文件夹只存放一个对应的 NIfTI 文件。

**使用示例（预处理 / 生境分析等 YAML 中的 ``data_dir``）：**

文件夹方式 — 在模块配置里将 ``data_dir`` 设为数据根目录：

.. code-block:: yaml

   data_dir: ./data_root
   out_dir: ./results

系统会自动扫描 ``data_root/images/`` 与 ``data_root/masks/`` ，读取各受试者数据。

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

YAML 清单方式（推荐）
---------------------

路径清单 YAML 结构
~~~~~~~~~~~~~~~~~~

当 ``data_dir`` 指向一份路径清单 YAML 时，受试者与各序列路径在该文件中列出（再由预处理 / 生境分析等配置引用）：

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

基于 `config/preprocessing/files_preprocessing.yaml`：

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

基于 `config/habitat/file_habitat.yaml`：

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
     - YAML 清单方式
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

**推荐使用 YAML 清单方式** ，原因如下：

1. **更灵活**: 可以自由组织数据，不受固定结构限制
2. **更易维护**: 配置文件易于管理和修改
3. **更易共享**: 只需分享配置文件，不需要分享整个数据集
4. **更易版本控制**: 配置文件可以纳入版本控制，便于追踪变更
5. **更清晰**: 配置文件结构清晰，易于理解

转换方法
--------

**从文件夹方式转换为 YAML 清单方式：**

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

然后在 **预处理 / 生境分析** 等模块 YAML 中指定：

.. code-block:: yaml

   data_dir: ./data_config.yaml
   out_dir: ./results

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
- 查看 :doc:`tutorials/index` 学习完整的教程（英文教程索引）
- 参考 :doc:`configuration_zh` 了解配置文件的详细说明
