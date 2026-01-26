CLI 参考文档
============

概述
----

HABIT 提供了强大的命令行接口（CLI），支持所有核心功能的批处理和自动化。

**主要命令：**

- `habit preprocess`: 图像预处理
- `habit get-habitat`: 生境分析
- `habit extract`: 特征提取
- `habit model`: 机器学习建模
- `habit compare`: 模型对比分析
- `habit icc`: ICC (组内相关系数) 分析
- `habit retest`: Test-retest 重现性分析
- `habit merge-csv`: CSV 文件合并
- `habit dicom-info`: DICOM 信息提取
- `habit dice`: Dice 系数计算
- `habit radiomics`: 传统影像组学特征提取
- `habit cv`: K-fold 交叉验证
- `habit --help`: 显示帮助信息
- `habit --version`: 显示版本信息

通用参数
--------

所有命令都支持以下通用参数：

**--config, -c**: 配置文件路径

- **类型**: 字符串
- **必需**: 是
- **说明**: 指定 YAML 配置文件的路径
- **示例**: `habit preprocess --config config_preprocessing.yaml`

**--debug**: 启用调试模式

- **类型**: 标志
- **必需**: 否
- **说明**: 启用详细的日志输出，便于调试
- **示例**: `habit preprocess --config config.yaml --debug`

**--help, -h**: 显示帮助信息

- **类型**: 标志
- **必需**: 否
- **说明**: 显示命令的帮助信息
- **示例**: `habit preprocess --help`

**--version, -v**: 显示版本信息

- **类型**: 标志
- **必需**: 否
- **说明**: 显示 HABIT 的版本信息
- **示例**: `habit --version`

图像预处理命令
--------------

**命令**: `habit preprocess`

**功能**: 对医学图像进行预处理，包括 DICOM 转换、重采样、配准、标准化等。

**基本用法**:

.. code-block:: bash

   habit preprocess --config config_preprocessing.yaml

**参数**:

- `--config, -c`: 配置文件路径（必需）
- `--debug`: 启用调试模式（可选）

**配置文件示例**:

.. code-block:: yaml

   data_dir: ./files_preprocessing.yaml
   out_dir: ./preprocessed

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

   processes: 2
   random_state: 42

**输出**:

- 预处理后的图像保存在 `out_dir` 指定的目录
- 处理日志保存在 `preprocessing.log`

生境分析命令
--------------

**命令**: `habit get-habitat`

**功能**: 对预处理后的图像进行生境分割和特征提取。

**基本用法**:

.. code-block:: bash

   # 训练模式（以配置文件 run_mode 为准）
   habit get-habitat --config config_habitat.yaml --mode train

   # 预测模式（需要 pipeline_path 或 --pipeline）
   habit get-habitat --config config_habitat.yaml --mode predict --pipeline ./results/habitat_pipeline.pkl

**参数**:

- `--config, -c`: 配置文件路径（必需）
- `--mode`: 运行模式（可选，用于覆盖配置文件中的 `run_mode`）
  - `train`: 训练新的生境分割模型
  - `predict`: 使用预训练模型进行预测
- `--pipeline`: Pipeline 文件路径（predict 模式必需，用于覆盖配置文件中的 `pipeline_path`）
- `--debug`: 启用调试模式（可选）

**配置文件示例**:

.. code-block:: yaml

   run_mode: train
   data_dir: ./file_habitat.yaml
   out_dir: ./results/habitat/train

   FeatureConstruction:
     voxel_level:
       method: concat(raw(delay2), raw(delay3), raw(delay5))
       params: {}

     supervoxel_level:
       supervoxel_file_keyword: '*_supervoxel.nrrd'
       method: mean_voxel_features()
       params:
         params_file: {}

   HabitatsSegmention:
     clustering_mode: two_step

     supervoxel:
       algorithm: kmeans
       n_clusters: 50
       random_state: 42

     habitat:
       algorithm: kmeans
       max_clusters: 10
       habitat_cluster_selection_method:
         - inertia
         - silhouette
       random_state: 42

   processes: 2
   plot_curves: true
   save_results_csv: true
   random_state: 42

**输出**:

- 结果表：`out_dir/habitats.csv`（若启用保存）
- 生境图：`out_dir/<subject>_habitats.nrrd`
- 超像素图（Two-Step）：`out_dir/<subject>_supervoxel.nrrd`
- 可视化图表：`out_dir/visualizations/`
- 处理日志：`out_dir/habitat_analysis.log`

特征提取命令
--------------

**命令**: `habit extract`

**功能**: 从生境图中提取各种特征。

**基本用法**:

.. code-block:: bash

   habit extract --config config_extract_features.yaml

**参数**:

- `--config, -c`: 配置文件路径（必需）
- `--debug`: 启用调试模式（可选）

**配置文件示例**:

.. code-block:: yaml

   params_file_of_non_habitat: ./parameter.yaml
   params_file_of_habitat: ./parameter_habitat.yaml

   raw_img_folder: ./preprocessed/processed_images
   habitats_map_folder: ./results/habitat
   out_dir: ./results/features

   n_processes: 3
   habitat_pattern: '*_habitats.nrrd'

   feature_types:
     - traditional
     - non_radiomics
     - whole_habitat
     - each_habitat
     - msi
     - ith_score

   debug: false

**输出**:

- 特征文件保存在 `out_dir` 指定的目录
- 处理日志保存在 `feature_extraction.log`

机器学习命令
--------------

**命令**: `habit model`

**功能**: 使用提取的特征进行机器学习建模。

**基本用法**:

.. code-block:: bash

   # 训练模式
   habit model --config config_machine_learning.yaml --mode train

   # 预测模式
   habit model --config config_machine_learning.yaml --mode predict --pipeline ./results/ml/model_pipeline.pkl

**参数**:

- `--config, -c`: 配置文件路径（必需）
- `--mode`: 运行模式（可选，默认：train）
  - `train`: 训练新的机器学习模型
  - `predict`: 使用预训练模型进行预测
- `--pipeline`: Pipeline 文件路径（predict 模式必需）
- `--debug`: 启用调试模式（可选）

**配置文件示例** (训练模式):

.. code-block:: yaml

   input:
     - path: ./ml_data/clinical_feature.csv
       subject_id_col: PatientID
       label_col: Label
   output: ./results/ml/train
   random_state: 42
   split_method: stratified
   test_size: 0.3

   FeatureSelection:
     enabled: true
     method: variance
     params:
       threshold: 0.0

   models:
     LogisticRegression:
       params:
         C: 1.0
         solver: liblinear
         random_state: 42

**配置文件示例** (预测模式):

.. code-block:: yaml

   model_path: ./results/ml/train/models/LogisticRegression_final_pipeline.pkl
   data_path: ./ml_data/new_data.csv
   output_dir: ./results/ml/predict
   evaluate: true
   label_col: Label

   models:
     RandomForest:
       params:
         n_estimators: 100
         max_depth: null
         min_samples_split: 2
         min_samples_leaf: 1
         random_state: 42

   ModelEvaluation:
     enabled: true
     metrics:
       - accuracy
       - precision
       - recall
       - f1
       - roc_auc
       - confusion_matrix
     cv: 5
     test_size: 0.2
     random_state: 42

   ModelSaving:
     enabled: true
     save_path: ./results/ml/model_pipeline.pkl
     save_format: pkl

   processes: 2
   random_state: 42

**输出**:

- 训练好的模型保存在 `save_path` 指定的路径
- 评估结果保存在 `out_dir` 指定的目录
- 评估图表保存在 `out_dir/plots` 目录
- 处理日志保存在 `ml_training.log`

帮助命令
---------

**命令**: `habit --help`

**功能**: 显示 HABIT 的帮助信息，包括所有可用命令和参数。

**基本用法**:

.. code-block:: bash

   habit --help

   # 显示特定命令的帮助
   habit preprocess --help
   habit get-habitat --help
   habit extract --help
   habit model --help

版本命令
---------

**命令**: `habit --version`

**功能**: 显示 HABIT 的版本信息。

**基本用法**:

.. code-block:: bash

   habit --version

命令行参数优先级
----------------

当同一个参数在配置文件和命令行中都指定时，命令行参数的优先级更高。

**优先级顺序**（从高到低）：

1. 命令行参数
2. 配置文件参数
3. 默认值

**示例**:

.. code-block:: bash

   # 配置文件中 processes: 2
   # 命令行中指定 processes: 4
   habit preprocess --config config.yaml --processes 4

   # 实际使用的进程数为 4（命令行参数优先）

错误处理
--------

当命令执行失败时，HABIT 会：

1. **显示错误信息**: 在控制台显示清晰的错误信息
2. **记录错误日志**: 在日志文件中记录详细的错误堆栈
3. **提供解决建议**: 对于常见错误，提供解决建议

**常见错误及解决方法**:

- **配置文件不存在**: 检查配置文件路径是否正确
- **数据路径不存在**: 检查数据路径是否正确
- **参数错误**: 检查配置文件中的参数是否正确
- **依赖缺失**: 安装缺失的依赖包

日志记录
--------

HABIT 会记录详细的日志信息，便于调试和问题排查。

**日志级别**:

- `DEBUG`: 详细的调试信息
- `INFO`: 一般信息
- `WARNING`: 警告信息
- `ERROR`: 错误信息
- `CRITICAL`: 严重错误信息

**日志文件**:

- 每个命令都会生成对应的日志文件
- 日志文件保存在输出目录中
- 日志文件命名格式：`<command>.log`

**启用调试模式**:

.. code-block:: bash

   habit preprocess --config config.yaml --debug

批处理示例
----------

**示例 1: 批量预处理**

.. code-block:: bash

   # 创建批处理脚本
   cat > batch_preprocess.sh << 'EOF'
   #!/bin/bash
   for config in config_*.yaml; do
       echo "Processing $config..."
       habit preprocess --config "$config" --debug
   done
   EOF

   # 运行批处理
   chmod +x batch_preprocess.sh
   ./batch_preprocess.sh

**示例 2: 完整工作流程**

.. code-block:: bash

   # 1. 预处理
   habit preprocess --config config_preprocessing.yaml

   # 2. 生境分析（训练）
   habit get-habitat --config config_habitat.yaml --mode train

   # 3. 特征提取
   habit extract --config config_extract_features.yaml

   # 4. 机器学习（训练）
   habit model --config config_machine_learning.yaml --mode train

**示例 3: 预测模式**

.. code-block:: bash

   # 1. 生境分析（预测）
   habit get-habitat --config config_habitat.yaml --mode predict --pipeline ./results/habitat_pipeline.pkl

   # 2. 特征提取
   habit extract --config config_extract_features.yaml

   # 3. 机器学习（预测）
   habit model --config config_machine_learning.yaml --mode predict --pipeline ./results/ml/model_pipeline.pkl

**示例 4: 模型对比分析**

.. code-block:: bash

   # 模型对比分析
   habit compare --config config_model_comparison.yaml

**示例 5: ICC 分析**

.. code-block:: bash

   # ICC 分析
   habit icc --config config_icc.yaml

**示例 6: Test-Retest 分析**

.. code-block:: bash

   # Test-Retest 分析
   habit retest --config config_test_retest.yaml

**示例 7: CSV 合并**

.. code-block:: bash

   # 合并多个 CSV 文件
   habit merge-csv file1.csv file2.csv file3.csv -o merged.csv --index-col PatientID

**示例 8: DICOM 信息提取**

.. code-block:: bash

   # 提取 DICOM 信息
   habit dicom-info -i ./dicom_directory -o dicom_info.csv --tags "PatientName,StudyDate,Modality"

**示例 9: Dice 系数计算**

.. code-block:: bash

   # 计算 Dice 系数
   habit dice --input1 ./masks1 --input2 ./masks2 --output dice_results.csv

**示例 10: 传统影像组学特征提取**

.. code-block:: bash

   # 传统影像组学特征提取
   habit radiomics --config config_radiomics.yaml

**示例 11: K-fold 交叉验证**

.. code-block:: bash

   # K-fold 交叉验证
   habit cv --config config_machine_learning.yaml

**命令详细说明**:

- `habit compare`: 模型对比分析
  - 参数: `--config, -c` (必需)
  - 功能: 比较多个模型的性能，生成 ROC 曲线、校准曲线等

- `habit icc`: ICC (组内相关系数) 分析
  - 参数: `--config, -c` (必需)
  - 功能: 评估特征在不同扫描条件下的可重复性

- `habit retest`: Test-retest 重现性分析
  - 参数: `--config, -c` (可选，也可以使用命令行参数)
  - 功能: 评估生境映射在测试-重测扫描中的稳定性

- `habit merge-csv`: CSV 文件合并
  - 参数: `input_files` (必需), `--output, -o` (必需), `--index-col, -c` (可选)
  - 功能: 基于索引列水平合并多个 CSV/Excel 文件

- `habit dicom-info`: DICOM 信息提取
  - 参数: `--input, -i` (必需), `--tags, -t` (可选), `--output, -o` (可选)
  - 功能: 提取 DICOM 文件的元数据信息

- `habit dice`: Dice 系数计算
  - 参数: `--input1` (必需), `--input2` (必需), `--output` (可选)
  - 功能: 计算两批图像之间的 Dice 系数

- `habit radiomics`: 传统影像组学特征提取
  - 参数: `--config, -c` (可选)
  - 功能: 提取传统影像组学特征

- `habit cv`: K-fold 交叉验证
  - 参数: `--config, -c` (必需)
  - 功能: 执行 K 折交叉验证评估

常见问题
--------

**Q1: 如何查看命令的详细帮助？**

A: 使用 `--help` 参数：

.. code-block:: bash

   habit preprocess --help

**Q2: 如何启用调试模式？**

A: 使用 `--debug` 参数：

.. code-block:: bash

   habit preprocess --config config.yaml --debug

**Q3: 如何查看 HABIT 的版本？**

A: 使用 `--version` 参数：

.. code-block:: bash

   habit --version

**Q4: 如何批处理多个配置文件？**

A: 使用 shell 脚本或循环：

.. code-block:: bash

   for config in config_*.yaml; do
       habit preprocess --config "$config"
   done

**Q5: 如何处理命令执行失败？**

A: 查看错误信息和日志文件：

.. code-block:: bash

   # 查看日志文件
   cat preprocessing.log

   # 使用调试模式重新运行
   habit preprocess --config config.yaml --debug

下一步
-------

现在您已经了解了 CLI 的使用方法，可以：

- :doc:`../configuration_zh`: 了解配置文件的详细说明
- :doc:`../user_guide/index_zh`: 了解各个功能的详细使用指南
- :doc:`../customization/index_zh`: 了解如何自定义和扩展功能
