# HABIT - Habitat Analysis Based on Imaging Traits

HABIT是一个用于医学图像生境分析的综合工具包，提供了放射组学特征提取、生境聚类分析、机器学习模型构建等功能。本工具包采用模块化设计，遵循Python最佳实践规范。

HABIT implements a two-step clustering approach to identify distinct tumor habitats based on imaging features:
1. Individual-level clustering: Divide each tumor into supervoxels
2. Population-level clustering: Cluster supervoxels across all subjects to identify habitats

## 项目结构

```
HABIT/
├── habit/                  # 主代码包
│   ├── core/               # 核心功能模块
│   │   ├── habitat_analysis/  # 生境分析模块
│   │   └── machine_learning/  # 机器学习模块
│   └── utils/              # 工具函数
├── scripts/                # 应用程序入口点
│   ├── extract_features.py     # 特征提取脚本
│   ├── generate_habitat_map.py # 生境图生成脚本
│   └── run_machine_learning.py # 机器学习分析脚本
├── tests/                  # 测试代码
│   ├── unit/                  # 单元测试
│   └── integration/           # 集成测试
├── docs/                   # 文档
│   ├── api/                   # API文档
│   └── user_guide/            # 用户指南
├── config/                 # 配置文件
│   ├── default_params.yaml    # 默认参数
│   └── ml_config.yaml         # 机器学习配置
├── INSTALL.md              # 安装说明
├── README.md               # 项目说明
├── pyproject.toml          # 项目配置和依赖
└── .pre-commit-config.yaml # 代码提交前检查配置
```

## 功能特性

1. **生境分析 (Habitat Analysis)**
   - 影像放射组学特征提取 (Radiomics feature extraction)
   - 基于聚类的生境识别 (Cluster-based habitat identification)
   - 生境特征统计与可视化 (Habitat feature statistics and visualization)
   - 多模态图像支持 (Multi-modal image support)
   - 灵活的特征构建表达式 (Flexible feature construction expressions)

2. **特征提取方法 (Feature Extraction Methods)**
   - 原始强度特征 (Raw intensity features)
   - 动态增强特征 (Kinetic features from dynamic images)
   - 体素级放射组学特征 (Voxel-level radiomics features)
   - 超体素级放射组学特征 (Supervoxel-level radiomics features)
   - 自定义特征提取器 (Custom feature extractors)

3. **聚类算法 (Clustering Algorithms)**
   - K-means聚类 (K-means clustering)
   - 层次聚类 (Hierarchical agglomerative clustering)
   - 谱聚类 (Spectral clustering)
   - 高斯混合模型 (Gaussian Mixture Models)

4. **机器学习分析 (Machine Learning Analysis)**
   - 特征选择与预处理 (Feature selection and preprocessing)
   - 模型训练与评估 (Model training and evaluation)
   - 预测与结果分析 (Prediction and result analysis)
   - 自动参数选择 (Automatic parameter selection)

5. **工具函数 (Utility Functions)**
   - 配置管理 (Configuration management)
   - 结果保存与读取 (Result saving and loading)
   - 可视化工具 (Visualization tools)
   - 进度显示 (Progress display)

## 安装方法

### 依赖环境

- Python 3.8+
- 依赖库: numpy, scipy, scikit-learn, pandas, matplotlib, SimpleITK, pyradiomics

### 安装步骤

1. 克隆仓库:
```bash
git clone https://github.com/yourusername/HABIT.git
cd HABIT
```

2. 创建虚拟环境:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖:
```bash
pip install -e .  # 以开发模式安装
# 或
pip install -r requirements.txt
```

## 使用方法

### 特征提取

```bash
python scripts/extract_features.py --config config/your_config.yaml
```

或者直接使用命令行参数:

```bash
python scripts/extract_features.py \
  --params_file_of_non_habitat parameter.yaml \
  --params_file_of_habitat parameter_habitat.yaml \
  --raw_img_folder /path/to/images \
  --habitats_map_folder /path/to/habitats \
  --out_dir /path/to/output \
  --feature_types traditional non_radiomics whole_habitat each_habitat
```

### 机器学习分析

```bash
python scripts/run_machine_learning.py --config config/ml_config.yaml --mode train
```

预测新数据:

```bash
python scripts/run_machine_learning.py \
  --config config/ml_config.yaml \
  --mode predict \
  --model_file models/your_model.pkl \
  --predict_data data/new_data.csv
```

## 配置文件说明

HABIT使用YAML格式的配置文件来定义分析流程。配置文件分为几个主要部分：

### 生境分析配置 (Habitat Analysis Configuration)

```yaml
# 数据路径
data_dir: /path/to/input/data
out_dir: /path/to/output/directory

# 预处理设置
Preprocessing:
  N4BiasCorrection:
    images: [pre_contrast, LAP, PVP, delay_3min]
    n_iterations: [50, 50, 30, 20]  # 每个尺度级别的迭代次数
    convergence_threshold: 0.001
    shrink_factor: 4
    spline_order: 3
    n_fitting_levels: 4
    bias_field_fwhm: 0.15

  resample:
    images: [pre_contrast, LAP, PVP, delay_3min]
    target_spacing: [1.0, 1.0, 1.0]  # 目标体素间距（毫米）
    mode: bilinear  # 图像插值模式
    mask_mode: nearest  # mask插值模式
    align_corners: false
    anti_aliasing: true
    preserve_range: true

  registration:
    images: [pre_contrast, LAP, PVP, delay_3min]
    fixed_image: PVP
    moving_images: [pre_contrast, LAP, delay_3min]
    type_of_transform: Rigid  # Rigid或Affine
    use_mask: true  # 是否使用mask进行配准
    optimizer_type: gradient_descent
    optimizer_params:
      learning_rate: 0.01
      number_of_iterations: 100
    metric_type: mutual_information
    metric_params:
      number_of_histogram_bins: 50
    interpolator_type: linear

# 特征提取设置
FeatureConstruction:
  voxel_level:
    method: concat(raw(pre_contrast), raw(LAP), raw(PVP), raw(delay_3min), timestamps)
    params:
      params_voxel_radiomics: ./config/params_voxel_radiomics.yaml
      kernelRadius: 2
      timestamps: /path/to/timestamps.xlsx

  supervoxel_level:
    supervoxel_file_keyword: '*_supervoxel.nrrd'
    method: mean_voxel_features()  # 使用体素特征的平均值
    params:
      params_file: ./config/parameter.yaml

  preprocessing:
    methods:
      - method: minmax
        global_normalize: true
      - method: winsorize
        winsor_limits: [0.05, 0.05]
        global_normalize: true

# 生境分割设置
HabitatsSegmention:
  # 超体素聚类设置
  supervoxel:
    algorithm: kmeans
    n_clusters: 50  # 超体素数量
    random_state: 42
    max_iter: 300
    n_init: 10

  # 生境聚类设置
  habitat:
    algorithm: kmeans
    max_clusters: 10  # 最大生境数量
    habitat_cluster_selection_method: inertia  # 聚类数选择方法：inertia, silhouette, calinski_harabasz
    best_n_clusters: null  # 设为null表示自动选择最佳聚类数
    random_state: 42
    max_iter: 300
    n_init: 10

# 通用设置
processes: 2  # 并行处理的进程数
plot_curves: true  # 是否生成和保存评估曲线
random_state: 42  # 随机种子
debug: false  # 是否启用调试模式
```

#### 特征构建表达式 (Feature Construction Expressions)

HABIT使用函数式语法来定义特征提取方法。表达式可以嵌套，格式为：

```
method_name(arg1, arg2, ..., paramName1, paramName2, ...)
```

常用方法包括：

- `raw(image_name)`: 从指定图像提取原始强度值
- `kinetic(image1, image2, ..., timestamps)`: 使用时间戳从多个图像提取动态特征
- `voxel_radiomics(image_name, params_file)`: 从指定图像提取体素级放射组学特征
- `supervoxel_radiomics(image_name, params_file)`: 为每个超体素提取放射组学特征
- `concat(method1(image1), method2(image2), ...)`: 连接多个方法的特征
- `mean_voxel_features()`: 计算每个超体素的体素特征平均值

### 特征提取配置 (Feature Extraction Configuration)

```yaml
# 特征提取配置示例
params_file_of_non_habitat: parameter.yaml
params_file_of_habitat: parameter_habitat.yaml
raw_img_folder: /path/to/images
habitats_map_folder: /path/to/habitats
out_dir: /path/to/output
n_processes: 4
habitat_pattern: '*_habitats.nrrd'
feature_types:
  - traditional
  - non_radiomics
  - whole_habitat
  - each_habitat
  - msi
n_habitats: 0  # 0表示自动检测
mode: both     # extract, parse, both
debug: false
```

### 机器学习配置 (Machine Learning Configuration)

```yaml
# 机器学习配置示例
input:
  - path: /path/to/features.csv
    name: features
    subject_id_col: PatientID
    label_col: Label
output: /path/to/ml_results
test_size: 0.3
random_state: 42
split_method: stratified
scaler: standard  # standard, minmax
feature_selection_methods:
  - name: variance_threshold
    params:
      threshold: 0.1
  - name: select_k_best
    params:
      k: 20
models:
  - name: random_forest
    params:
      n_estimators: 100
      max_depth: 5
  - name: svm
    params:
      C: 1.0
      kernel: rbf
is_visualize: true
is_save_model: true
model_file: /path/to/save/model.pkl
```

更多详细配置说明，请参阅[配置指南](doc/configuration.md)。

## 文档

- [使用指南](doc/usage_guide.md): 详细介绍如何使用HABIT
- [配置指南](doc/configuration.md): 详细解释配置参数
- [API参考](doc/api_reference.md): 类和函数参考

## 贡献指南

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开 Pull Request

## 许可证

此项目基于 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件

## 引用

如果您在研究中使用了HABIT，请引用我们的论文：

```
@article{habit2023,
  title={HABIT: Habitat Analysis Based on Imaging Traits},
  author={Author, A. and Author, B.},
  journal={Journal of Medical Imaging},
  year={2023},
  volume={1},
  pages={1-10}
}
```

## TODO list
- windows 调用Dataloader并行处理问题。可能需要定义到main中，但还是希望其它方式解决。实在不行就自己定义多进程
- 特征构建使用monai的compose模块，但要注意保留原始值
```
class YourTransform(MapTransform):
    def __init__(self, source_key, target_key):
        self.source_key = source_key
        self.target_key = target_key

    def __call__(self, data):
        data[self.target_key] = data[self.source_key].clone()  # 深拷贝
        return data
```

## 联系方式

如有任何问题，请联系项目维护者 lichao19870617@163.com

# HABIT - Medical Image Preprocessing Pipeline

## Image Preprocessing Pipeline

The HABIT preprocessing pipeline provides a comprehensive set of tools for medical image preprocessing, including resampling, registration, and N4 bias field correction.

### Configuration File

The preprocessing pipeline is configured using a YAML file. Here's an example configuration:

```yaml
data_dir: "/path/to/your/data"
out_dir: "/path/to/output"

Preprocessing:
  resample:
    images: [t1, t2, T2FLAIR]  # List of modalities to resample
    target_spacing: [1.0, 1.0, 1.0]  # Target voxel spacing in mm

  registration:
    images: [t2, T2FLAIR]  # Images to register
    fixed_image: t1  # Reference image
    type_of_transform: "SyN"  # Type of transformation
    metric: "MI"  # Similarity metric
    optimizer: "gradient_descent"  # Optimization method
    use_mask: true  # Whether to use masks for registration

```

### Preprocessing Steps

#### 1. Resampling

Resamples images to a target voxel spacing.

Parameters:
- `target_spacing`: Target voxel spacing in mm (e.g., [1.0, 1.0, 1.0])
- `img_mode`: Interpolation mode for images
  - Options: "nearest", "linear", "bilinear", "bspline", "bicubic", "gaussian", "lanczos", "hamming", "cosine", "welch", "blackman"
- `padding_mode`: Padding mode for out-of-bound values
- `align_corners`: Whether to align corners

#### 2. Registration

Registers images to a reference image using ANTs.

Parameters:
- `fixed_image`: Key of the reference image to register to
- `type_of_transform`: Type of transformation
  - Options:
    - "Rigid": Rigid transformation
    - "Affine": Affine transformation
    - "SyN": Symmetric normalization (deformable)
    - "SyNRA": SyN + Rigid + Affine
    - "SyNOnly": SyN without initial rigid/affine
    - "TRSAA": Translation + Rotation + Scaling + Affine
    - "Elastic": Elastic transformation
    - "SyNCC": SyN with cross-correlation metric
    - "SyNabp": SyN with mutual information metric
    - "SyNBold": SyN optimized for BOLD images
    - "SyNBoldAff": SyN + Affine for BOLD images
    - "SyNAggro": SyN with aggressive optimization
    - "TVMSQ": Time-varying diffeomorphism with mean square metric
- `metric`: Similarity metric
  - Options: "CC" (Cross-correlation), "MI" (Mutual information), "MeanSquares", "Demons"
- `optimizer`: Optimization method
  - Options: "gradient_descent", "lbfgsb", "amoeba"
- `use_mask`: Whether to use masks for registration

#### 3. N4 Bias Field Correction

Corrects intensity inhomogeneity in medical images.

Parameters:
- `num_fitting_levels`: Number of fitting levels for the bias field correction
- `num_iterations`: Number of iterations at each fitting level
- `convergence_threshold`: Convergence threshold for the correction

### Usage

```python
from habit.core.preprocessing.image_processor_pipeline import BatchProcessor

# Initialize the processor with your config file
processor = BatchProcessor(config_path="./config/config.yaml")

# Process all images
processor.process_batch()
```

### Output Structure

The processed images will be saved in the following structure:

```
out_dir/
├── processed_images/
│   ├── images/
│   │   └── subject_id/
│   │       ├── modality1/
│   │       │   └── modality1.nii.gz
│   │       └── modality2/
│   │           └── modality2.nii.gz
│   └── masks/
│       └── subject_id/
│           ├── modality1/
│           │   └── mask_modality1.nii.gz
│           └── modality2/
│               └── mask_modality2.nii.gz
└── logs/
    └── processing.log
```

### Notes

1. All images should be in NIfTI format (.nii.gz)
2. Masks should be named with the prefix "mask_" followed by the modality name
3. The pipeline will automatically handle both images and their corresponding masks
4. Processing logs are saved in the `logs` directory
5. The pipeline supports parallel processing for faster execution