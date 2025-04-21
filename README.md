# HABIT - Habitat Analysis Tool for Medical Images

HABIT是一个用于医学图像生境分析的综合工具包，提供了放射组学特征提取、生境聚类分析、机器学习模型构建等功能。本工具包采用模块化设计，遵循Python最佳实践规范。

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

1. **生境分析**
   - 影像放射组学特征提取
   - 基于聚类的生境识别
   - 生境特征统计与可视化

2. **机器学习分析**
   - 特征选择与预处理
   - 模型训练与评估
   - 预测与结果分析

3. **工具函数**
   - 配置管理
   - 结果保存与读取
   - 可视化工具

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

### 特征提取配置

```yaml
# 示例配置文件
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
n_habitats: 0  # 0表示自动检测
mode: both     # extract, parse, both
debug: false
```

### 机器学习配置

```yaml
# 示例配置文件
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

## 贡献指南

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开 Pull Request

## 许可证

此项目基于 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件

## 联系方式

如有任何问题，请联系项目维护者 example@example.com 