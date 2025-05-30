# HABIT - Habitat Analysis Tool for Medical Images

HABIT (Habitat Analysis Tool) 是一个专为医学影像设计的综合性肿瘤微环境分析工具，专注于基于影像组学特征和机器学习技术的肿瘤栖息地分析。

## 🚀 主要功能

### 核心分析模块
- **医学影像预处理** - 支持DICOM转换、重采样、配准、标准化等预处理步骤
- **栖息地分析** - 两阶段聚类分析：个体级supervoxel聚类和群体级habitat聚类  
- **影像组学特征提取** - 支持传统影像组学、纹理特征、小波特征等多种特征提取方法
- **机器学习建模** - 集成多种机器学习算法进行预测建模和模型比较
- **特征选择** - 包含多种特征选择方法如mRMR、统计学方法等
- **模型评估与可视化** - 全面的模型评估指标和可视化工具

### 支持的功能模块
- 🔬 **影像预处理管道** - 完整的医学影像预处理流程
- 🧮 **统计分析** - ICC分析、相关性分析等
- 📊 **数据可视化** - 丰富的图表和可视化工具
- 🤖 **AutoML支持** - 自动化机器学习流程
- 📈 **模型比较** - 多模型性能对比分析

## 📁 项目结构

```
habit_project/
├── habit/                      # 核心代码包
│   ├── core/                   # 核心功能模块
│   │   ├── habitat_analysis/   # 栖息地分析
│   │   ├── machine_learning/   # 机器学习
│   │   └── preprocessing/      # 影像预处理
│   └── utils/                  # 工具函数
├── scripts/                    # 应用脚本
├── doc/                        # 文档
├── config/                     # 配置文件
└── requirements.txt            # 依赖包列表
```

## 🛠️ 安装指南

详细安装步骤请参考 [INSTALL.md](INSTALL.md)

### 快速安装
```bash
# 克隆仓库
git clone <repository_url>
cd habit_project

# 创建虚拟环境
conda create -n habit python=3.8.16
conda activate habit

# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .
```

## 📖 使用指南

### 主要应用脚本

| 脚本名称 | 功能描述 | 文档链接 |
|---------|---------|----------|
| `app_getting_habitat_map.py` | 栖息地分析与映射 | [详细文档](doc/app_getting_habitat_map.md) |
| `app_image_preprocessing.py` | 医学影像预处理 | [详细文档](doc/app_image_preprocessing.md) |
| `app_traditional_radiomics_extractor.py` | 传统影像组学特征提取 | - |
| `app_extracting_habitat_features.py` | 栖息地特征提取 | [详细文档](doc/app_extracting_habitat_features.md) |
| `app_of_machine_learning.py` | 机器学习建模 | [详细文档](doc/app_of_machine_learning.md) |
| `app_model_comparison_plots.py` | 模型比较可视化 | [详细文档](doc/app_model_comparison_plots.md) |
| `app_icc_analysis.py` | ICC 一致性分析 | [详细文档](doc/app_icc_analysis.md) |

### 基本使用示例

#### 1. 栖息地分析
```bash
python scripts/app_getting_habitat_map.py --config config/habitat_config.yaml
```

#### 2. 影像预处理
```bash
python scripts/app_image_preprocessing.py --config config/preprocess_config.yaml
```

#### 3. 机器学习建模
```bash
python scripts/app_of_machine_learning.py --config config/ml_config.yaml
```

### Python API 使用示例

```python
from habit.core.habitat_analysis import HabitatAnalysis
from habit.core.machine_learning import Modeling

# 栖息地分析
habitat_analyzer = HabitatAnalysis(
    root_folder="path/to/data",
    out_folder="path/to/output",
    feature_config=config
)
habitat_analyzer.run()

# 机器学习建模
ml_model = Modeling(
    data_path="path/to/features.csv",
    target_column="target"
)
results = ml_model.train_and_evaluate()
```

## 📋 依赖要求

主要依赖包：
- Python 3.8.16
- SimpleITK 2.2.1
- antspyx 0.4.2
- scikit-learn
- pandas
- numpy
- matplotlib
- pyradiomics
- xgboost
- torch

完整依赖列表请查看 [requirements.txt](requirements.txt)

## 🔧 配置文件

项目使用YAML格式的配置文件来管理各种参数。配置文件示例可在 `habit/utils/example_paths_config.yaml` 中找到。

### 主要配置项：
- **数据路径配置** - 指定输入数据和输出目录
- **特征提取配置** - 设置特征提取方法和参数
- **聚类配置** - 配置supervoxel和habitat聚类参数
- **机器学习配置** - 设置模型类型、特征选择等参数

## 📚 文档

- [安装指南](INSTALL.md)
- [应用文档](doc/)
- [预处理模块说明](habit/core/preprocessing/README.md)

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 [MIT License](LICENSE) 许可证。

## 🙋‍♀️ 支持与反馈

如果您在使用过程中遇到问题或有改进建议，请：
1. 查看 [文档](doc/) 获取详细使用说明
2. 提交 [Issue](../../issues) 报告问题
3. 联系项目维护者

## 🔬 研究引用

如果您在研究中使用了 HABIT 工具，请考虑引用相关论文（待添加）。

---

**版本**: 0.1.0  
**更新日期**: 2024年
