# Habitat Analysis: Biomedical Imaging Toolkit (HABIT)

<p align="center">
  <img src="https://www.imagilys.com/wp-content/uploads/2018/09/radiomics-illustration-500x500.png" alt="Radiomics" width="200"/>
</p>

<p align="center">
  <strong>📖 语言 / Language</strong><br>
  <a href="README.md">🇨🇳 简体中文</a> | <a href="README_en.md">🇬🇧 English</a>
</p>

<p align="center">
    <a href="https://github.com/your-repo/habit_project/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
    <a href="#"><img src="https://img.shields.io/badge/Python-3.8+-brightgreen.svg"></a>
    <a href="#"><img src="https://img.shields.io/badge/Status-Active-green.svg"></a>
</p>

**HABIT (Habitat Analysis: Biomedical Imaging Toolkit)** 是一个专为医学影像设计的、基于Python的综合性肿瘤"生境"分析工具包。它提供从影像预处理到机器学习的端到端解决方案，使研究人员能够通过影像组学和高级分析方法深入探究肿瘤的异质性。

> 🎯 **新功能**: HABIT 现已支持统一的命令行界面（CLI）！安装后只需使用 `habit` 命令即可访问所有功能。详见[快速入门](#-快速入门)部分。

---

## 📖 核心工作流

HABIT的核心思想是识别和表征肿瘤内部具有不同影像表型的亚区，即“生境”。这一目标通过一个多阶段的流水线实现：

<p align="center">
  <b>影像 → 体素特征 → 超体素 → 生境 → 生境特征 → 预测模型</b>
</p>

1.  **体素级特征提取**: 为肿瘤内的每一个体素提取丰富的特征（如信号强度、纹理、动态增强特征等）。
2.  **超体素聚类**: 将空间上相邻且特征相似的体素分组，形成“超体素”。这一过分割步骤在简化图像的同时保留了局部边界信息。
3.  **生境聚类**: 在整个患者队列中对超体素进行聚类，以识别共通的、反复出现的模式，从而形成最终的“生境”。
4.  **特征工程**: 从这些生境中提取高阶特征，如它们的大小、形状、空间关系（MSI特征）和异质性（ITH分数）。
5.  **机器学习**: 使用工程化的生境特征来训练预测模型，用于如患者生存期、治疗反应或疾病诊断等临床终点的预测。

---

## 🔬 完整研究流程

一个典型的基于HABIT的影像组学研究项目包含以下步骤。HABIT工具包为其中标记为 `[HABIT]` 的步骤提供了强大支持。

1.  **数据采集与下载**:
    *   从医院PACS系统或公开数据集中获取原始影像数据（通常为DICOM格式）。
    *   *此步骤为项目前期准备，在HABIT工具包外部完成。*

2.  **数据整理与匿名化**:
    *   将数据按 `患者/序列/文件` 的结构进行整理。
    *   对患者隐私信息进行匿名化处理。
    *   `[HABIT]` `dcm2niix_converter` 模块支持在转换格式时进行匿名化。

3.  **格式转换 (DICOM to NIfTI)**:
    *   `[HABIT]` 使用 `dcm2niix_converter` 模块或 `app_image_preprocessing.py` 脚本将DICOM序列转换为NIfTI格式（`.nii.gz`）。

4.  **感兴趣区域 (ROI) 分割**:
    *   由放射科医生或研究人员使用ITK-SNAP, 3D Slicer等专业软件手动勾画肿瘤区域（ROI），并保存为mask文件（如 `mask.nii.gz`）。
    *   *此步骤通常在HABIT工具包外部完成，生成后续步骤所需的`mask`文件。*

5.  **影像预处理**:
    *   `[HABIT]` 使用 `app_image_preprocessing.py` 脚本进行一系列预处理，包括：
        *   **配准**: 将不同序列或模态的影像对齐到同一空间。
        *   **重采样**: 将所有影像统一到相同的体素间距。
        *   **强度标准化**: 如Z-Score标准化。
        *   **N4偏置场校正**: 校正MRI的信号不均匀性。

6.  **生境分析与特征提取**:
    *   `[HABIT]` 运行核心脚本 `app_getting_habitat_map.py` 来识别肿瘤生境。
        *   **支持两种聚类模式**：
            *   **一步法** (One-Step): 在个体水平对每个肿瘤单独聚类，自动确定最佳聚类数，适合个性化异质性分析
            *   **二步法** (Two-Step): 先生成supervoxels再进行群体聚类，识别跨患者的共通生境，适合队列研究
    *   `[HABIT]` 运行 `app_extracting_habitat_features.py` 从生境中提取高级特征（如MSI, ITH分数等）。

7.  **构建与评估预测模型**:
    *   `[HABIT]` 使用 `app_of_machine_learning.py` 进行特征选择、模型训练和内部验证。
    *   `[HABIT]` 使用 `app_model_comparison_plots.py` 对不同模型进行性能比较和可视化。

8.  **结果分析与论文撰写**:
    *   解释模型的发现，并撰写研究论文。
    *   *此步骤在HABIT工具包外部完成。*

## 🚀 主要功能

| 类别 | 功能 | 描述 | 文档 |
| :--- | :--- | :--- | :---: |
| 🖼️ **影像处理** | **预处理流水线** | 提供DICOM转换、重采样、配准和标准化的端到端工具。 | [📖](doc/app_image_preprocessing.md) |
| | **N4偏置场校正** | 校正MRI扫描中的信号强度不均匀性。 | [📖](doc/app_image_preprocessing.md) |
| | **直方图标准化** | 在不同患者或扫描仪之间标准化信号强度值。 | [📖](doc/app_image_preprocessing.md) |
| 🧬 **生境分析** | **一步法聚类** | 个体水平聚类，自动确定最佳聚类数，适合个性化异质性分析。 | [📖](doc/app_habitat_analysis.md) |
| | **二步法聚类** | 稳健的两阶段过程（超体素 → 生境），识别跨患者共通生境，适合队列研究。 | [📖](doc/app_habitat_analysis.md) |
| | **灵活的特征输入** | 支持多种体素级特征，包括原始信号强度、动态增强和影像组学特征。 | [📖](doc/app_habitat_analysis.md) |
| 🔬 **特征提取** | **高级特征集** | 提取传统影像组学、非影像组学统计、整体生境、独立生境、多区域空间交互（`msi`）和肿瘤内异质性（`ith_score`）等特征。 | [📖](doc/app_extracting_habitat_features.md) |
| | **可配置引擎** | 使用PyRadiomics和可定制的参数文件进行定制化特征提取。 | [📖](doc/app_extracting_habitat_features.md) |
| 🤖 **机器学习** | **完整工作流** | 包括数据分割、特征选择、模型训练和评估。 | [📖](doc/app_of_machine_learning.md) |
| | **丰富的算法支持** | 支持多种模型（逻辑回归、SVM、随机森林、XGBoost）和众多特征选择方法（ICC、VIF、mRMR、LASSO、RFE）。 | [📖](doc/app_of_machine_learning.md) |
| | **K折交叉验证** | 完善的K折交叉验证流程，支持多模型评估和可视化。 | [📖](doc/app_kfold_cross_validation.md) |
| | **模型比较** | 提供生成ROC曲线、决策曲线分析（DCA）和执行DeLong检验的工具。 | [📖](doc/app_model_comparison_plots.md) |
| 📊 **验证与工具** | **可复现性分析** | 包括测试-重测（Test-Retest）和组内相关系数（ICC）分析工具。 | [📖](doc/app_icc_analysis.md) |
| | **DICOM转换** | DICOM格式到NIfTI格式的转换工具。 | [📖](doc/app_dcm2nii.md) |
| | **模块化与可配置** | 所有步骤均通过易于编辑的YAML配置文件控制。 | [📖](HABIT_CLI.md) |

## 📁 项目结构

```
habit_project/
├── habit/                      # 核心Python源代码包
│   ├── core/                   # 主要分析模块
│   │   ├── habitat_analysis/   # 生境识别逻辑
│   │   ├── machine_learning/   # 机器学习建模与评估
│   │   └── preprocessing/      # 影像处理功能
│   └── utils/                  # 辅助工具（I/O、日志等）
├── scripts/                    # 用于运行分析的入口脚本
├── config/                     # 所有脚本的YAML配置文件
├── doc/                        # 每个模块的详细文档
├── requirements.txt            # Python依赖
├── INSTALL.md                  # 详细的安装指南
└── QUICKSTART.md               # 5分钟新用户入门教程
```

## 🛠️ 安装

详细指南请参见 [**INSTALL.md**](INSTALL.md)。

快速设置步骤：
```bash
# 1. 克隆仓库
git clone <repository_url>
cd habit_project

# 2. 创建并激活Conda环境
conda create -n habit python=3.8
conda activate habit

# 3. 安装依赖
pip install -r requirements.txt

# 4. 以可编辑模式安装HABIT包
pip install -e .
```

## 📖 快速入门

HABIT新手？请跟随我们的 [**QUICKSTART.md**](QUICKSTART.md) 指南，在几分钟内运行您的第一次生境分析！

### 🎯 统一命令行界面 (CLI) - **推荐使用方式**

**HABIT 提供了统一、简洁的命令行界面！** ✨ 

使用基于 **Click** 构建的 CLI 系统，您只需使用 `habit` 命令即可访问所有功能，无需记住复杂的脚本路径。

#### 安装后立即使用

完成 `pip install -e .` 后，`habit` 命令将在您的环境中全局可用：

```bash
# 查看所有可用命令
habit --help

# 查看特定命令的帮助信息
habit ml --help
habit kfold --help
```

#### 核心命令示例

```bash
# 1️⃣ 图像预处理 - 重采样、配准、标准化
habit preprocess --config config/config_image_preprocessing.yaml
# 📖 详细文档: doc/app_image_preprocessing.md

# 2️⃣ 生成 Habitat 地图 - 识别肿瘤亚区
# 支持一步法（个性化）或二步法（队列研究）
habit habitat --config config/config_getting_habitat.yaml
# 📖 详细文档: doc/app_habitat_analysis.md

# 3️⃣ 提取 Habitat 特征 - MSI, ITH等高级特征
habit extract-features --config config/config_extract_features.yaml
# 📖 详细文档: doc/app_extracting_habitat_features.md

# 4️⃣ 机器学习 - 训练预测模型
habit ml --config config/config_machine_learning.yaml --mode train
# 📖 详细文档: doc/app_of_machine_learning.md

# 5️⃣ 模型预测 - 使用训练好的模型
habit ml --mode predict \
  --model ./ml_data/model_package.pkl \
  --data ./new_data.csv \
  --output ./predictions/
# 📖 详细文档: doc/app_of_machine_learning.md

# 6️⃣ K折交叉验证 - 更稳健的模型评估
habit kfold --config config/config_machine_learning_kfold.yaml
# 📖 详细文档: doc/app_kfold_cross_validation.md

# 7️⃣ 模型比较 - ROC, DCA, 校准曲线等可视化
habit compare --config config/config_model_comparison.yaml
# 📖 详细文档: doc/app_model_comparison_plots.md

# 8️⃣ ICC分析 - 特征可重复性评估
habit icc --config config/config_icc_analysis.yaml
# 📖 详细文档: doc/app_icc_analysis.md

# 9️⃣ 传统影像组学特征提取
habit radiomics --config config/config_traditional_radiomics.yaml

# 🔟 测试-重测Habitat映射
habit test-retest --config config/config_habitat_test_retest.yaml
```

#### 快速参考表

| 命令 | 功能 | 配置文件 | 文档 |
|------|------|----------|:---:|
| `habit preprocess` | 图像预处理 | `config_image_preprocessing.yaml` | [📖](doc/app_image_preprocessing.md) |
| `habit habitat` | 生成Habitat地图 | `config_getting_habitat.yaml` | [📖](doc/app_habitat_analysis.md) |
| `habit extract-features` | 提取Habitat特征 | `config_extract_features.yaml` | [📖](doc/app_extracting_habitat_features.md) |
| `habit ml` | 机器学习训练/预测 | `config_machine_learning.yaml` | [📖](doc/app_of_machine_learning.md) |
| `habit kfold` | K折交叉验证 | `config_machine_learning_kfold.yaml` | [📖](doc/app_kfold_cross_validation.md) |
| `habit compare` | 模型比较与可视化 | `config_model_comparison.yaml` | [📖](doc/app_model_comparison_plots.md) |
| `habit icc` | ICC可重复性分析 | `config_icc_analysis.yaml` | [📖](doc/app_icc_analysis.md) |
| `habit radiomics` | 传统影像组学特征 | `config_traditional_radiomics.yaml` | [📖](HABIT_CLI.md) |
| `habit test-retest` | 测试-重测映射 | `config_habitat_test_retest.yaml` | [📖](doc/app_habitat_test_retest.md) |

#### 优势

✅ **简洁统一** - 所有功能通过 `habit` 命令访问  
✅ **即开即用** - 安装后无需配置路径  
✅ **帮助信息** - 每个命令都有 `--help` 选项  
✅ **彩色输出** - 清晰的成功/错误提示  
✅ **参数验证** - 自动检查必需参数  

📚 **完整 CLI 文档**: 请参阅 [**HABIT_CLI.md**](HABIT_CLI.md) 获取完整的命令行使用指南，包括安装说明、故障排除和高级用法。

---

### 传统脚本方式（兼容旧版）

> ⚠️ **注意**: 推荐使用上面的CLI命令。脚本方式仍然可用，但CLI提供了更好的用户体验。

如果您更喜欢直接运行Python脚本：

```bash
# 运行生境分析
python scripts/app_getting_habitat_map.py --config config/config_getting_habitat.yaml

# 提取生境特征
python scripts/app_extracting_habitat_features.py --config config/config_extract_features.yaml

# 训练机器学习模型
python scripts/app_of_machine_learning.py --config config/config_machine_learning.yaml
```

## 🤝 贡献

欢迎各种形式的贡献！请参考贡献指南（待添加）或开启一个Issue来讨论您的想法。

1.  Fork 本仓库。
2.  创建您的特性分支 (`git checkout -b feature/AmazingFeature`)。
3.  提交您的更改 (`git commit -m 'Add some AmazingFeature'`)。
4.  推送到分支 (`git push origin feature/AmazingFeature`)。
5.  开启一个 Pull Request。

## 📄 许可证

本项目采用 MIT 许可证 - 详情请见 [LICENSE](LICENSE) 文件。

## 🔬 引用

如果您在研究中使用了 HABIT，请考虑引用：
> [引用信息待添加]

## 🙋‍♀️ 支持

如果您遇到任何问题或有改进建议，请：
1.  阅读 `doc/` 文件夹中的详细文档。
2.  在 GitHub 上提交一个 [Issue](https://github.com/your-repo/habit_project/issues)。

### 📖 多语言文档

HABIT提供完整的中英文双语文档：
- **中文文档**: 位于 `doc/` 目录
- **English Documentation**: 位于 `doc_en/` 目录

💡 **语言切换**: 点击页面顶部的 "🇬🇧 English" 或 "🇨🇳 简体中文" 链接即可快速切换语言。
