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
            *   **一步法** (One-Step): 直接从体素聚类到生境，每个肿瘤自动确定最佳聚类数，生境标签独立
            *   **二步法** (Two-Step): 先个体聚类生成supervoxels，再群体聚类识别habitats，所有患者共享统一生境标签
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
| 🧬 **生境分析** | **一步法聚类** | 直接聚类到生境，每个肿瘤独立确定聚类数，生境标签不统一。 | [📖](doc/app_habitat_analysis.md) |
| | **二步法聚类** | 两阶段聚类（个体supervoxels → 群体habitats），统一生境标签体系。 | [📖](doc/app_habitat_analysis.md) |
| | **灵活的特征输入** | 支持多种体素级特征，包括原始信号强度、动态增强和影像组学特征。 | [📖](doc/app_habitat_analysis.md) |
| | **🎨 自动可视化** | 自动生成2D/3D聚类散点图、最优聚类数曲线等高质量可视化结果。 | [📖](#2-habitat-分析) |
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

---
# 安装指南

# HABIT 安装指南

本指南提供安装 HABIT 工具包及其所有依赖项的详细说明。

---

## 1. 系统要求

-   **操作系统**: Windows 10/11, Linux (Ubuntu 18.04+), 或 macOS 10.15+。
-   **Python 版本**: 推荐使用 3.8, 3.9, 或 3.10。
-   **内存 (RAM)**: 最低 16 GB，**强烈推荐 32 GB 或更多**，以便处理大型数据集。
-   **存储空间**: 至少 10 GB 可用磁盘空间。

## 2. 外部依赖

在安装 Python 包之前，您必须先安装以下外部工具：

### A. Conda

**强烈推荐**使用 `conda` (来自 Anaconda 或 Miniconda) 进行环境管理。
-   下载并安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 [Anaconda](https://www.anaconda.com/products/distribution)。

### B. Git

用于克隆本项目的代码仓库。
-   从 [Git 官网](https://git-scm.com/downloads) 安装 `git`。

### C. dcm2niix (DICOM 转换需要)

如果您计划将 DICOM 影像转换为 NIfTI 格式，则必须安装 `dcm2niix`。
1.  前往 [dcm2niix 的 GitHub 发布页面](https://github.com/rordenlab/dcm2niix/releases)。
2.  下载适用于您操作系统的预编译版本。
3.  解压可执行文件（在 Windows 上是 `dcm2niix.exe`），并将其所在位置添加到系统的 **PATH 环境变量**中。
4.  打开一个新的终端并运行 `dcm2niix --version` 来验证安装。

### D. R 语言 (可选)

**仅当您计划在机器学习流程中使用 `stepwise` (逐步回归) 特征选择方法时**，才需要安装 R。
1.  从 [R 项目官网](https://cran.r-project.org/)下载并安装 R。
2.  在安装过程中，请记下安装路径。
3.  您可能需要在机器学习的配置文件中指定此路径。

## 3. 安装步骤

推荐使用 Conda 进行安装。

### 第一步：克隆代码仓库

打开终端（或在 Windows 上打开 Anaconda Prompt）并运行：
```bash
git clone <repository_url>
cd habit_project
```

### 第二步：创建并激活 Conda 环境

为 HABIT 创建一个独立的环境以避免依赖冲突。
```bash
# 创建一个名为 'habit' 的环境，使用 Python 3.8
conda create -n habit python=3.8

# 激活新环境
conda activate habit
```

### 第三步：安装 Python 依赖

使用 `requirements.txt` 文件安装所有必需的 Python 包。
```bash
pip install -r requirements.txt
```

### 第四步：安装 HABIT 包

最后，以“可编辑”模式安装 HABIT 工具包。这使您可以在修改源代码后无需重新安装。
```bash
pip install -e .
```

## 4. 验证安装

为确保一切设置正确，请在终端中（已激活 `habit` 环境）运行以下检查。

1.  **检查基础包导入：**
    ```bash
    python -c "import habit; print(f'HABIT version {habit.__version__} installed successfully!')"
    ```

2.  **检查核心模块可用性：**
    ```bash
    python -c "from habit.utils.import_utils import check_dependencies; check_dependencies(['SimpleITK', 'antspyx', 'torch', 'sklearn', 'pyradiomics'])"
    ```
    此命令应报告所有列出的模块都可用。

3.  **检查脚本入口点：**
    ```bash
    python scripts/app_getting_habitat_map.py --help
    ```
    此命令应显示主分析脚本的帮助菜单。

## 5. 故障排除

-   **`antspyx` 或 `SimpleITK` 安装失败**：这些包有时可能存在编译问题。在运行 `pip install -r requirements.txt` 之前，尝试使用 `conda` 单独安装它们：
    ```bash
    conda install -c conda-forge antspyx simpleitk -y
    ```

-   **与 R 相关的 `stepwise` 选择错误**：如果您看到与 `rpy2` 或 R 相关的错误，请确保 R 已正确安装，并且如果需要，您的配置文件（例如 `config/config_machine_learning.yaml`）中的 `Rhome` 路径指向了正确的 R 安装目录。

-   **内存错误**：如果在分析过程中遇到 `MemoryError`，请尝试在您的 YAML 配置文件中减少 `processes` 的数量。

-   **CUDA/GPU 错误**：如果您有兼容的 NVIDIA GPU 并希望使用它，请确保已安装正确的 NVIDIA 驱动和 CUDA 工具包。然后，按照 [PyTorch 官网](https://pytorch.org/get-started/locally/)的说明安装支持 GPU 的 PyTorch 版本。

---

您的安装现已完成。请继续阅读 [**QUICKSTART.md**](QUICKSTART.md) 指南来运行您的第一次分析。

---
# 快速入门

# HABIT Quick Start Guide

Welcome to HABIT! This guide will walk you through running your first tumor habitat analysis in just a few minutes.

---

### **Step 0: Installation**

Before you begin, make sure you have successfully installed HABIT and its dependencies. If not, please follow the [**INSTALL.md**](INSTALL.md) guide.

Once installed, activate your environment:
```bash
conda activate habit
```

### **Step 1: Prepare Your Data**

HABIT expects a specific directory structure for your data. Create a main data folder and organize your patient images and masks as shown below.

**Required Structure:**
```
your_project_folder/
├── data/                  # Your main data directory
│   ├── images/            # Contains one subfolder per patient
│   │   ├── patient_001/
│   │   │   ├── pre_contrast/  # Subfolder for each modality
│   │   │   │   └── image.nii.gz (or a series of DICOM files)
│   │   │   ├── LAP/
│   │   │   │   └── image.nii.gz
│   │   │   └── PVP/
│   │   │       └── image.nii.gz
│   │   └── patient_002/
│   │       └── ...
│   └── masks/             # Structure is identical to /images
│       ├── patient_001/
│       │   ├── pre_contrast/
│       │   │   └── mask.nii.gz
│       │   ├── LAP/
│       │   │   └── mask.nii.gz
│       │   └── PVP/
│       │       └── mask.nii.gz
│       └── patient_002/
│           └── ...
└── output/                # An empty directory for results
```
- The folder names (`pre_contrast`, `LAP`, `PVP`) are **keys** that you will reference in the configuration file.
- The `masks` directory mirrors the `images` directory structure. Each image file should have a corresponding mask file.
- Supported formats include `.nii.gz`, `.nii`, `.nrrd`, and `.mha`.

#### Alternative: Using a File Manifest

If your data is not organized in the structure above, you can instead create a YAML file (like `config/image_files.yaml`) to explicitly define the path for each image. This is useful when your files are in different locations.

**Example `image_files.yaml`:**
```yaml
images:
  subj003:
    T1: F:\\path\\to\\subj003\\T1_folder
    T2: F:\\path\\to\\subj003\\T2_folder
  subj004:
    T1: /another/path/to/subj004/t1_folder
    T2: /another/path/to/subj004/t2_folder
```

In your main analysis configuration, you would then reference this manifest file instead of specifying a `data_dir`.

### **Step 2: Configure Your Analysis**

All analyses in HABIT are controlled by YAML configuration files. Let's copy and edit a sample configuration.

1.  **Copy the example config**:
    ```bash
    cp config/config_getting_habitat.yaml my_first_analysis.yaml
    ```

2.  **Edit `my_first_analysis.yaml`** with a text editor and change the following critical paths:
    ```yaml
    # 1. Set your data and output directories
    data_dir: /path/to/your_project_folder/data  # Point to your main data folder
    out_dir: /path/to/your_project_folder/output # Point to your output folder
    
    # 2. Define the image keys for feature extraction
    # These must match the filenames from Step 1
    FeatureConstruction:
      voxel_level:
        method: concat(raw(pre_contrast), raw(LAP), raw(PVP))
        # ... other params
    
    # 3. (Optional) Adjust clustering parameters
    HabitatsSegmention:
      supervoxel:
        n_clusters: 50  # Number of initial supervoxels per patient
      habitat:
        mode: training
        max_clusters: 8 # Maximum number of final habitats to find
    
    # 4. (Optional) Adjust number of parallel processes based on your CPU cores
    processes: 4
    ```

### **Step 3: Run the Habitat Analysis**

Now, you can run the main analysis script using the configuration file you just created.

```bash
python scripts/app_getting_habitat_map.py --config my_first_analysis.yaml
```

This process will perform voxel feature extraction, supervoxel clustering, and habitat clustering. You will see progress bars for each stage.

### **Step 4: Understand the Output**

Once the analysis is complete, check your `output` directory. You will find several new folders:

-   `supervoxel_maps/`: Contains the intermediate supervoxel segmentation for each patient.
-   `habitat_maps/`: Contains the final habitat segmentation for each patient.
-   `features/`: Stores the raw voxel-level and processed supervoxel-level features in `.csv` files.
-   `clustering_models/`: The trained clustering models are saved here.
-   `plots/`: Visualizations, such as elbow plots for determining the optimal number of clusters.

**The most important output is the set of images in `habitat_maps/`.** You can view these with a medical image viewer (e.g., ITK-SNAP, 3D Slicer) to see the identified tumor sub-regions.

### **Step 5: What's Next?**

With the habitats identified, you can now proceed to the next steps in the radiomics pipeline.

**1. Extract High-Level Habitat Features:**
Use the generated habitat maps to extract advanced features like MSI (spatial relationships) and ITH (heterogeneity score).
```bash
# First, configure `config/config_extract_features.yaml` with your paths
python scripts/app_extracting_habitat_features.py --config config/config_extract_features.yaml
```

**2. Train a Predictive Model:**
Use the extracted features to train a machine learning model.
```bash
# First, configure `config/config_machine_learning.yaml` with your feature files
python scripts/app_of_machine_learning.py --config config/config_machine_learning.yaml
```

---

🎉 **Congratulations!** You have successfully run your first analysis with HABIT. Explore the other scripts and configuration files to discover the full power of the toolkit.

---
# 命令行界面使用指南

# HABIT 命令行界面使用指南

> **HABIT (Habitat Analysis: Biomedical Imaging Toolkit)** 统一命令行界面文档

---

## 📖 目录

- [快速开始](#快速开始)
- [安装](#安装)
- [使用方法](#使用方法)
- [所有命令](#所有命令)
- [常见问题](#常见问题)
- [完整示例](#完整示例)

---

## ⚡ 快速开始

### 三步上手

```bash
# 1. 安装
pip install -e .

# 2. 测试
python -m habit --help

# 3. 使用
python -m habit preprocess -c config/config_image_preprocessing.yaml
```

---

## 🔧 安装

### 前提条件

- Python >= 3.8
- 已安装项目依赖

### 安装步骤

```bash
# 步骤 1: 清理缓存（如果之前安装过）
Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Recurse -Force

# 步骤 2: 安装依赖
pip install -r requirements.txt

# 步骤 3: 安装 HABIT 包
pip install -e .
```

### 验证安装

```bash
# 方法 1: 使用 habit 命令
habit --help

# 方法 2: 使用 Python 模块（推荐，更可靠）
python -m habit --help
```

**✅ 看到命令列表说明安装成功！**

---

## 🚀 使用方法

### 三种运行方式

#### 方式 1: `habit` 命令（需要正确安装）

```bash
habit --help
habit preprocess -c config.yaml
```

#### 方式 2: Python 模块（推荐⭐）

```bash
python -m habit --help
python -m habit preprocess -c config.yaml
```

#### 方式 3: 直接运行脚本（开发调试）

```bash
python habit/cli.py --help
python habit/cli.py preprocess -c config.yaml
```

### 基本语法

```bash
habit <命令> [选项]
```

**常用选项：**
- `--config, -c`: 指定配置文件（必需）
- `--help`: 显示帮助信息
- `--version`: 显示版本号

---

## 📋 所有命令

### 命令列表

| 命令 | 说明 | 对应原脚本 |
|------|------|-----------|
| `preprocess` | 图像预处理 | `app_image_preprocessing.py` |
| `habitat` | 生成 Habitat 地图 | `app_getting_habitat_map.py` |
| `extract-features` | 提取 Habitat 特征 | `app_extracting_habitat_features.py` |
| `ml` | 机器学习（训练/预测） | `app_of_machine_learning.py` |
| `kfold` | K折交叉验证 | `app_kfold_cv.py` |
| `compare` | 模型比较 | `app_model_comparison_plots.py` |
| `icc` | ICC 分析 | `app_icc_analysis.py` |
| `radiomics` | 传统影像组学 | `app_traditional_radiomics_extractor.py` |
| `test-retest` | Test-Retest 分析 | `app_habitat_test_retest_mapper.py` |

### 1. 图像预处理

```bash
python -m habit preprocess -c config/config_image_preprocessing.yaml
```

**功能**: 对医学图像进行重采样、配准、标准化处理

### 2. Habitat 分析

```bash
python -m habit habitat -c config/config_getting_habitat.yaml

# 启用调试模式
python -m habit habitat -c config/config_getting_habitat.yaml --debug
```

**功能**: 通过聚类分析生成 Habitat 地图

#### 🎨 自动生成的可视化结果

当 `plot_curves: true` 时（默认启用），会自动生成以下可视化文件：

**输出目录结构**:
```
<out_dir>/
├── visualizations/
│   ├── supervoxel_clustering/              # Supervoxel聚类散点图
│   │   ├── {subject}_supervoxel_clustering_2D.png  # 2D PCA散点图
│   │   ├── {subject}_supervoxel_clustering_3D.png  # 3D PCA散点图
│   │   └── ...
│   ├── optimal_clusters/                   # 最优聚类数曲线（one_step模式）
│   │   ├── {subject}_cluster_validation.png
│   │   └── ...
│   └── habitat_clustering/                 # Habitat聚类散点图（two_step模式）
│       ├── habitat_clustering_2D.png
│       └── habitat_clustering_3D.png
├── habitat_clustering_scores.png          # Habitat评分曲线（two_step模式）
└── {subject}_supervoxel.nrrd              # Supervoxel标签图像
```

**可视化特点**:
- ✅ 自动PCA降维（高维→2D/3D）
- ✅ 显示聚类中心（红色×标记）
- ✅ 坐标轴显示PCA解释方差百分比
- ✅ 高质量输出（DPI=300，适合发表）
- ✅ 使用viridis配色方案（色盲友好）

**配置示例 - One-step模式** (每个样本独立确定最优聚类数):

**方式1：自动找最优聚类数**（推荐用于样本差异大的情况）
```yaml
HabitatsSegmention:
  clustering_strategy: one_step
  
  supervoxel:
    algorithm: kmeans
  
  # One-step模式配置 - 为每个样本自动寻找最优聚类数
  one_step_settings:
    min_clusters: 2                   # 测试的最小聚类数
    max_clusters: 10                  # 测试的最大聚类数
    selection_method: silhouette      # 选择方法: silhouette, davies_bouldin, calinski_harabasz
    plot_validation_curves: true      # 为每个样本绘制验证曲线
  
  habitat:
    mode: training
```

**方式2：使用固定聚类数**（推荐用于样本相似的情况，类似two_step）
```yaml
HabitatsSegmention:
  clustering_strategy: one_step
  
  supervoxel:
    algorithm: kmeans
  
  # 使用固定的聚类数，跳过优化过程
  one_step_settings:
    best_n_clusters: 5  # 直接指定所有样本使用5个聚类，不进行自动优化
  
  habitat:
    mode: training
```

**配置示例 - Two-step模式** (群体层面统一habitat):
```yaml
HabitatsSegmention:
  clustering_strategy: two_step
  
  supervoxel:
    algorithm: kmeans
    n_clusters: 50  # 固定的supervoxel数量
  
  habitat:
    algorithm: kmeans
    min_clusters: 2
    max_clusters: 10
    habitat_cluster_selection_method: [silhouette, davies_bouldin]
    mode: training
```

**One-step vs Two-step 对比**:

| 特性 | One-step | Two-step |
|------|----------|----------|
| 聚类策略 | 每个样本独立聚类 | 先聚类到supervoxel，再聚类到habitat |
| 聚类数 | 每个样本自动确定 | 全局统一 |
| Habitat标签 | 不统一 | 统一（可跨样本比较） |
| 适用场景 | 样本差异大 | 样本相似，关注群体模式 |

### 3. 提取特征

```bash
python -m habit extract-features -c config/config_extract_features.yaml
```

**功能**: 从聚类后的图像提取 Habitat 特征

### 4. 机器学习

#### 训练模型

```bash
python -m habit ml -c config/config_machine_learning.yaml -m train
```

#### 预测（使用训练好的模型）

```bash
python -m habit \
  -c config/config_machine_learning.yaml \
  -m predict \
  --model ./ml_data/ml/rad/model_package.pkl \
  --data ./ml_data/breast_cancer_dataset.csv \
  -o ./ml_data/predictions/
```

**参数说明：**
- `-m, --mode`: 模式（`train` 或 `predict`）
- `--model`: 模型文件路径（.pkl）
- `--data`: 数据文件路径（.csv）
- `-o, --output`: 输出目录
- `--model-name`: 指定使用的模型名称
- `--evaluate/--no-evaluate`: 是否评估性能（默认评估）

### 5. K折交叉验证

```bash
python -m habit kfold -c config/config_machine_learning_kfold.yaml
```

**功能**: 对模型进行 K 折交叉验证

**输出文件**:
- `kfold_cv_results.json` - 详细的交叉验证结果
- `kfold_performance_summary.csv` - 性能摘要表
- `all_prediction_results.csv` - **兼容格式的预测结果**（可用于模型比较）
- `kfold_roc_curves.pdf` - ROC曲线（如果启用可视化）
- `kfold_calibration_curves.pdf` - 校准曲线（如果启用可视化）
- `kfold_dca_curves.pdf` - DCA决策曲线（如果启用可视化）
- `kfold_confusion_matrix_*.pdf` - 混淆矩阵（如果启用可视化）

**可视化配置**:
```yaml
is_visualize: true  # 在配置文件中启用可视化
```

**提示**: 
- K折验证完成后，会自动生成 `all_prediction_results.csv` 文件，该文件与标准 ml 命令的输出格式完全兼容
- 启用 `is_visualize` 后，会自动生成 ROC、DCA、校准曲线等可视化图表
- 可视化基于所有 fold 的聚合预测生成，全面反映模型性能

### 6. 模型比较

```bash
python -m habit compare -c config/config_model_comparison.yaml
```

**功能**: 生成多个模型的比较图表和统计数据

**兼容性**: 
- ✅ 支持标准 ml 命令的输出（`all_prediction_results.csv`）
- ✅ 支持 kfold 交叉验证的输出（`all_prediction_results.csv`）
- 可以比较来自不同训练方式的模型结果

### 7. ICC 分析

```bash
python -m habit icc -c config/config_icc_analysis.yaml
```

**功能**: 执行组内相关系数（ICC）分析

### 8. 传统影像组学

```bash
python -m habit radiomics -c config/config_traditional_radiomics.yaml
```

**功能**: 提取传统影像组学特征

### 9. Test-Retest 分析

```bash
python -m habit test-retest -c config/config_habitat_test_retest.yaml
```

**功能**: 执行 test-retest 重复性分析

---

## ❓ 常见问题

### Q1: `habit` 命令找不到

**原因**: 环境变量未配置或未正确安装

**解决方案**: 使用 Python 模块方式（更可靠）

```bash
python -m habit --help
```

### Q2: ImportError 或模块找不到

**原因**: Python 缓存问题或安装不完整

**解决方案**:

```bash
# 1. 清理缓存
Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Recurse -Force

# 2. 重新安装
pip uninstall HABIT -y
pip install -e .
```

### Q3: 配置文件路径错误

**解决方案**: 使用绝对路径或确保在正确的目录下

```bash
# 相对路径（推荐在项目根目录运行）
python -m habit preprocess -c ./config/config_image_preprocessing.yaml

# 绝对路径
python -m habit preprocess -c F:/work/research/.../config.yaml
```

### Q4: 原有脚本还能用吗？

**答**: 完全可以！新 CLI 不影响原有脚本。

```bash
# 旧方式仍然可用
python scripts/app_image_preprocessing.py --config config.yaml

# 新方式（更简洁）
python -m habit preprocess -c config.yaml
```

---

## 💡 完整示例

### 完整的 Habitat 分析工作流

```bash
# 步骤 1: 图像预处理
python -m habit preprocess -c config/config_image_preprocessing.yaml

# 步骤 2: 生成 Habitat 地图
python -m habit habitat -c config/config_getting_habitat.yaml

# 步骤 3: 提取 Habitat 特征
python -m habit extract-features -c config/config_extract_features.yaml

# 步骤 4: 训练机器学习模型（两种方式任选其一）

## 方式 A: 标准训练/测试集分割
python -m habit ml -c config/config_machine_learning.yaml -m train

## 方式 B: K折交叉验证（推荐用于小样本）
python -m habit kfold -c config/config_machine_learning_kfold.yaml

# 步骤 5: 模型比较（支持两种方式的结果）
python -m habit compare -c config/config_model_comparison.yaml

# 提示：compare 命令会自动读取 all_prediction_results.csv 文件
# 无论是来自 ml 命令还是 kfold 命令，格式完全兼容
```

### 使用训练好的模型预测新数据

```bash
python -m habit \
  -c config/config_machine_learning.yaml \
  -m predict \
  --model ./ml_data/ml/rad/model_package.pkl \
  --data ./ml_data/new_patient_data.csv \
  --output ./ml_data/predictions/ \
  --model-name XGBoost \
  --evaluate
```

---

## 📚 相关文档

### 中文文档
- **详细使用手册**: `doc/CLI_USAGE.md`
- **原功能文档**: `doc/app_*.md`

### 英文文档
- **Usage Manual**: `doc_en/CLI_USAGE.md`
- **Feature Docs**: `doc_en/app_*.md`

### 其他文档
- **主 README**: `README.md`
- **安装指南**: `INSTALL.md`
- **快速入门**: `QUICKSTART.md`

---

## 🎓 新旧方式对比

### 旧方式（脚本）

```bash
python scripts/app_image_preprocessing.py --config config.yaml
python scripts/app_getting_habitat_map.py --config config.yaml --debug
python scripts/app_of_machine_learning.py --config config.yaml --mode train
python scripts/app_kfold_cv.py --config config.yaml
```

### 新方式（CLI）

```bash
python -m habit preprocess -c config.yaml
python -m habit habitat -c config.yaml --debug
python -m habit ml -c config.yaml -m train
python -m habit kfold -c config.yaml
```

**优势**:
- ✅ 更简洁、更直观
- ✅ 统一的命令风格
- ✅ 自动生成帮助文档
- ✅ 参数验证和错误提示
- ✅ 支持短选项（`-c` 代替 `--config`）

---

## 💡 使用技巧

### 1. 使用短选项

```bash
# --config 可以简写为 -c
python -m habit preprocess -c config.yaml

# --mode 可以简写为 -m
python -m habit ml -c config.yaml -m train

# --output 可以简写为 -o
python -m habit ml -c config.yaml -m predict --model m.pkl --data d.csv -o ./out/
```

### 2. 查看命令帮助

```bash
# 每个命令都有详细帮助
python -m habit --help              # 所有命令列表
python -m habit preprocess --help   # 预处理命令帮助
python -m habit ml --help           # 机器学习命令帮助
```

### 3. 推荐的命令别名

如果你经常使用，可以在 PowerShell 配置文件中添加别名：

```powershell
# 编辑 PowerShell 配置文件
notepad $PROFILE

# 添加以下内容
function habit { python -m habit $args }

# 之后就可以直接使用
habit --help
habit preprocess -c config.yaml
```

---

## 📧 技术支持

- **邮箱**: lichao19870617@163.com
- **问题反馈**: 请详细描述问题和错误信息

---

## 📝 版本信息

- **版本**: 0.1.0
- **更新日期**: 2025-11-10
- **状态**: ✅ 稳定可用

### 最新更新 (2025-11-10)

#### 🎨 新增：聚类可视化功能
- ✅ 自动生成Supervoxel聚类2D/3D散点图（每个样本）
- ✅ 自动生成Habitat聚类2D/3D散点图（群体层面）
- ✅ 自动生成最优聚类数验证曲线（one_step模式）
- ✅ 使用PCA降维实现高维数据可视化
- ✅ 显示聚类中心和方差解释百分比
- ✅ 高质量输出（DPI=300，适合论文发表）
- ✅ 所有可视化自动保存到 `visualizations/` 目录

**使用方法**: 在配置文件中设置 `plot_curves: true`（默认已启用）即可自动生成所有可视化结果。详见 [Habitat分析](#2-habitat-分析) 章节。

---

**祝使用愉快！** 🎉