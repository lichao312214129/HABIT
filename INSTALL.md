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