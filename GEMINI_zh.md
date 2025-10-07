# Gemini 项目伴侣：HABIT

本文档提供了 HABIT 项目的摘要，以帮助 Gemini（AI 助手）理解代码库并协助开发任务。

## 1. 项目摘要

**HABIT (Habitat Analysis: Biomedical Imaging Toolkit)** 是一个基于 Python 的端到端肿瘤栖息地分析工具包。它使研究人员能够通过影像组学和机器学习来研究肿瘤的异质性。

核心工作流程是：
**影像 → 体素特征 → 超体素 → 栖息地 → 栖息地特征 → 预测模型**

- **技术栈**: Python 3.8+
- **核心库**: PyRadiomics, SimpleITK, scikit-learn, pandas, numpy。
- **配置**: 由位于 `config/` 目录中的 YAML 文件驱动。
- **执行**: 通过 `scripts/` 目录中的 Python 脚本执行。

## 2. 项目结构

- `habit/`: 核心 Python 源代码包。
    - `core/`: 用于分析的主要模块。
        - `habitat_analysis/`: 栖息地识别逻辑。
        - `machine_learning/`: 机器学习建模与评估。
        - `preprocessing/`: 图像处理功能。
    - `utils/`: 辅助工具（I/O、日志记录等）。
- `scripts/`: 用于运行分析的入口脚本。这些是用户的主要可执行文件。
- `config/`: 所有脚本的 YAML 配置文件。
- `doc/` & `doc_en/`: 每个模块的详细文档。
- `requirements.txt`: Python 依赖项。
- `INSTALL.md`: 详细的安装指南。
- `QUICKSTART.md`: 为新用户准备的 5 分钟教程。

## 3. 关键文件和脚本

- **配置文件 (`config/`)**:
    - `config_getting_habitat.yaml`: 栖息地分析的主要配置。
    - `config_extract_features.yaml`: 提取高级别栖息地特征的配置。
    - `config_machine_learning.yaml`: 训练和评估机器学习模型的配置。
    - `image_files.yaml`: 替代基于目录的数据结构，允许指定文件路径。

- **主要脚本 (`scripts/`)**:
    - `app_getting_habitat_map.py`: 识别肿瘤栖息地的核心脚本。
    - `app_extracting_habitat_features.py`: 从栖息地中提取特征（MSI, ITH）。
    - `app_of_machine_learning.py`: 训练和评估预测模型。
    - `app_image_preprocessing.py`: 图像预处理（转换、配准、归一化）的流程。

## 4. 常见工作流程

所有工作流程都通过从 `scripts/` 目录运行一个脚本并附带相应的配置文件来启动。

**1. 运行栖息地分析:**
```bash
python scripts/app_getting_habitat_map.py --config config/config_getting_habitat.yaml
```

**2. 提取栖息地特征:**
```bash
python scripts/app_extracting_habitat_features.py --config config/config_extract_features.yaml
```

**3. 训练机器学习模型:**
```bash
python scripts/app_of_machine_learning.py --config config/config_machine_learning.yaml
```

## 5. 给 Gemini 的说明

- **代码风格**: 遵循 PEP 8 for Python。遵守项目中现有的代码风格。
- **配置**: 当被要求执行任务时，首先检查 `config/` 目录中相关的 YAML 配置文件。
- **执行**: 使用 `scripts/` 目录中的脚本来运行分析。除非特定任务需要，否则不要直接调用 `habit/` 包中的函数。
- **文件路径**: 修改配置文件时，请注意文件路径。用户可能在 Windows 或 Linux 上。为保持兼容性，请使用正斜杠（/）表示路径，或使用 `os.path.join`。
- **依赖项**: 如果需要新的依赖项，请将其添加到 `requirements.txt`。
- **代码检查/格式化**: 项目使用 `.pre-commit-config.yaml`。在提交之前，最好运行 `pre-commit run --all-files`。
- **文档**: 如果添加了新功能，请更新 `doc/` 和 `doc_en/` 目录中相应的文档。

## 6. Gemini 能力 (IDE 连接时)

当连接到 IDE 时，Gemini 作为一个实践编程助手，具备以下能力：

- **完整的项目访问**: Gemini 可以读取和理解整个项目结构和文件内容，提供具有上下文感知能力的帮助。
- **直接操作**: Gemini 可以直接执行以下操作：
    - **编写和修改代码**。
    - **创建、删除和管理文件**。
    - **执行 shell 命令**（例如，运行脚本、安装依赖、运行测试、使用 git）。
- **无缝工作流**: Gemini 集成在开发环境中，实现了流畅高效的工作流程，无需切换上下文。
- **验证**: Gemini 可以运行测试和代码检查工具来验证自己的工作，确保代码质量和正确性。

本质上，当连接到 IDE 时，Gemini 从一个“知识顾问”转变为一个可以积极参与开发生命周期的“开发助手”。