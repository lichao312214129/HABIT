# HABIT 项目文档结构

## 📚 文档整理说明

所有 CLI 相关文档已经整合，现在结构更清晰、更易用。

---

## 📁 当前文档结构

### 🎯 主要文档（项目根目录）

| 文档 | 说明 | 用途 |
|------|------|------|
| **HABIT_CLI.md** | ⭐ **统一 CLI 文档** | CLI 使用的完整指南（推荐从这里开始） |
| README.md | 项目主页（中文） | 项目概述、功能介绍 |
| README_en.md | 项目主页（英文） | 项目概述、功能介绍 |
| INSTALL.md | 安装指南（中文） | 详细的环境配置和安装步骤 |
| INSTALL_en.md | 安装指南（英文） | 详细的环境配置和安装步骤 |
| QUICKSTART.md | 快速入门 | 5分钟快速上手教程 |
| setup.py | 安装配置 | Python 包安装脚本 |
| pyproject.toml | 项目配置 | Poetry 项目配置 |
| requirements.txt | 依赖列表 | Python 依赖包列表 |

### 📖 功能文档（doc/ 目录）

**中文文档** (`doc/`)：
- `CLI_USAGE.md` - CLI 快速参考（指向主文档）
- `app_*.md` - 各功能模块的详细文档
- `FONT_CONFIGURATION.md` - 字体配置说明
- `import_robustness_guide.md` - 导入健壮性指南

**英文文档** (`doc_en/`)：
- `CLI_USAGE.md` - CLI quick reference (points to main doc)
- `app_*.md` - Detailed documentation for each module
- `FONT_CONFIGURATION.md` - Font configuration guide
- `import_robustness_guide.md` - Import robustness guide

### 🔧 配置文件（config/ 目录）

- `config_image_preprocessing.yaml` - 图像预处理配置
- `config_getting_habitat.yaml` - Habitat 分析配置
- `config_extract_features.yaml` - 特征提取配置
- `config_machine_learning.yaml` - 机器学习配置
- `config_machine_learning_kfold.yaml` - K折验证配置
- `config_model_comparison.yaml` - 模型比较配置
- `config_icc_analysis.yaml` - ICC 分析配置
- `config_traditional_radiomics.yaml` - 传统影像组学配置
- 其他参数文件...

### 💻 源代码（habit/ 目录）

```
habit/
├── __init__.py              # 包初始化
├── __main__.py              # 支持 python -m habit
├── cli.py                   # 主 CLI 入口
├── cli_commands/            # CLI 命令模块
│   └── commands/            # 具体命令实现
├── core/                    # 核心功能
│   ├── habitat_analysis/    # Habitat 分析
│   ├── machine_learning/    # 机器学习
│   └── preprocessing/       # 图像预处理
└── utils/                   # 工具函数
```

### 📜 应用脚本（scripts/ 目录）

传统的 Python 脚本（仍然可用）：
- `app_image_preprocessing.py`
- `app_getting_habitat_map.py`
- `app_extracting_habitat_features.py`
- `app_of_machine_learning.py`
- `app_kfold_cv.py`
- `app_model_comparison_plots.py`
- 等等...

---

## 🗺️ 文档导航

### 新用户入门路径

```
1. README.md → 了解项目
2. INSTALL.md → 安装环境
3. HABIT_CLI.md → 学习 CLI 使用
4. QUICKSTART.md → 快速实践
```

### 命令行使用路径

```
HABIT_CLI.md → 完整的 CLI 使用指南
```

### 深入学习路径

```
doc/app_*.md → 各模块详细文档
```

---

## ✅ 已删除的冗余文档

为了简化文档结构，以下临时/冗余文档已删除：

- ❌ `CLI_FILES_CREATED.md` - 临时文件清单
- ❌ `CLI_FIX_CIRCULAR_IMPORT.md` - 技术问题修复记录
- ❌ `CLI_IMPLEMENTATION_SUMMARY.md` - 实现细节总结
- ❌ `CLI_INSTALL.md` - 单独的安装文档（已合并）
- ❌ `CLI_QUICKSTART.md` - 单独的快速开始（已合并）
- ❌ `CLI_FINAL_SOLUTION.md` - 最终方案文档（已合并）
- ❌ `README_CLI.md` - 旧的 CLI README（已替换）

**所有重要内容都已整合到** `HABIT_CLI.md`

---

## 📋 快速参考

### 我想...

#### 🔍 了解 HABIT 是什么
→ 阅读 `README.md`

#### 🛠️ 安装 HABIT
→ 阅读 `INSTALL.md`

#### ⚡ 快速上手
→ 阅读 `QUICKSTART.md` 或 `HABIT_CLI.md`

#### 💻 使用命令行界面
→ 阅读 `HABIT_CLI.md` （⭐ 推荐）

#### 📖 深入学习某个功能
→ 阅读 `doc/app_*.md` 对应的文档

#### 🔧 配置参数
→ 查看 `config/` 目录下的 YAML 文件

#### 👨‍💻 修改源代码
→ 查看 `habit/` 目录下的 Python 代码

---

## 🎯 推荐阅读顺序

### 对于新用户

1. **README.md** - 了解项目概况
2. **INSTALL.md** - 完成环境配置
3. **HABIT_CLI.md** - 学习命令行使用
4. 实际操作，运行示例命令

### 对于熟悉旧版的用户

1. **HABIT_CLI.md** - 快速了解新的 CLI 方式
2. 对比新旧命令格式
3. 开始使用新的 CLI

### 对于开发者

1. **README.md** - 项目概述
2. **HABIT_CLI.md** - CLI 架构
3. `habit/cli.py` 和 `habit/cli_commands/` - 源代码
4. `doc/app_*.md` - 功能模块文档

---

## 💡 使用建议

### ✅ 推荐

- **使用 `HABIT_CLI.md`** 作为 CLI 的主要参考
- **使用 `python -m habit`** 而不是直接 `habit` 命令（更可靠）
- **配置文件放在 `config/` 目录**
- **查看命令帮助**: `python -m habit <命令> --help`

### ⚠️ 注意

- 原有的 `scripts/app_*.py` 脚本仍然可用
- CLI 和脚本两种方式可以并存
- 配置文件格式完全相同

---

## 📧 反馈与建议

如果您觉得文档还需要改进，请联系：

**Email**: lichao19870617@163.com

---

## 📝 更新日志

**2025-10-19**: 
- ✅ 整合所有 CLI 相关文档到 `HABIT_CLI.md`
- ✅ 删除 7 个冗余文档
- ✅ 简化 `doc/CLI_USAGE.md` 和 `doc_en/CLI_USAGE.md`
- ✅ 创建本文档结构说明

---

**文档版本**: 1.0  
**最后更新**: 2025-10-19

