# Sphinx 文档设计方案

## 概述

本文档描述如何使用 Sphinx 为 HABIT 项目生成完整的文档系统，包括 API 文档、用户指南、教程等。

## 目录结构设计

```
habit_project/
├── docs/                          # Sphinx 文档根目录
│   ├── conf.py                    # Sphinx 配置文件
│   ├── Makefile                   # 构建命令（Unix）
│   ├── make.bat                   # 构建命令（Windows）
│   ├── requirements.txt           # 文档依赖
│   │
│   ├── source/                    # 文档源文件
│   │   ├── index.rst             # 主入口
│   │   ├── getting_started/      # 快速开始
│   │   │   ├── index.rst
│   │   │   ├── installation.rst
│   │   │   └── quickstart.rst
│   │   │
│   │   ├── user_guide/           # 用户指南
│   │   │   ├── index.rst
│   │   │   ├── preprocessing.rst
│   │   │   ├── habitat_analysis.rst
│   │   │   ├── feature_extraction.rst
│   │   │   ├── machine_learning.rst
│   │   │   └── configuration.rst
│   │   │
│   │   ├── api/                  # API 文档（自动生成）
│   │   │   ├── index.rst
│   │   │   ├── habitat_analysis.rst
│   │   │   ├── machine_learning.rst
│   │   │   ├── preprocessing.rst
│   │   │   └── cli.rst
│   │   │
│   │   ├── tutorials/            # 教程
│   │   │   ├── index.rst
│   │   │   ├── basic_workflow.rst
│   │   │   └── advanced_usage.rst
│   │   │
│   │   ├── architecture/         # 架构文档
│   │   │   ├── index.rst
│   │   │   ├── overview.rst
│   │   │   └── design_decisions.rst
│   │   │
│   │   ├── examples/             # 示例
│   │   │   └── index.rst
│   │   │
│   │   └── _static/              # 静态资源
│   │       ├── css/
│   │       └── images/
│   │
│   ├── build/                    # 构建输出（gitignore）
│   │   ├── html/                 # HTML 输出
│   │   └── latex/                # LaTeX 输出（PDF）
│   │
│   └── _templates/               # 自定义模板
│       └── layout.html
│
└── [其他项目文件...]
```

## 配置文件设计

### conf.py 核心配置

```python
# conf.py
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 项目信息
project = 'HABIT'
copyright = '2025, HABIT Team'
author = 'HABIT Team'
release = '0.1.0'
version = '0.1.0'

# 扩展
extensions = [
    'sphinx.ext.autodoc',           # 自动生成 API 文档
    'sphinx.ext.autosummary',       # 自动摘要
    'sphinx.ext.viewcode',          # 源码链接
    'sphinx.ext.napoleon',          # Google/NumPy 风格文档字符串
    'sphinx.ext.intersphinx',       # 交叉引用
    'sphinx.ext.mathjax',           # 数学公式
    'sphinx.ext.githubpages',       # GitHub Pages 支持
    'myst_parser',                  # Markdown 支持
    'sphinx_rtd_theme',             # Read the Docs 主题
]

# 主题
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# 路径
templates_path = ['_templates']
html_static_path = ['_static']

# 源文件
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst',
}

# 排除模式
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# 语言
language = 'en'

# Autodoc 配置
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_mock_imports = ['antspyx', 'SimpleITK']  # 可选：模拟难以安装的依赖

# Napoleon 配置（Google/NumPy 风格）
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Intersphinx 映射
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# 输出选项
html_title = 'HABIT Documentation'
html_logo = '_static/logo.png'  # 如果有 logo
html_favicon = '_static/favicon.ico'
```

## 文档组织策略

### 1. 整合现有 Markdown 文档

**方案 A：使用 MyST Parser（推荐）**

- 安装：`pip install myst-parser`
- 在 `conf.py` 中启用 `myst_parser`
- 直接引用现有 Markdown 文件

**示例：在 RST 中引用 Markdown**

```rst
.. _habitat_analysis_guide:

Habitat Analysis Guide
=====================

.. include:: ../../doc_en/app_habitat_analysis.md
   :parser: myst_parser
```

**方案 B：转换为 RST**

- 使用 `pandoc` 批量转换
- 手动调整格式

### 2. API 文档自动生成

**创建 `source/api/index.rst`：**

```rst
API Reference
=============

.. toctree::
   :maxdepth: 2

   habitat_analysis
   machine_learning
   preprocessing
   cli
```

**创建 `source/api/habitat_analysis.rst`：**

```rst
Habitat Analysis Module
=======================

.. automodule:: habit.core.habitat_analysis
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: habit.core.habitat_analysis.habitat_analysis.HabitatAnalysis
   :members:
   :undoc-members:
   :show-inheritance:

Strategies
----------

.. automodule:: habit.core.habitat_analysis.strategies
   :members:

.. autoclass:: habit.core.habitat_analysis.strategies.base_strategy.BaseClusteringStrategy
   :members:
   :show-inheritance:

Managers
--------

.. automodule:: habit.core.habitat_analysis.managers
   :members:
```

### 3. 主入口 index.rst

```rst
Welcome to HABIT's documentation!
==================================

**HABIT (Habitat Analysis: Biomedical Imaging Toolkit)** is a comprehensive
Python toolkit for medical image analysis, focusing on tumor habitat clustering
and radiomics feature extraction.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/index
   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   user_guide/preprocessing
   user_guide/habitat_analysis
   user_guide/feature_extraction
   user_guide/machine_learning
   user_guide/configuration

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture/index
   architecture/overview

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```

## 构建和部署

### 本地构建

```bash
# 安装依赖
pip install sphinx sphinx-rtd-theme myst-parser

# 生成 HTML
cd docs
make html

# 或使用 sphinx-build
sphinx-build -b html source build/html
```

### GitHub Actions 自动构建

创建 `.github/workflows/docs.yml`：

```yaml
name: Build Documentation

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'
      
      - name: Install dependencies
        run: |
          pip install -r docs/requirements.txt
          pip install sphinx sphinx-rtd-theme myst-parser
      
      - name: Build documentation
        run: |
          cd docs
          make html
      
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        if: github.ref == 'refs/heads/main'
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs/build/html
```

### Read the Docs 部署

1. 在 Read the Docs 注册项目
2. 连接 GitHub 仓库
3. 设置构建配置：
   - Python 版本：3.8
   - 配置文件：`docs/conf.py`
   - 构建命令：`cd docs && make html`

## 多语言支持

### 方案 A：多目录结构（推荐）

```
docs/
├── source/
│   ├── en/          # 英文文档
│   │   ├── conf.py
│   │   └── index.rst
│   └── zh/          # 中文文档
│       ├── conf.py
│       └── index.rst
```

使用 `sphinx-intl` 进行国际化：

```bash
pip install sphinx-intl

# 提取可翻译文本
sphinx-build -b gettext source build/gettext

# 生成翻译文件
sphinx-intl update -p build/gettext -l zh_CN

# 构建中文文档
sphinx-build -b html -D language=zh_CN source build/html/zh
```

### 方案 B：整合现有 doc/ 和 doc_en/

在 RST 中条件包含：

```rst
.. ifconfig:: language == 'en'
   .. include:: ../../doc_en/app_habitat_analysis.md
      :parser: myst_parser

.. ifconfig:: language == 'zh'
   .. include:: ../../doc/app_habitat_analysis.md
      :parser: myst_parser
```

## 最佳实践

### 1. 代码文档字符串规范

使用 Google 风格或 NumPy 风格：

```python
def run_habitat(
    config_file: str,
    debug_mode: bool = False
) -> None:
    """
    Run habitat analysis pipeline.

    Args:
        config_file: Path to configuration YAML file
        debug_mode: Whether to enable debug mode

    Returns:
        None

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid

    Example:
        >>> run_habitat('config.yaml', debug_mode=True)
    """
    pass
```

### 2. 使用交叉引用

```rst
参考 :class:`~habit.core.habitat_analysis.HabitatAnalysis` 类。

更多信息见 :ref:`habitat_analysis_guide`。
```

### 3. 包含代码示例

```rst
.. code-block:: python
   :linenos:
   :emphasize-lines: 3,5

   from habit.core.habitat_analysis import HabitatAnalysis
   config = HabitatAnalysisConfig.from_file('config.yaml')
   analysis = HabitatAnalysis(config)
   results = analysis.run()
```

### 4. 使用 admonitions

```rst
.. note::
   这是重要提示。

.. warning::
   这是警告信息。

.. tip::
   这是使用技巧。
```

## 快速开始脚本

创建 `docs/setup_docs.sh`：

```bash
#!/bin/bash
# 设置 Sphinx 文档环境

# 创建目录
mkdir -p docs/source/{getting_started,user_guide,api,tutorials,architecture}
mkdir -p docs/{build,_static,_templates}

# 安装依赖
pip install sphinx sphinx-rtd-theme myst-parser sphinx-intl

# 初始化 Sphinx（如果还没有）
if [ ! -f docs/conf.py ]; then
    sphinx-quickstart docs
fi

echo "Documentation setup complete!"
echo "Run 'cd docs && make html' to build documentation."
```

## 总结

Sphinx 文档系统将提供：

1. ✅ **自动生成的 API 文档** - 从代码文档字符串生成
2. ✅ **整合现有 Markdown** - 使用 MyST Parser
3. ✅ **多语言支持** - 中英文文档
4. ✅ **美观的主题** - Read the Docs 主题
5. ✅ **搜索功能** - 内置全文搜索
6. ✅ **交叉引用** - 文档间链接
7. ✅ **多种输出格式** - HTML, PDF, ePub

这将大大提升项目的专业性和可用性！
