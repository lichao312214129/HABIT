# Sphinx 配置文件 - HABIT 项目文档
#

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 项目信息
project = 'HABIT'
copyright = '2024, HABIT Team'
author = 'HABIT Team'
version = '1.0.0'
release = '1.0.0'

# 语言
language = 'zh_CN'

# 源文件后缀 - 只使用 .rst
source_suffix = '.rst'

# 扩展 - 不使用 myst_parser
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
]

# Napoleon 配置
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# 主题
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
}

# 模板路径
templates_path = ['_templates']

# 静态文件路径
html_static_path = ['_static']

# 自动文档生成选项
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'ignore-module-all': True,
}

# 模拟缺失的模块，避免 autodoc 崩溃
import sys
import importlib

class MockModule:
    """模拟缺失的模块"""
    def __init__(self, name):
        self.__name__ = name
        self.__all__ = []

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"module '{self.__name__}' has no attribute '{name}'")
        # 返回一个模拟的类
        class MockClass:
            pass
        return MockClass

# 检查并模拟缺失的模块
missing_modules = ['SimpleITK', 'shap', 'habitat_clustering']
for mod_name in missing_modules:
    if mod_name not in sys.modules:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            sys.modules[mod_name] = MockModule(mod_name)

# 抑制 autodoc 导入警告
autodoc_mock_imports = [
    'SimpleITK', 
    'antspy', 
    'antspyx', 
    'pyradiomics', 
    'radiomics',
    'habitat_clustering',
    'cv2',
    'trimesh',
    'openpyxl',
    'tqdm',
    'torch',
    'mrmr',
    'matplotlib',
    'matplotlib.pyplot',
    'scipy',
    'scipy.stats',
    'scipy.ndimage',
    'scipy.optimize',
    'pingouin',
    'statsmodels',
    'statsmodels.api',
    'seaborn',
    'shap',
    'lifelines',
    'pydicom',
    'sklearn',
    'pandas',
    'numpy'
]

# Intersphinx 配置
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# 忽略的文件
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Pygments 高亮样式
pygments_style = 'sphinx'

# GitHub Pages 配置
github_pages = True
github_repo = 'lichao312214129/HABIT'
github_version = 'main'

# Todo 配置
todo_include_todos = True
