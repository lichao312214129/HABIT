# Sphinx 配置文件 - HABIT 项目文档
#

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def _load_project_urls_module() -> ModuleType:
    """Load project_urls.py without importing habit.utils (avoids runtime deps in CI)."""
    urls_path = project_root / "habit" / "utils" / "project_urls.py"
    spec = importlib.util.spec_from_file_location("habit_project_urls", urls_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load project URLs from {urls_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 项目信息
project = 'HABIT'
copyright = '2024, HABIT Team'
author = 'HABIT Team'
version = '1.0.0'
release = '1.0.0'

# Language
language = 'en'

# 源文件后缀 - 只使用 .rst
source_suffix = '.rst'

# 扩展 - 不使用 myst_parser
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.graphviz',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
]
try:
    import sphinx_copybutton  # noqa: F401
    extensions.append('sphinx_copybutton')
except ImportError:
    pass

# Napoleon 配置
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# 主题（与 PyRadiomics 相同使用 sphinx-rtd-theme，配合 custom.css 增强排版）
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'style_external_links': True,
}

# 模板路径
templates_path = ['_templates']

# 静态文件路径
html_static_path = ['_static']
html_css_files = ['custom.css']

html_show_sourcelink = True
html_show_sphinx = True
html_title = 'HABIT Documentation'
html_short_title = 'HABIT'

# 自动文档生成选项
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'ignore-module-all': True,
    'exclude-members': '__weakref__,__dict__,__pydantic_extra__,__pydantic_fields_set__,__pydantic_private__',
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

# 模拟缺失的模块，避免 autodoc 崩溃（勿 mock numpy/pandas/scipy/sklearn：Pydantic 与 autodoc 依赖其类型）
autodoc_mock_imports = [
    'SimpleITK',
    'antspy',
    'antspyx',
    'pyradiomics',
    'radiomics',
    'habitat_clustering',
    'habitat_clustering.clustering',
    'cv2',
    'trimesh',
    'openpyxl',
    'tqdm',
    'torch',
    'mrmr',
    'shap',
    'lifelines',
    'pydicom',
    'ants',
]

# Intersphinx 配置
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'pyradiomics': ('https://pyradiomics.readthedocs.io/en/latest/', None),
}

# 忽略的文件
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
]

# Pygments 高亮样式
pygments_style = 'sphinx'

# GitHub Pages configuration (single source: habit/utils/project_urls.py)
_project_urls = _load_project_urls_module()
DOCS_BASE_URL = _project_urls.DOCS_BASE_URL
GITHUB_REPO_SLUG = _project_urls.GITHUB_REPO_SLUG
GITHUB_REPO_URL = _project_urls.GITHUB_REPO_URL
github_issues_url = _project_urls.github_issues_url

github_pages = True
github_repo = GITHUB_REPO_SLUG
github_version = 'main'
html_baseurl = DOCS_BASE_URL

# Todo 配置
todo_include_todos = True

# 自定义变量（便携包配套资源网盘；维护者更新链接与提取码）
# config/tests 若与 demo_data 不同分享，请分别修改下方链接
rst_epilog = """
.. |demo_data_link| replace:: https://pan.baidu.com/s/1vDx6JZeM4Ay7VR1GAt7a-g?pwd=hkvq
.. |demo_data_code| replace:: hkvq
.. |config_pack_link| replace:: https://pan.baidu.com/s/1k1AVXRU6N0V8ggG1cZVtnQ?pwd=ziex
.. |config_pack_code| replace:: ziex
.. |tests_pack_link| replace:: https://pan.baidu.com/s/1cBw6WtLtOXNE7vpF8429NA
.. |tests_pack_code| replace:: xypk
.. |cpu_pack_link| replace:: https://pan.baidu.com/s/1dG4ibQONxvMOFZm1mOKpFw?pwd=ycva
.. |cpu_pack_code| replace:: ycva
.. |gpu_pack_link| replace:: https://pan.baidu.com/s/1bzh3DvNmiL4m-Wdw7K0Tcg?pwd=8wzx
.. |gpu_pack_code| replace:: 8wzx
.. |torch_wheel_link| replace:: https://pan.baidu.com/s/1eY4lmNegCYh5KgQB640FmA?pwd=nt7k
.. |torch_wheel_code| replace:: nt7k
.. |docs_base| replace:: """ + DOCS_BASE_URL + """
.. |github_repo| replace:: """ + GITHUB_REPO_URL + """
.. |github_issues| replace:: """ + github_issues_url() + """
"""
