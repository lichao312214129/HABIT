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

# Mermaid diagrams (developer architecture pages). Rendered client-side via
# mermaid.js, so it works on GitHub Pages without a local graphviz binary.
try:
    import sphinxcontrib.mermaid  # noqa: F401
    extensions.append('sphinxcontrib.mermaid')
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
# Windows CPU/GPU 便携包共用一个百度网盘分享目录（v5xa）。
# demo_data / config / tests 为独立分享；tests 分享目录内另含 config.rar（共 2 个文件，vv2c）。
#
# Docutils 不支持在 `` `text <|url_subst|>`_ `` 的 URL 位置展开 substitution；
# 必须在 epilog 里定义「整条链接」substitution（见 _build_rst_epilog）。
NETDISK_SHARES = {
    "demo_data": {
        "url": "https://pan.baidu.com/s/1K1m8U47wUWV9CCUNahNZuw?pwd=9ws9",
        "code": "9ws9",
        "link_label": "Download demo_data.rar",
    },
    "config_pack": {
        "url": "https://pan.baidu.com/s/1zk-UCZVMudMvbtC8M3xdgA?pwd=kbg5",
        "code": "kbg5",
        "link_label": "Download config.rar",
    },
    "tests_pack": {
        "url": "https://pan.baidu.com/s/1EAcC2s4qIKGp1h08UtbApA?pwd=vv2c",
        "code": "vv2c",
        "link_label": "Download tests bundle",
    },
    "cpu_pack": {
        "url": "https://pan.baidu.com/s/1iGYw1lSxKSri5oFrbAjzkg?pwd=v5xa",
        "code": "v5xa",
        "link_label": "Download CPU pack",
    },
    "gpu_pack": {
        "url": "https://pan.baidu.com/s/1iGYw1lSxKSri5oFrbAjzkg?pwd=v5xa",
        "code": "v5xa",
        "link_label": "Download GPU pack",
    },
    "torch_wheel": {
        "url": "https://pan.baidu.com/s/1eY4lmNegCYh5KgQB640FmA?pwd=nt7k",
        "code": "nt7k",
        "link_label": "Download wheel",
    },
}


def _rst_external_link_substitution(name: str, url: str, label: str) -> str:
    """Return an rst_epilog block that expands to an HTML external hyperlink."""
    return (
        f".. |{name}| raw:: html\n\n"
        f'   <a class="reference external" href="{url}">{label}</a>\n'
    )


def _build_rst_epilog() -> str:
    """Build Sphinx rst_epilog with working download-link substitutions."""
    lines: list[str] = []
    for key, share in NETDISK_SHARES.items():
        lines.append(
            _rst_external_link_substitution(
                f"download_{key}",
                share["url"],
                share["link_label"],
            )
        )
        lines.append(f".. |{key}_code| replace:: {share['code']}")
    lines.append(f".. |docs_base| replace:: {DOCS_BASE_URL}")
    lines.append(f".. |github_repo| replace:: {GITHUB_REPO_URL}")
    lines.append(
        _rst_external_link_substitution("link_github_repo", GITHUB_REPO_URL, "GitHub")
    )
    lines.append(
        _rst_external_link_substitution(
            "link_github_issues",
            github_issues_url(),
            "GitHub Issues",
        )
    )
    return "\n".join(lines) + "\n"


rst_epilog = _build_rst_epilog()
