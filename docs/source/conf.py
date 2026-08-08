# Sphinx configuration file for HABIT documentation.
#

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

# Add the project root to the import path.
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


def _load_package_version() -> str:
    """Read the package version without importing HABIT runtime dependencies."""
    version_scope: dict[str, object] = {}
    version_file = project_root / "habit" / "_version.py"
    exec(
        compile(version_file.read_text(encoding="utf-8"), str(version_file), "exec"),
        version_scope,
    )
    return str(version_scope["__version__"])


# Project information.
project = "HABIT"
copyright = "2024, HABIT Team"
author = "HABIT Team"
version = _load_package_version()
release = version

# Language
language = "en"

# Source file suffix: use .rst only.
source_suffix = ".rst"

# Extensions; myst_parser is intentionally not used.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.graphviz",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
]
try:
    import sphinx_copybutton  # noqa: F401

    extensions.append("sphinx_copybutton")
except ImportError:
    pass

# Mermaid diagrams (developer architecture pages). Rendered client-side via
# mermaid.js — works on GitHub Pages without a local graphviz binary.
# Required: pip install sphinxcontrib-mermaid (see docs/requirements.txt).
extensions.append("sphinxcontrib.mermaid")

# Client-side rendering for static hosting (GitHub Pages).
mermaid_output_format = "raw"

# Napoleon configuration.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# Use the Read the Docs theme with the project custom stylesheet.
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
    "style_external_links": True,
}

# Template path.
templates_path = ["_templates"]

# Static asset paths.
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_show_sourcelink = True
html_show_sphinx = True
html_title = "HABIT Documentation"
html_short_title = "HABIT"

# Automatic API documentation options.
# NOTE: ``undoc-members`` is deliberately off. Besides hiding members that
# still need docstrings (which keeps the reference honest), autodoc documents
# dataclass fields twice when it is combined with ``members`` (once from the
# annotation scan and once from the class-docstring Attributes section),
# producing "duplicate object description" warnings on every contract page.
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "ignore-module-all": True,
    "exclude-members": "__weakref__,__dict__,__pydantic_extra__,__pydantic_fields_set__,__pydantic_private__,model_config,model_fields",
}

# Autosummary: generate one stub page per public symbol (scikit-learn style
# API reference). Stubs land in api/generated/ and are gitignored.
autosummary_generate = True
autosummary_imported_members = False
autosummary_filename_map = {}

# Render classes under their short name (``Subject`` not
# ``habit.contracts.subject.Subject``) in page titles; the canonical dotted
# path stays visible in the autodoc directive itself.
add_module_names = False

# Mock missing modules so autodoc can complete.
import sys
import importlib


class MockModule:
    """Represent a missing optional module during documentation builds."""

    def __init__(self, name):
        self.__name__ = name
        self.__all__ = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(f"module '{self.__name__}' has no attribute '{name}'")

        # Return a placeholder class for the requested attribute.
        class MockClass:
            pass

        return MockClass


# Check optional modules and install placeholders when necessary.
missing_modules = ["SimpleITK", "shap", "habitat_clustering"]
for mod_name in missing_modules:
    if mod_name not in sys.modules:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            sys.modules[mod_name] = MockModule(mod_name)

# Mock optional modules for autodoc. Do not mock numpy, pandas, scipy, or
# sklearn because Pydantic and autodoc require their type information.
autodoc_mock_imports = [
    "SimpleITK",
    "antspy",
    "antspyx",
    "pyradiomics",
    "radiomics",
    "habitat_clustering",
    "habitat_clustering.clustering",
    "cv2",
    "trimesh",
    "openpyxl",
    "tqdm",
    "torch",
    "mrmr",
    "shap",
    "lifelines",
    "pydicom",
    "ants",
]

# Intersphinx configuration.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pyradiomics": ("https://pyradiomics.readthedocs.io/en/latest/", None),
}

# Files ignored by Sphinx.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# Pygments highlighting style.
pygments_style = "sphinx"

# GitHub Pages configuration (single source: habit/utils/project_urls.py)
_project_urls = _load_project_urls_module()
DOCS_BASE_URL = _project_urls.DOCS_BASE_URL
GITHUB_REPO_SLUG = _project_urls.GITHUB_REPO_SLUG
GITHUB_REPO_URL = _project_urls.GITHUB_REPO_URL
github_issues_url = _project_urls.github_issues_url

github_pages = True
github_repo = GITHUB_REPO_SLUG
github_version = "main"
html_baseurl = DOCS_BASE_URL

# Todo configuration.
todo_include_todos = True

# Custom download variables for project resource bundles. Maintainers update
# these links and extraction codes.
# demo_data, config, and tests are separate bundles; the tests bundle also
# contains config.rar. The Windows lightweight installer ZIP is distributed
# separately from these netdisk demo/config/test shares.
#
# Docutils does not expand substitutions inside the URL of
# `` `text <|url_subst|>`_``. Define complete-link substitutions in the epilog
# (see _build_rst_epilog).
NETDISK_SHARES = {
    "demo_data": {
        "url": "https://pan.baidu.com/s/1w8r0IUJ8YXVDrkFYCAOQWw?pwd=9bi3",
        "code": "9bi3",
        "link_label": "Download preprocessed.zip",
    },
    "ml_data": {
        "url": "https://pan.baidu.com/s/1qOmZJ3uDgkDKHpHGVRpcEA?pwd=atnp",
        "code": "atnp",
        "link_label": "Download ml_data.zip",
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
