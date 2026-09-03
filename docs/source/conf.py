# Sphinx configuration file for HABIT documentation.
#

import importlib.util
import os
import re
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
# Copy button on every code block (nilearn / sphinx-copybutton).
extensions.append("sphinx_copybutton")
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Executed Guide / Quickstart examples (nilearn-style sphinx-gallery).
extensions.append("sphinx_gallery.gen_gallery")


def _patch_sphinx_gallery_replace_md5() -> None:
    """Skip missing ``*.examples.new`` on Windows case-insensitive filesystems.

    sphinx-gallery finalises backreferences by renaming ``name.examples.new``
    to ``name.examples``. Two identified names that differ only by case, or a
    second pass after a successful rename, raise ``FileNotFoundError``
    (WinError 2) and abort the build. Linux CI never hits this.
    """
    import sphinx_gallery.utils as sg_utils

    original = sg_utils._replace_md5

    def _safe_replace_md5(fname_new, fname_old=None, **kwargs):
        fname_new = str(fname_new)
        if not os.path.isfile(fname_new):
            return None
        try:
            return original(fname_new, fname_old, **kwargs)
        except FileNotFoundError:
            return None

    sg_utils._replace_md5 = _safe_replace_md5


_patch_sphinx_gallery_replace_md5()

# Mermaid diagrams (developer architecture pages). Rendered client-side via
# mermaid.js — works on GitHub Pages without a local graphviz binary.
# Required: pip install sphinxcontrib-mermaid (see docs/requirements.txt).
extensions.append("sphinxcontrib.mermaid")

# Headless matplotlib + skip napari windows while the gallery executes.
os.environ.setdefault("HABIT_NO_VIEW", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

sphinx_gallery_conf = {
    "examples_dirs": [
        str(project_root / "examples" / "guide"),
        str(project_root / "examples" / "quickstart"),
    ],
    "gallery_dirs": ["auto_examples", "auto_quickstart"],
    "filename_pattern": r"plot_",
    "ignore_pattern": r"__init__",
    "abort_on_example_error": True,
    "doc_module": ("habit",),
    "reference_url": {"habit": None},
    "capture_repr": ("_repr_html_", "__repr__"),
    "remove_config_comments": True,
    "nested_sections": True,
    "within_subsection_order": "FileNameSortKey",
    # Same directory as autosummary stubs so generated pages can
    # ``.. include:: {{fullname}}.examples`` (nilearn pattern).
    "backreferences_dir": str(Path("api") / "generated"),
    "download_all_examples": False,
    "plot_gallery": True,
    "reset_modules": ("matplotlib",),
    "min_reported_time": 1,
}

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
    # sphinx-rtd-theme no longer accepts display_version
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
autosummary_generate_overwrite = True
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
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pyradiomics": ("https://pyradiomics.readthedocs.io/en/latest/", None),
}

# Files ignored by Sphinx.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Generated at build time and pulled in via ``.. include::`` only.
    "api/_generated_plugin_catalog.rst",
    "how_to/_generated_catalog_*.rst",
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
# contains config.rar.
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


#: Habitat chooser page includes one generated catalog per scientific domain.
_HABITAT_CHOOSER_DOMAINS: tuple[str, ...] = (
    "voxel_feature_extractor",
    "feature_preprocessing_method",
    "supervoxelizer",
    "supervoxel_feature_extractor",
    "pooling",
    "habitat_model_fitter",
    "habitat_assigner",
    "habitat_feature_extractor",
)


def _write_plugin_catalog_include() -> None:
    """Regenerate the Spec/create catalog from live plugin metadata.

    ``format_plugin_catalog_rst`` prefers a plugin ``params_model`` when one
    exists; built-ins have none, so rows come from the constructor signature
    (``_signature_param_infos``). Do not change that generator here.
    """
    from habit.api.plugins import format_plugin_catalog_rst

    dest = Path(__file__).parent / "api" / "_generated_plugin_catalog.rst"
    dest.write_text(format_plugin_catalog_rst(), encoding="utf-8")
    how_to = Path(__file__).parent / "how_to"
    for domain in _HABITAT_CHOOSER_DOMAINS:
        (how_to / f"_generated_catalog_{domain}.rst").write_text(
            format_plugin_catalog_rst(domain),
            encoding="utf-8",
        )


_write_plugin_catalog_include()


# sphinx-gallery nested_sections=True writes one hidden toctree of subsection
# index files at the *end* of auto_examples/index.rst, after the last inlined
# heading ("6. Scaled Execution"). Sphinx attaches a toctree to the nearest
# preceding section, so sphinx_rtd_theme nests the whole Habitat Guide under
# Parallel runs. Lift that toctree to the gallery root heading instead.
_GALLERY_NESTED_TOCTREE_RE = re.compile(
    r"\n\.\. toctree::\n"
    r"   :hidden:\n"
    r"   :includehidden:\n"
    r"\n+"
    r"(?:   [^\n]*auto_examples/[^\n]+/index(?:\.rst)?\n)+"
    r"\n*",
)


def _is_rst_h1_rule(line: str) -> bool:
    """Return True if *line* is an RST ``====`` heading underline."""
    stripped = line.strip()
    return bool(stripped) and set(stripped) == {"="} and len(stripped) >= 3


_BROKEN_H1_RE = re.compile(
    r"(?m)^(?P<title>\S[^\n]*?)\n\n+(?P<rule>=+)\s*$"
)


def _repair_broken_rst_h1(text: str) -> str:
    """Join a title and ``====`` underline that gallery split with a blank line.

    sphinx-gallery sometimes writes ``Title\\n\\n=======``, which is not a
    heading. Sphinx then omits that subsection index from the sidebar, so
    its pages appear as leaves under the previous numbered section (no ``+``).
    """

    def _join(match: re.Match[str]) -> str:
        title = match.group("title")
        rule = match.group("rule")
        if len(rule) < len(title):
            rule = "=" * len(title)
        return f"{title}\n{rule}"

    return _BROKEN_H1_RE.sub(_join, text)


def _repair_gallery_headings(app, docname: str, source: list[str]) -> None:
    """Fix broken H1s in generated Habitat Guide index pages.

    Also write the repair back to disk so the next gallery pass does not
    leave ``Title\\n\\n====`` in ``auto_examples/**/index.rst``.
    """
    if not docname.startswith("auto_examples"):
        return
    source[0] = _repair_broken_rst_h1(source[0])
    path = Path(app.srcdir) / f"{docname}.rst"
    if path.is_file():
        on_disk = path.read_text(encoding="utf-8")
        fixed = _repair_broken_rst_h1(on_disk)
        if fixed != on_disk:
            path.write_text(fixed, encoding="utf-8")


def _lift_gallery_nested_toctree(app, docname: str, source: list[str]) -> None:
    """Move the Habitat Guide subsection toctree under the document title.

    Also demote inlined subsection headings (``====`` copied from each
    folder README) to ``----``. Those extra H1s are siblings of the
    document title, so RTD would otherwise repeat them as extra
    top-level sidebar entries next to Habitat Guide.

    Parameters
    ----------
    app:
        Sphinx application (unused; required by the ``source-read`` event).
    docname:
        Document name relative to the source directory.
    source:
        Single-element list with the RST text; rewritten in place.
    """
    if docname != "auto_examples/index":
        return
    text = source[0]
    match = _GALLERY_NESTED_TOCTREE_RE.search(text)
    if match is None:
        return
    toctree = match.group(0)
    remainder = text[: match.start()] + text[match.end() :]
    lines = remainder.splitlines(keepends=True)
    insert_at = None
    for i, line in enumerate(lines):
        if _is_rst_h1_rule(line):
            insert_at = i + 1
            break
    if insert_at is None:
        return
    if not toctree.startswith("\n"):
        toctree = "\n" + toctree
    if not toctree.endswith("\n"):
        toctree += "\n"
    lines.insert(insert_at, toctree)
    seen_root_title = False
    demoted: list[str] = []
    for line in lines:
        if _is_rst_h1_rule(line):
            if seen_root_title:
                demoted.append(line.replace("=", "-"))
                continue
            seen_root_title = True
        demoted.append(line)
    source[0] = "".join(demoted)


def _ensure_api_generated_dir(app) -> None:
    """Create ``api/generated`` before gallery / autodoc write ``.examples`` files.

    Parameters
    ----------
    app:
        Sphinx application. ``srcdir`` is the documentation source root.
    """
    generated = Path(app.srcdir) / "api" / "generated"
    generated.mkdir(parents=True, exist_ok=True)


def touch_example_backreferences(
    app,
    what: str,
    name: str,
    obj: object,
    options: dict,
    lines: list[str],
) -> None:
    """Create empty sphinx-gallery ``.examples`` files when a symbol has no hits.

    Autosummary stubs ``include`` ``{{fullname}}.examples``. Without a
    placeholder, objects that never appear in the Habitat Guide fail the
    build. Matches nilearn ``doc/conf.py``.

    Parameters
    ----------
    app:
        Sphinx application.
    what:
        Object type (``class``, ``function``, …). Required by the event.
    name:
        Fully-qualified object name (the ``.examples`` file stem).
    obj:
        The documented object. Required by the event; unused.
    options:
        Autodoc options. Required by the event; unused.
    lines:
        Docstring lines. Required by the event; unused.
    """
    del what, obj, options, lines
    if not name:
        return
    examples_path = Path(app.srcdir) / "api" / "generated" / f"{name}.examples"
    examples_path.parent.mkdir(parents=True, exist_ok=True)
    if not examples_path.exists():
        examples_path.touch()


def setup(app) -> dict[str, object]:
    """Register docs-only Sphinx event hooks."""
    app.connect("builder-inited", _ensure_api_generated_dir)
    app.connect("source-read", _repair_gallery_headings)
    app.connect("source-read", _lift_gallery_nested_toctree)
    app.connect("autodoc-process-docstring", touch_example_backreferences)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
