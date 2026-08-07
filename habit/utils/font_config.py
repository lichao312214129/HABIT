# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Global font and backend configuration for publication-quality plots.

Prefers Arial on Windows/macOS when installed; falls back to DejaVu Sans on
Linux/WSL where Arial is usually unavailable.

Also selects a non-interactive matplotlib backend by default, because HABIT
workflows only write figures to disk and never run a GUI event loop.
"""

from __future__ import annotations

import os
from typing import Dict, List

from habit.utils.optional_deps import require

#: What this module needs matplotlib for, reused by every gate below so the
#: three import failures all read the same way.
_VIZ_PURPOSE = "publication-quality figures (font and backend configuration)"

# matplotlib is an OPTIONAL dependency (habitat-analysis[viz]). This module has
# no reason to exist without it -- every symbol it exports configures
# matplotlib -- so the gate sits at module scope: importing this module without
# the viz extra raises OptionalDependencyError (naming the extra) instead of a
# bare ModuleNotFoundError. Nothing on HABIT's habitat kernel path imports it.
mpl = require("matplotlib", extra="viz", purpose=_VIZ_PURPOSE)

_BACKEND_ENV_VAR = "HABIT_MPL_BACKEND"
_DEFAULT_BACKEND = "Agg"


def _select_backend() -> str:
    """
    Select and activate the matplotlib backend used by HABIT.

    HABIT saves every figure to disk, so an interactive backend such as TkAgg
    only adds GUI objects that are never driven by an event loop. Those objects
    raise ``RuntimeError: main thread is not in main loop`` inside tkinter
    destructors when garbage collection happens on a worker thread (for example
    during AutoGluon training or SHAP plotting). A non-interactive backend
    avoids the problem entirely.

    Set the ``HABIT_MPL_BACKEND`` environment variable to override the default,
    e.g. ``HABIT_MPL_BACKEND=TkAgg`` when interactive windows are wanted.

    Returns:
        str: Name of the backend that is active after this call.
    """
    requested: str = (os.environ.get(_BACKEND_ENV_VAR) or _DEFAULT_BACKEND).strip()
    try:
        mpl.use(requested, force=True)
    except Exception:
        # An unavailable backend must not break plotting; keep matplotlib's own
        # choice instead of failing the whole workflow.
        pass
    return mpl.get_backend()


ACTIVE_BACKEND: str = _select_backend()

# pyplot must be imported after the backend is selected so that the canvas
# classes bound at import time match ACTIVE_BACKEND.
plt = require("matplotlib.pyplot", extra="viz", purpose=_VIZ_PURPOSE)
font_manager = require(
    "matplotlib.font_manager", extra="viz", purpose=_VIZ_PURPOSE
)


def is_interactive_backend() -> bool:
    """
    Check whether the active matplotlib backend can display figure windows.

    Returns:
        bool: True when ``plt.show()`` is able to open a window.
    """
    return mpl.get_backend().lower() not in {"agg", "pdf", "ps", "svg", "cairo", "template"}


def show_or_close_figure() -> None:
    """
    Display the current figure interactively, or release it when running headless.

    Under a non-interactive backend ``plt.show()`` cannot open a window and only
    emits a warning, while the figure would stay in memory. Closing it instead
    keeps long batch runs from accumulating figures.
    """
    if is_interactive_backend():
        plt.show()
    else:
        plt.close()


_PREFERRED_FONT = "Arial"
_FALLBACK_FONTS: List[str] = [
    "DejaVu Sans",
    "Liberation Sans",
    "Bitstream Vera Sans",
    "sans-serif",
]


def _available_font_names() -> set[str]:
    """Return normalized font names registered with matplotlib."""
    return {info.name for info in font_manager.fontManager.ttflist}


def _is_font_available(font_name: str) -> bool:
    """
    Check whether a font family is available to matplotlib.

    Args:
        font_name: Candidate font family name.

    Returns:
        bool: True when matplotlib can resolve the font.
    """
    needle = font_name.lower()
    return any(needle in name.lower() for name in _available_font_names())


def resolve_publication_font() -> str:
    """
    Resolve the best publication font for the current platform.

    Returns:
        str: ``Arial`` when installed, otherwise the first usable fallback.
    """
    if _is_font_available(_PREFERRED_FONT):
        return _PREFERRED_FONT
    for candidate in _FALLBACK_FONTS:
        if candidate == "sans-serif":
            continue
        if _is_font_available(candidate):
            return candidate
    return "DejaVu Sans"


PUBLICATION_FONT: str = resolve_publication_font()


def _build_sans_serif_stack(primary_font: str) -> List[str]:
    """Build a sans-serif stack with the resolved primary font first."""
    stack = [primary_font]
    for candidate in [_PREFERRED_FONT, *_FALLBACK_FONTS]:
        if candidate not in stack:
            stack.append(candidate)
    return stack


def setup_publication_font() -> Dict[str, object]:
    """
    Configure matplotlib for publication-quality plots.

    Uses Arial when available; otherwise selects a Linux-safe fallback so WSL
    runs do not emit repeated ``findfont`` warnings.

    Returns:
        Dict[str, object]: Applied rcParams fragment.
    """
    primary_font = resolve_publication_font()
    font_config: Dict[str, object] = {
        "font.family": "sans-serif",
        "font.sans-serif": _build_sans_serif_stack(primary_font),
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.2,
    }

    mpl.rcParams.update(font_config)
    plt.rcParams.update(font_config)
    return font_config


def get_font_config() -> Dict[str, object]:
    """
    Return standard font kwargs for matplotlib/plotly text elements.

    Returns:
        Dict[str, object]: ``fontfamily`` and ``fontsize`` for plot calls.
    """
    return {
        "fontfamily": resolve_publication_font(),
        "fontsize": 10,
    }


def apply_font_to_text_elements(ax, fontfamily: str | None = None) -> None:
    """
    Apply the publication font to all text elements in a matplotlib axis.

    Args:
        ax: Matplotlib axis object.
        fontfamily: Optional override; defaults to :data:`PUBLICATION_FONT`.
    """
    resolved_font = fontfamily or resolve_publication_font()

    if ax.get_title():
        ax.set_title(ax.get_title(), fontfamily=resolved_font)

    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontfamily=resolved_font)
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontfamily=resolved_font)

    for label in ax.get_xticklabels():
        label.set_fontfamily(resolved_font)
    for label in ax.get_yticklabels():
        label.set_fontfamily(resolved_font)

    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_fontfamily(resolved_font)


setup_publication_font()
