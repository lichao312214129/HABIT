# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
"""
Global font configuration for publication-quality plots.

Prefers Arial on Windows/macOS when installed; falls back to DejaVu Sans on
Linux/WSL where Arial is usually unavailable.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

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
