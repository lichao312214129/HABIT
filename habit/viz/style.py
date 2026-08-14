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
"""Journal style presets, applied as a matplotlib context.

HABIT's figures are drawn ONCE and published in MANY places, and each journal
has its own physical constraints: column width, minimum resolution, whether
sans or serif type is expected. Encoding those as named presets -- applied
through a context manager that touches nothing outside the figure -- is what
lets the same plot function serve a draft and a camera-ready figure without a
single edit to the plotting code.

The presets are deliberately conservative: a white background, no chartjunk,
and a colour palette that stays distinguishable in greyscale and for the most
common colour-vision deficiencies. A preset is a **geometry / typography
starting point**, not a full submission checklist (file format, panel-letter
case, de-identification, figure legends, and venue redraw policies still
belong to the author).

Built-in numbers are taken from public author guides (verify before
submission; venues change specs):

* ``nature`` — Nature final-submission / final-artwork guides
  (https://www.nature.com/nature/for-authors/final-submission,
  https://www.nature.com/documents/nature-final-artwork.pdf):
  89 mm / 183 mm; Helvetica or Arial; other text max 7 pt / min 5 pt.
* ``radiology`` — RSNA *Radiology* Instructions for Authors
  (https://pubs.rsna.org/page/radiology/author-instructions):
  imaging panels 3–7 inches wide at ≥300 dpi; graphs/illustrations at
  1200 dpi; sans-serif (Arial) labels.
* ``lancet`` — Lancet family / *The Lancet Digital Health* Information for
  Authors (figures ≥300 dpi and ≥107 mm wide; Times New Roman for figure
  headings/legends; most line art is redrawn in house):
  https://www.thelancet.com/pb-assets/Lancet/authors/tldh-info-for-authors-1778587678573.pdf
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Dict, Iterator, Tuple

from habit.exceptions import HABITAPIError
from habit.utils.optional_deps import require

__all__ = [
    "StyleSpec",
    "use_style",
    "get_style",
    "register_style",
    "available_styles",
]

#: Millimetres per inch, for converting journal column widths to matplotlib's
#: inch-based figure size.
_MM_PER_INCH = 25.4

#: Prefer journal-named faces first; DejaVu remains a portable fallback so
#: headless CI without Arial/Times still renders.
_SANS_FONT_LIST = ("Arial", "Helvetica", "DejaVu Sans")
_SERIF_FONT_LIST = ("Times New Roman", "Times", "DejaVu Serif")

#: Okabe–Ito colour-blind-safe palette (also survives greyscale printing).
_OKABE_ITO: Tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)


@dataclass(frozen=True)
class StyleSpec:
    """
    A named bundle of figure-geometry and typography choices.

    Attributes:
        name: Preset identifier used with :func:`use_style`.
        single_column_mm: Width of a single-column (or minimum submission)
            figure in millimetres.
        double_column_mm: Width of a full-width figure in millimetres.
        dpi: Default ``savefig.dpi`` hint for raster export. Callers may
            override at ``savefig`` time (e.g. RSNA line art at 1200 dpi).
        font_family: ``"sans-serif"`` or ``"serif"``.
        font_size: Base font size in points; tick/legend sizes scale from it.
        line_width: Base line width in points.
        palette: Colour cycle for groups/models.
        extra_rcparams: Any further rcParams the preset overrides.
    """

    name: str
    single_column_mm: float = 89.0
    double_column_mm: float = 183.0
    dpi: int = 300
    font_family: str = "sans-serif"
    font_size: float = 8.0
    line_width: float = 1.0
    palette: Tuple[str, ...] = field(default=_OKABE_ITO)
    extra_rcparams: Dict[str, object] = field(default_factory=dict)

    def rcparams(self) -> Dict[str, object]:
        """Translate the spec into a matplotlib rcParams mapping."""
        # Do NOT set savefig.bbox='tight': tight cropping changes the physical
        # width so the figure no longer matches the journal column millimetres.
        base: Dict[str, object] = {
            "figure.dpi": 100.0,  # on-screen; the saved DPI is set at savefig
            "savefig.dpi": float(self.dpi),
            "font.family": self.font_family,
            "font.size": self.font_size,
            "axes.titlesize": self.font_size,
            "axes.labelsize": self.font_size,
            "axes.linewidth": self.line_width,
            "axes.edgecolor": "#333333",
            "axes.grid": False,
            "axes.prop_cycle": _color_cycle(self.palette),
            "xtick.labelsize": max(self.font_size - 1.0, 5.0),
            "ytick.labelsize": max(self.font_size - 1.0, 5.0),
            "xtick.major.width": self.line_width,
            "ytick.major.width": self.line_width,
            "legend.fontsize": max(self.font_size - 1.0, 5.0),
            "legend.frameon": False,
            "lines.linewidth": self.line_width,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,  # embed TrueType so editors can re-typeset
            "ps.fonttype": 42,
            "svg.fonttype": "none",  # keep text as text in SVG
            # ASCII hyphen-minus on ticks (U+2212 is not journal-safe here).
            "axes.unicode_minus": False,
        }
        if self.font_family == "serif":
            base["font.serif"] = list(_SERIF_FONT_LIST)
        else:
            base["font.sans-serif"] = list(_SANS_FONT_LIST)
        base.update(self.extra_rcparams)
        return base

    def figsize(self, *, columns: int = 1, height_mm: float | None = None) -> Tuple[float, float]:
        """
        Return an inch figure size for a one- or two-column figure.

        Args:
            columns: ``1`` for a single-column-width figure, ``2`` for full
                text width.
            height_mm: Optional explicit height in millimetres; defaults to a
                4:3-ish proportion of the chosen width.

        Returns:
            ``(width_in, height_in)`` suitable for ``matplotlib``.
        """
        width_mm = self.single_column_mm if columns == 1 else self.double_column_mm
        if height_mm is None:
            height_mm = width_mm * 0.75
        return (width_mm / _MM_PER_INCH, height_mm / _MM_PER_INCH)


def _color_cycle(palette: Tuple[str, ...]):
    """Build an rcParams colour cycler without importing matplotlib eagerly."""
    from cycler import cycler

    return cycler(color=list(palette))


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, StyleSpec] = {}


def _register_defaults() -> None:
    """Populate the built-in presets once, at import."""
    # Draft / generic: Nature-like geometry at a lighter raster DPI.
    default = StyleSpec(name="default")

    # RSNA Radiology author instructions (imaging 3–7 in @ ≥300 dpi;
    # graphs/illustrations 1200 dpi; Arial sans-serif labels).
    # savefig.dpi=600 is a practical combo-figure default; override to 1200
    # for pure line art when submitting revisions.
    # Imaging callouts in RSNA guides use ≥12 pt Arial; axis / tick text stays
    # at 8 pt here — enlarge annotation artists separately when labelling
    # medical-image panels.
    radiology = replace(
        default,
        name="radiology",
        single_column_mm=3.0 * _MM_PER_INCH,  # 76.2 mm — official minimum
        double_column_mm=7.0 * _MM_PER_INCH,  # 177.8 mm — official maximum
        dpi=600,
        font_family="sans-serif",
        font_size=8.0,
    )

    # Nature final artwork: 89 / 183 mm; Helvetica or Arial; other text ≤7 pt.
    nature = replace(
        default,
        name="nature",
        single_column_mm=89.0,
        double_column_mm=183.0,
        dpi=600,
        font_family="sans-serif",
        font_size=7.0,
        line_width=0.8,  # Nature guidance: strokes ~0.25–1 pt at final size
    )

    # Lancet Digital Health / Lancet family author guides: figures ≥107 mm
    # wide at ≥300 dpi; Times New Roman for headings/legends; no box around
    # graphs; line art is typically redrawn in house. Double width uses a
    # full-page layout (~180 mm) because the public PDF does not publish a
    # second column millimetre for charts (only the 107 mm minimum).
    lancet = replace(
        default,
        name="lancet",
        single_column_mm=107.0,
        double_column_mm=180.0,
        dpi=600,
        font_family="serif",
        font_size=8.0,
        extra_rcparams={
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )
    for spec in (default, radiology, nature, lancet):
        _REGISTRY[spec.name] = spec


def register_style(spec: StyleSpec) -> None:
    """
    Register a custom preset, making it usable by name in :func:`use_style`.

    Args:
        spec: The preset to register. Re-registering an existing name
            replaces it, so a project can ship its own house style.
    """
    _REGISTRY[spec.name] = spec


def get_style(name: str) -> StyleSpec:
    """
    Return the preset registered under ``name``.

    Args:
        name: Preset identifier, e.g. ``"radiology"``.

    Returns:
        The :class:`StyleSpec`.

    Raises:
        HABITAPIError: If the name is not registered.
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise HABITAPIError(
            f"Unknown viz style {name!r}. Available: {available_styles()}. "
            "Register a custom one with habit.viz.register_style()."
        ) from exc


def available_styles() -> Tuple[str, ...]:
    """Return the registered preset names, sorted."""
    return tuple(sorted(_REGISTRY))


@contextmanager
def use_style(name: str | StyleSpec) -> Iterator[StyleSpec]:
    """
    Apply a preset's rcParams for the duration of the block.

    Args:
        name: A registered preset name, or a :class:`StyleSpec` for a one-off
            style that need not be registered.

    Yields:
        The active :class:`StyleSpec`, so the block can read sizes and
        palette from it.

    Example:
        >>> from habit.viz import use_style, plot_kaplan_meier  # doctest: +SKIP
        >>> with use_style("radiology") as style:
        ...     fig = plot_kaplan_meier(...)
        >>> fig.savefig("km.tiff", dpi=style.dpi)  # doctest: +SKIP
    """
    spec = name if isinstance(name, StyleSpec) else get_style(name)
    # matplotlib is an OPTIONAL dependency (habitat-analysis[viz]). Style
    # presets can be registered and inspected without it; only ACTIVATING one
    # touches rcParams, so the gate sits here.
    mpl = require("matplotlib", extra="viz", purpose="applying a figure style preset")

    with mpl.rc_context(spec.rcparams()):
        yield spec


_register_defaults()
