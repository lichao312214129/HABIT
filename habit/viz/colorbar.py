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
"""Short vertical colorbars for slice and heatmap figures.

Matplotlib's default ``fig.colorbar(..., fraction=0.046)`` sizes the bar to
the full parent axes, including title padding. On a wide liver slice drawn
with ``aspect='equal'`` that makes the bar taller than the image. This
helper keeps a shorter, thinner bar that tracks the image, and stays
compatible with ``constrained_layout`` / ``tight_layout`` by using
``Figure.colorbar`` (not ``make_axes_locatable``, which fights those
layout engines).

This module is an internal viz helper. It is not part of the public API.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

from habit.exceptions import HABITAPIError
from habit.viz.labels import sanitize_label

__all__ = [
    "ColorbarSpec",
    "DEFAULT_COLORBAR_ASPECT",
    "DEFAULT_COLORBAR_FRACTION",
    "DEFAULT_COLORBAR_PAD",
    "DEFAULT_COLORBAR_SHRINK",
    "DEFAULT_HABITAT_CBAR_LABEL",
    "add_discrete_habitat_colorbar",
    "add_image_colorbar",
    "add_image_colorbar_from_spec",
    "colorbar_is_enabled",
    "colorbar_style_kwargs",
    "discrete_habitat_mappable",
]

#: English colorbar label for integer habitat-ID maps (journal figures).
DEFAULT_HABITAT_CBAR_LABEL: str = "Habitat"

#: Height of the bar relative to the parent axes when the image is
#: square or tall. Obviously shorter than a full-height axes bar.
DEFAULT_COLORBAR_SHRINK: float = 0.72
#: Width of the bar as a fraction of the parent axes (thinner than
#: matplotlib's default ``0.046``).
DEFAULT_COLORBAR_FRACTION: float = 0.028
#: Gap between the parent axes and the colorbar (smaller than ``0.04``).
DEFAULT_COLORBAR_PAD: float = 0.02
#: Long-to-short ratio of the bar; higher is thinner.
DEFAULT_COLORBAR_ASPECT: float = 22.0

ColorbarSpec = Union[bool, Mapping[str, Any]]


def colorbar_is_enabled(colorbar: ColorbarSpec) -> bool:
    """
    Return whether a colorbar should be drawn.

    ``False`` turns it off. ``True`` and any mapping (including ``{}``)
    turn it on. A mapping may set ``enabled=False`` to disable while
    still looking like a kwargs dict.

    Args:
        colorbar: ``True`` / ``False`` or a style mapping.

    Returns:
        ``True`` when a colorbar should be attached.
    """
    if colorbar is False:
        return False
    if colorbar is True:
        return True
    if isinstance(colorbar, Mapping):
        return colorbar.get("enabled", True) is not False
    return bool(colorbar)


def colorbar_style_kwargs(colorbar: ColorbarSpec) -> dict[str, Any]:
    """
    Extract style kwargs from a mapping spec.

    Drops the reserved ``enabled`` flag so the rest can be forwarded to
    :func:`add_image_colorbar`.

    Args:
        colorbar: ``True`` / ``False`` or a style mapping.

    Returns:
        A shallow copy of the mapping without ``enabled``. Empty when
        ``colorbar`` is not a mapping.
    """
    if not isinstance(colorbar, Mapping):
        return {}
    out = dict(colorbar)
    out.pop("enabled", None)
    return out


def _pin_colorbar_to_image(ax: Any, cbar_ax: Any, shrink: float) -> None:
    """
    Keep the colorbar height locked to the displayed image box.

    ``aspect='equal'`` / ``adjustable='box'`` shrinks the image axes to
    the array aspect, but ``Figure.colorbar`` installs a locator that
    still sizes the bar to the original subplot slot (title padding
    included). That is why a flat liver slice grew a bar taller than
    the image.

    Image axes are created first, so their ``apply_aspect`` (letterbox)
    runs before the colorbar locator. We wrap that locator and reuse
    its x/width (layout-reserved slot) while matching y/height to the
    already-letterboxed parent. This stays compatible with
    constrained_layout / tight_layout without ``make_axes_locatable``.

    Args:
        ax: Parent image axes.
        cbar_ax: Colorbar axes created by ``Figure.colorbar``.
        shrink: Extra height scale relative to the displayed image box
            (``1.0`` matches the image; ``0.72`` is the default short bar).
    """
    from matplotlib.transforms import Bbox

    original_locator = cbar_ax.get_axes_locator()

    def _locator(cax: Any, renderer: Any) -> Any:
        if original_locator is not None:
            slot = original_locator(cax, renderer)
        else:
            slot = cax._original_position.frozen()
        # Parent apply_aspect has already letterboxed _position.
        image_pos = ax._position.frozen()
        new_height = image_pos.height * float(shrink)
        if new_height <= 0.0:
            return slot
        new_y = image_pos.y0 + 0.5 * (image_pos.height - new_height)
        return Bbox.from_bounds(slot.x0, new_y, slot.width, new_height)

    cbar_ax.set_axes_locator(_locator)
    cbar_ax._habit_cbar_shrink = float(shrink)


def add_image_colorbar(
    mappable: Any,
    ax: Any = None,
    *,
    label: Optional[str] = None,
    shrink: Optional[Union[float, str]] = None,
    pad: Optional[float] = None,
    fraction: Optional[float] = None,
    aspect: Optional[float] = None,
    ticks: Optional[Any] = None,
    ticklabels: Optional[Sequence[Any]] = None,
    extend: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Attach a short vertical colorbar next to ``ax``.

    Compatible with ``constrained_layout`` and ``tight_layout`` because it
    uses ``Figure.colorbar``. All figure text is ASCII-sanitised.

    Args:
        mappable: Matplotlib image / ``ScalarMappable``.
        ax: Parent axes. Defaults to ``mappable.axes``.
        label: Optional colorbar label (ASCII-sanitised).
        shrink: Height relative to the **displayed image box** after
            ``aspect='equal'`` letterboxing. ``None`` / ``\"auto\"`` use
            ``0.72`` so the bar is clearly shorter than the image and
            does not stick out above or below a flat slice. Pass ``1.0``
            to match the image height exactly.
        pad: Gap between axes and colorbar (default ``0.02``).
        fraction: Colorbar width as a fraction of the parent (default
            ``0.028``, thinner than matplotlib's ``0.046``).
        aspect: Long/short ratio of the bar (default ``22``, thinner).
        ticks: Optional tick locations forwarded to the colorbar.
        ticklabels: Optional tick labels (ASCII-sanitised). Used for
            discrete habitat bars so display slots ``1..K`` can show the
            original integer habitat IDs.
        extend: Matplotlib ``extend`` mode (``\"neither\"``, ``\"min\"``,
            ``\"max\"``, ``\"both\"``).
        **kwargs: Extra arguments forwarded to ``Figure.colorbar``.

    Returns:
        The matplotlib ``Colorbar``.

    Raises:
        HABITAPIError: When no parent axes can be resolved.
    """
    if ax is None:
        ax = getattr(mappable, "axes", None)
    if ax is None:
        raise HABITAPIError(
            "add_image_colorbar: ax is required when mappable has no axes."
        )
    fig = ax.figure
    if shrink is None or shrink == "auto":
        shrink_value = DEFAULT_COLORBAR_SHRINK
    else:
        shrink_value = float(shrink)
    # Let the layout engine reserve a thin full-height slot. Visual
    # height is applied afterwards so aspect-equal letterboxing is
    # respected (Figure.colorbar shrink uses the subplot slot, not
    # the displayed image box).
    kwargs.pop("shrink", None)
    cbar_kwargs: dict[str, Any] = {
        "ax": ax,
        "fraction": DEFAULT_COLORBAR_FRACTION if fraction is None else float(fraction),
        "pad": DEFAULT_COLORBAR_PAD if pad is None else float(pad),
        "aspect": DEFAULT_COLORBAR_ASPECT if aspect is None else float(aspect),
        "location": "right",
    }
    if extend is not None:
        cbar_kwargs["extend"] = extend
    cbar_kwargs.update(kwargs)
    cbar = fig.colorbar(mappable, **cbar_kwargs)
    if label is not None:
        cbar.set_label(sanitize_label(label))
    if ticks is not None:
        cbar.set_ticks(ticks)
    if ticklabels is not None:
        cbar.set_ticklabels([sanitize_label(str(text)) for text in ticklabels])
    cbar.ax.set_facecolor("white")
    _pin_colorbar_to_image(ax, cbar.ax, shrink_value)
    return cbar


def discrete_habitat_mappable(
    habitat_ids: Sequence[int],
    colors: Sequence[Any],
) -> Tuple[Any, List[float], List[str]]:
    """
    Build a discrete ScalarMappable keyed by integer habitat IDs.

    Background ``0`` is excluded. Each present ID becomes one equal-sized
    colour block (ListedColormap + BoundaryNorm) even when the ID sequence
    has gaps (``1, 3, 5`` → three blocks labelled ``1``, ``3``, ``5``).
    Tick locations sit at the centre of each block so the bar is a
    categorical legend, not a continuous smear.

    Args:
        habitat_ids: Positive integer habitat IDs (``0`` is ignored).
            Order is preserved after de-duplication.
        colors: RGB tuples or hex strings, cycled if shorter than the
            unique ID list.

    Returns:
        ``(mappable, tick_locations, tick_labels)``. Tick locations are
        ``1..K`` in display space; labels are the original habitat IDs.

    Raises:
        HABITAPIError: When no positive habitat IDs remain, or ``colors``
            is empty.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap

    ordered: List[int] = []
    seen: set[int] = set()
    for raw in habitat_ids:
        habitat_id = int(raw)
        if habitat_id <= 0 or habitat_id in seen:
            continue
        seen.add(habitat_id)
        ordered.append(habitat_id)
    if not ordered:
        raise HABITAPIError(
            "discrete_habitat_mappable: no positive habitat IDs."
        )
    if not colors:
        raise HABITAPIError(
            "discrete_habitat_mappable: colors must not be empty."
        )
    n_habitats = len(ordered)
    face_colors = [colors[index % len(colors)] for index in range(n_habitats)]
    cmap = ListedColormap(face_colors)
    # Boundaries at 0.5, 1.5, ..., K+0.5 centre integer ticks on each block.
    boundaries = [0.5 + float(index) for index in range(n_habitats + 1)]
    norm = BoundaryNorm(boundaries, ncolors=n_habitats)
    mappable = ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    ticks = [float(index) for index in range(1, n_habitats + 1)]
    ticklabels = [str(habitat_id) for habitat_id in ordered]
    return mappable, ticks, ticklabels


def add_discrete_habitat_colorbar(
    ax: Any,
    habitat_ids: Sequence[int],
    colors: Sequence[Any],
    *,
    colorbar: ColorbarSpec = True,
    label: str = DEFAULT_HABITAT_CBAR_LABEL,
    **defaults: Any,
) -> Any:
    """
    Attach a discrete habitat-ID colorbar, or skip when disabled / empty.

    One tick and one opaque colour per positive habitat ID. Background
    ``0`` does not appear on the bar.

    Args:
        ax: Parent image axes.
        habitat_ids: Positive integer habitat IDs.
        colors: Palette aligned with ``habitat_ids`` (cycled if shorter).
        colorbar: ``True`` / ``False`` or a style mapping (same spec as
            :func:`add_image_colorbar_from_spec`).
        label: Default colorbar label (English ``\"Habitat\"``).
        **defaults: Extra :func:`add_image_colorbar` kwargs.

    Returns:
        The matplotlib ``Colorbar``, or ``None`` when disabled or when
        there are no positive habitat IDs.
    """
    if not colorbar_is_enabled(colorbar):
        return None
    ordered = [int(v) for v in habitat_ids if int(v) > 0]
    if not ordered:
        return None
    mappable, ticks, ticklabels = discrete_habitat_mappable(ordered, colors)
    n_habitats = len(ticks)
    # Pass explicit boundaries so Colorbar draws flat blocks, not a
    # linearly interpolated smear of the ListedColormap.
    boundaries = [0.5 + float(index) for index in range(n_habitats + 1)]
    cbar = add_image_colorbar_from_spec(
        mappable,
        colorbar,
        ax=ax,
        label=label,
        ticks=ticks,
        ticklabels=ticklabels,
        boundaries=boundaries,
        spacing="uniform",
        drawedges=True,
        **defaults,
    )
    if cbar is not None:
        # Pin ticks to block centres. Matplotlib's default colorbar locator
        # otherwise snaps to boundaries and the bar reads as a continuum.
        from matplotlib.ticker import FixedFormatter, FixedLocator

        cbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
        cbar.ax.yaxis.set_major_formatter(
            FixedFormatter([sanitize_label(str(text)) for text in ticklabels])
        )
        cbar.minorticks_off()
        cbar.ax.tick_params(which="major", length=3.0, width=0.6)
    return cbar


def add_image_colorbar_from_spec(
    mappable: Any,
    colorbar: ColorbarSpec,
    ax: Any = None,
    *,
    label: Optional[str] = None,
    **defaults: Any,
) -> Any:
    """
    Draw a colorbar from a ``bool`` / mapping spec, or skip when off.

    Mapping keys are forwarded to :func:`add_image_colorbar` and override
    ``defaults`` / ``label``. Use this at plotter call sites so
    ``colorbar=False`` and ``colorbar={\"shrink\": 0.6}`` share one path.

    Args:
        mappable: Matplotlib image / ``ScalarMappable``.
        colorbar: ``True`` / ``False`` or a style mapping.
        ax: Parent axes. Defaults to ``mappable.axes``.
        label: Default label when the spec does not set ``label``.
        **defaults: Extra :func:`add_image_colorbar` kwargs (for example
            ``extend=``). Spec keys win on conflict.

    Returns:
        The matplotlib ``Colorbar``, or ``None`` when disabled.
    """
    if not colorbar_is_enabled(colorbar):
        return None
    spec = colorbar_style_kwargs(colorbar)
    resolved_label = spec.pop("label", label)
    merged = {**defaults, **spec}
    return add_image_colorbar(mappable, ax=ax, label=resolved_label, **merged)
