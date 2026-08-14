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

from typing import Any, Mapping, Optional, Union

from habit.exceptions import HABITAPIError
from habit.viz.labels import sanitize_label

__all__ = [
    "ColorbarSpec",
    "DEFAULT_COLORBAR_ASPECT",
    "DEFAULT_COLORBAR_FRACTION",
    "DEFAULT_COLORBAR_PAD",
    "DEFAULT_COLORBAR_SHRINK",
    "add_image_colorbar",
    "add_image_colorbar_from_spec",
    "colorbar_is_enabled",
    "colorbar_style_kwargs",
]

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
    cbar.ax.set_facecolor("white")
    _pin_colorbar_to_image(ax, cbar.ax, shrink_value)
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
