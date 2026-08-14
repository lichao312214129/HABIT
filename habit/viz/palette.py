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
"""Qualitative colours for habitat maps (Radiology / color-blind safe).

Habitat IDs are categorical labels, not a low-to-high ramp. The default
bank is Okabe–Ito (Bang Wong) in the same order as :mod:`habit.viz.style`,
then Paul Tol muted extras so typical K=2–12 maps do not recycle a colour.

Black from the original Okabe–Ito octet is replaced by a mid-grey: black
vanishes on dark MRI and makes two habitats look like one background.
Orange is fifth, not second, so a 4-habitat colorbar is not two oranges.

Callers must request ``n`` colours (one per present ID). Do not silently
``% 8`` a short palette — that is what made docs colorbars show duplicates.
"""

from __future__ import annotations

import colorsys
from typing import List, Sequence, Tuple

from habit.exceptions import HABITAPIError

__all__ = [
    "HABITAT_QUALITATIVE_HEX",
    "hex_to_rgb",
    "habitat_hex_colors",
    "habitat_rgb_colors",
]

#: Okabe–Ito first seven (official order), mid-grey instead of black, then
#: Paul Tol muted extras (https://personal.sron.nl/~pault/). Sixteen slots
#: cover auto-K up to the usual demo maximum without a wrap.
HABITAT_QUALITATIVE_HEX: Tuple[str, ...] = (
    "#0072B2",  # 1  Okabe blue (cool, dark)
    "#D55E00",  # 2  Okabe vermillion (warm)
    "#009E73",  # 3  Okabe bluish green
    "#CC79A7",  # 4  Okabe reddish purple
    "#E69F00",  # 5  Okabe orange (not next to vermillion in the first four)
    "#332288",  # 6  Tol indigo (dark violet; not a second sky-blue)
    "#F0E442",  # 7  Okabe yellow
    "#7E7E7E",  # 8  mid-grey (visible on anatomy and on a white colorbar)
    "#882255",  # 9  Tol wine
    "#117733",  # 10 Tol forest (dark green; not a second cyan)
    "#88CCEE",  # 11 Tol cyan
    "#56B4E9",  # 12 Okabe sky blue (after the usual K<=10 set)
    "#DDCC77",  # 13 Tol sand
    "#44AA99",  # 14 Tol teal
    "#AA4499",  # 15 Tol purple
    "#999933",  # 16 Tol olive
)

RGB = Tuple[float, float, float]


def hex_to_rgb(color: str) -> RGB:
    """
    Convert ``#RRGGBB`` to an RGB triple in ``[0, 1]``.

    Args:
        color: Six-digit hex colour with a leading ``#``.

    Returns:
        ``(r, g, b)`` each in ``[0, 1]``.

    Raises:
        HABITAPIError: When the string is not a 7-character hex colour.
    """
    text = str(color).strip()
    if len(text) != 7 or not text.startswith("#"):
        raise HABITAPIError(
            f"hex_to_rgb: expected #RRGGBB, got {color!r}."
        )
    try:
        red = int(text[1:3], 16) / 255.0
        green = int(text[3:5], 16) / 255.0
        blue = int(text[5:7], 16) / 255.0
    except ValueError as exc:
        raise HABITAPIError(
            f"hex_to_rgb: expected #RRGGBB, got {color!r}."
        ) from exc
    return (red, green, blue)


def _rgb_to_hex(color: RGB) -> str:
    """Convert an RGB triple in ``[0, 1]`` to ``#RRGGBB``."""
    red, green, blue = color
    return "#{:02X}{:02X}{:02X}".format(
        int(round(max(0.0, min(1.0, red)) * 255.0)),
        int(round(max(0.0, min(1.0, green)) * 255.0)),
        int(round(max(0.0, min(1.0, blue)) * 255.0)),
    )


def _shift_hex_lightness(color: str, factor: float) -> str:
    """
    Lighten or darken a hex colour in HLS space (same hue).

    Used only when the caller asks for more colours than the designed
    16-colour bank. The result is never identical to ``color``.

    Args:
        color: ``#RRGGBB`` source.
        factor: Multiplier for lightness (``< 1`` darker, ``> 1`` lighter).

    Returns:
        A different ``#RRGGBB`` with the same hue.
    """
    red, green, blue = hex_to_rgb(color)
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    shifted = max(0.12, min(0.88, lightness * float(factor)))
    if abs(shifted - lightness) < 0.08:
        shifted = 0.35 if lightness > 0.5 else 0.72
    new_rgb = colorsys.hls_to_rgb(hue, shifted, saturation)
    return _rgb_to_hex((new_rgb[0], new_rgb[1], new_rgb[2]))


def habitat_hex_colors(
    n: int,
    base: Sequence[str] = HABITAT_QUALITATIVE_HEX,
) -> List[str]:
    """
    Return ``n`` distinct hex colours for habitat IDs  (sorted-ID order).

    The first ``min(n, 16)`` colours are the designed Radiology-safe bank.
    Extra slots shift lightness of earlier hues so two IDs never share a
    hex (the old ``index % 8`` wrap).

    Args:
        n: Number of positive habitat IDs to colour.
        base: Optional replacement bank (hex strings). Defaults to
            :data:`HABITAT_QUALITATIVE_HEX`.

    Returns:
        ``n`` ``#RRGGBB`` strings.

    Raises:
        HABITAPIError: When ``n < 1`` or ``base`` is empty.
    """
    count = int(n)
    if count < 1:
        raise HABITAPIError("habitat_hex_colors: n must be >= 1.")
    bank = tuple(str(item) for item in base)
    if not bank:
        raise HABITAPIError("habitat_hex_colors: base palette must not be empty.")
    if count <= len(bank):
        return list(bank[:count])
    extra: List[str] = []
    need = count - len(bank)
    for index in range(need):
        source = bank[index % len(bank)]
        wave = index // len(bank)
        factor = 0.68 if wave % 2 == 0 else 1.22
        extra.append(_shift_hex_lightness(source, factor))
    return list(bank) + extra


def habitat_rgb_colors(
    n: int,
    base: Sequence[str] = HABITAT_QUALITATIVE_HEX,
) -> List[RGB]:
    """
    Same as :func:`habitat_hex_colors`, as RGB triples in ``[0, 1]``.

    Args:
        n: Number of positive habitat IDs to colour.
        base: Optional replacement hex bank.

    Returns:
        ``n`` ``(r, g, b)`` triples.
    """
    return [hex_to_rgb(item) for item in habitat_hex_colors(n, base=base)]
