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
"""Habitat qualitative palette: distinct colours, Radiology-safe order."""

from __future__ import annotations

import numpy as np
import pytest

from habit.exceptions import HABITAPIError
from habit.viz.habitat_overlay import _habitat_color_list, _habitat_color_lookup
from habit.viz.palette import (
    HABITAT_QUALITATIVE_HEX,
    habitat_hex_colors,
    habitat_rgb_colors,
    hex_to_rgb,
)

pytestmark = pytest.mark.unit


def test_first_four_are_okabe_ito_not_two_oranges() -> None:
    """K=4 must be blue / vermillion / green / purple, not orange twice."""
    hexes = habitat_hex_colors(4)
    assert hexes == ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    assert "#E69F00" not in hexes


def test_ten_habitats_are_all_distinct() -> None:
    """K=10 (graph demo) must not wrap habitat 9 onto habitat 1."""
    hexes = habitat_hex_colors(10)
    assert len(hexes) == 10
    assert len(set(hexes)) == 10
    # Two light blues in the first 10 read as one colour on a thin bar.
    assert "#56B4E9" not in hexes
    assert "#88CCEE" not in hexes
    rgb = np.asarray(habitat_rgb_colors(10), dtype=float)
    for i in range(len(rgb)):
        for j in range(i + 1, len(rgb)):
            distance = float(np.linalg.norm(rgb[i] - rgb[j]))
            assert distance > 0.18, (hexes[i], hexes[j], distance)


def test_no_black_in_designed_bank() -> None:
    """Black disappears on dark MRI; the 8th slot is mid-grey."""
    assert "#000000" not in HABITAT_QUALITATIVE_HEX
    assert HABITAT_QUALITATIVE_HEX[7] == "#7E7E7E"


def test_overlay_lookup_matches_hex_bank() -> None:
    """Overlay and graph must paint the same ID with the same colour."""
    ids = (1, 2, 3, 4, 9, 10)
    lookup = _habitat_color_lookup(ids)
    hexes = habitat_hex_colors(len(ids))
    listed = _habitat_color_list(ids)
    for index, habitat_id in enumerate(ids):
        assert lookup[habitat_id] == hex_to_rgb(hexes[index])
        assert listed[index] == lookup[habitat_id]


def test_beyond_bank_still_unique() -> None:
    """Asking for more than 16 colours must not recycle an identical hex."""
    hexes = habitat_hex_colors(20)
    assert len(hexes) == 20
    assert len(set(hexes)) == 20


def test_hex_to_rgb_rejects_bad_input() -> None:
    """Invalid hex strings raise HABITAPIError."""
    with pytest.raises(HABITAPIError):
        hex_to_rgb("0072B2")
    with pytest.raises(HABITAPIError):
        habitat_hex_colors(0)
