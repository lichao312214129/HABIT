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
"""Unit tests for the habitat-matching figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from habit.exceptions import HABITAPIError
from habit.kernels.habitat_label_match import match_rows_to_prototypes
from habit.viz import plot_label_overlap_matrix, plot_prototype_matching


def _outlined_cells(fig) -> set:
    """Return ``(row, col)`` of every outlined (Hungarian) cell."""
    cells = set()
    for patch in fig.axes[0].patches:
        x, y = patch.get_xy()
        cells.add((int(round(y + 0.5)), int(round(x + 0.5))))
    return cells


def test_overlap_matrix_outlines_the_hungarian_pairs() -> None:
    """Swapped ids 1 <-> 2 are outlined off the diagonal; 3 stays on it."""
    reference = np.array([1, 1, 1, 2, 2, 3, 3, 0])
    moving = np.array([2, 2, 2, 1, 1, 3, 3, 0])
    fig = plot_label_overlap_matrix(reference, moving)
    # Rows are moving ids (1, 2, 3); columns reference ids (1, 2, 3).
    assert _outlined_cells(fig) == {(0, 1), (1, 0), (2, 2)}
    numbers = [t.get_text() for t in fig.axes[0].texts]
    assert numbers.count("3") == 1 and numbers.count("2") == 2
    plt.close(fig)


def test_overlap_matrix_marks_extra_moving_habitat_as_new() -> None:
    """A fourth moving habitat has no partner and is tick-labelled new."""
    reference = np.array([1, 1, 2, 2, 3, 3])
    moving = np.array([1, 1, 2, 2, 3, 4])
    fig = plot_label_overlap_matrix(reference, moving, normalize=True)
    ticks = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert ticks == ["H1", "H2", "H3", "H4 (new)"]
    assert len(_outlined_cells(fig)) == 3
    plt.close(fig)


def test_overlap_matrix_rejects_shape_mismatch() -> None:
    """Different grids cannot be overlap-matched."""
    with pytest.raises(HABITAPIError):
        plot_label_overlap_matrix(np.ones((2, 3), int), np.ones((3, 2), int))


def test_prototype_plot_draws_one_point_per_habitat() -> None:
    """Kernel result: every habitat is a point, prototypes are crosses."""
    blocks = [
        np.array([[0.0, 0.0], [5.0, 5.0]]),
        np.array([[5.2, 4.9], [0.1, -0.2], [9.0, 1.0]]),
    ]
    match = match_rows_to_prototypes(blocks)
    fig = plot_prototype_matching(blocks, match, block_names=["A", "B"])
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert labels == ["P1", "P2", "P3"]
    legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert legend[:2] == ["A", "B"] and "prototype" in legend
    plt.close(fig)


def test_prototype_plot_cosine_draws_rays_not_crosses() -> None:
    """Cosine prototypes are directions: no cross markers, one ray each."""
    blocks = [np.array([[1.0, 0.1], [0.1, 1.0]]), np.array([[3.0, 0.2], [0.2, 2.5]])]
    match = match_rows_to_prototypes(blocks, metric="cosine")
    fig = plot_prototype_matching(blocks, match, show_links=False)
    assert len(fig.axes[0].texts) == 0
    assert len(fig.axes[0].lines) == 2
    plt.close(fig)


def test_prototype_plot_marks_unnamed_rows() -> None:
    """``max_distance`` leftovers get the 'unnamed' legend entry."""
    blocks = [np.array([[0.0, 0.0], [5.0, 5.0]]), np.array([[0.1, 0.0], [50.0, 50.0]])]
    match = match_rows_to_prototypes(blocks, max_distance=2.0)
    fig = plot_prototype_matching(blocks, match)
    legend = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "unnamed" in legend
    plt.close(fig)


def test_prototype_plot_rejects_foreign_blocks() -> None:
    """Blocks must be the ones that were matched."""
    blocks = [np.zeros((2, 2)), np.ones((2, 2))]
    match = match_rows_to_prototypes(blocks)
    with pytest.raises(HABITAPIError):
        plot_prototype_matching([np.zeros((3, 2)), np.ones((2, 2))], match)
