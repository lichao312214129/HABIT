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
"""Tests for the short image-colorbar helper and plotter wiring."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from habit.viz import (
    plot_confusion_matrix,
    plot_intensity_slice,
    plot_msi_matrix,
    plot_voxel_texture_slice,
)
from habit.viz.colorbar import (
    DEFAULT_COLORBAR_SHRINK,
    add_discrete_habitat_colorbar,
    add_image_colorbar,
    add_image_colorbar_from_spec,
    colorbar_is_enabled,
    discrete_habitat_mappable,
)

pytestmark = pytest.mark.unit


def _wide_image() -> np.ndarray:
    """Return a wide 2D array (flat liver-like aspect)."""
    rng = np.random.RandomState(0)
    return rng.normal(loc=80.0, scale=10.0, size=(20, 80)).astype(np.float32)


def _cbar_axes(fig: Figure) -> list:
    """Return axes that are colorbars (no images)."""
    return [ax for ax in fig.axes if not ax.images]


def test_colorbar_is_enabled_false_and_mapping() -> None:
    """False disables; True and mappings (including empty) enable."""
    assert colorbar_is_enabled(False) is False
    assert colorbar_is_enabled(True) is True
    assert colorbar_is_enabled({}) is True
    assert colorbar_is_enabled({"shrink": 0.5}) is True
    assert colorbar_is_enabled({"enabled": False}) is False


def test_add_image_colorbar_is_shorter_than_axes() -> None:
    """Default bar height is clearly shorter than the parent axes."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    image = ax.imshow(_wide_image(), cmap="gray", aspect="equal")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Wide slice")
    ax.axis("off")
    cbar = add_image_colorbar(image, ax=ax, label="Intensity")
    fig.canvas.draw()
    ax_height = ax.get_window_extent().height
    cbar_height = cbar.ax.get_window_extent().height
    assert cbar_height < ax_height * 0.90
    assert cbar.ax.get_ylabel() == "Intensity"
    assert str(cbar.ax.get_ylabel()).isascii()
    plt.close(fig)


def test_add_image_colorbar_custom_shrink_and_label() -> None:
    """Callers can pass shrink / label / ticks."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(constrained_layout=True)
    image = ax.imshow(np.arange(16, dtype=float).reshape(4, 4))
    cbar = add_image_colorbar(
        image, ax=ax, label="Custom", shrink=0.5, ticks=[0.0, 15.0]
    )
    fig.canvas.draw()
    ax_height = ax.get_window_extent().height
    cbar_height = cbar.ax.get_window_extent().height
    assert cbar_height < ax_height * 0.70
    assert cbar.ax.get_ylabel() == "Custom"
    ticks = [float(t) for t in cbar.get_ticks()]
    assert ticks == [0.0, 15.0]
    plt.close(fig)


def test_add_image_colorbar_from_spec_false_skips() -> None:
    """colorbar=False leaves a single image axes."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    image = ax.imshow(np.ones((4, 4)))
    result = add_image_colorbar_from_spec(image, False, ax=ax, label="X")
    assert result is None
    assert len(fig.axes) == 1
    plt.close(fig)


def test_add_image_colorbar_auto_shrink_wide_image() -> None:
    """Auto shrink is the short default; the bar tracks the image box."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
    # Physical extent 4:1 wide, like a flat axial liver FOV.
    image = ax.imshow(
        np.ones((10, 40)),
        extent=(0.0, 400.0, 0.0, 100.0),
        aspect="equal",
    )
    ax.set_aspect("equal", adjustable="box")
    cbar = add_image_colorbar(image, ax=ax, shrink="auto")
    fig.canvas.draw()
    ax_height = ax.get_window_extent().height
    cbar_height = cbar.ax.get_window_extent().height
    assert cbar_height < ax_height * 0.90
    assert getattr(cbar.ax, "_habit_cbar_shrink") == pytest.approx(
        DEFAULT_COLORBAR_SHRINK
    )
    plt.close(fig)


def test_plot_intensity_slice_default_cbar_is_short() -> None:
    """Default intensity colorbar is shorter than the image axes."""
    import matplotlib.pyplot as plt

    image = _wide_image().reshape(1, 20, 80)
    fig = plot_intensity_slice(image, axis=0)
    fig.canvas.draw()
    image_axes = [ax for ax in fig.axes if ax.images]
    cbar_axes = _cbar_axes(fig)
    assert len(image_axes) == 1
    assert len(cbar_axes) == 1
    ax_height = image_axes[0].get_window_extent().height
    cbar_height = cbar_axes[0].get_window_extent().height
    assert cbar_height < ax_height * 0.90
    plt.close(fig)


def test_plot_intensity_slice_colorbar_dict_overrides_label() -> None:
    """A mapping spec customises the intensity colorbar."""
    import matplotlib.pyplot as plt

    image = _wide_image().reshape(1, 20, 80)
    fig = plot_intensity_slice(
        image,
        axis=0,
        colorbar={"label": "Custom intensity", "shrink": 0.5},
    )
    assert isinstance(fig, Figure)
    image_axes = [ax for ax in fig.axes if ax.images]
    cbar_axes = _cbar_axes(fig)
    assert len(image_axes) == 1
    assert len(cbar_axes) == 1
    labels = " ".join(ax.get_ylabel() for ax in cbar_axes)
    assert "Custom intensity" in labels
    plt.close(fig)


def test_plot_voxel_texture_slice_colorbar_false() -> None:
    """colorbar=False keeps only the feature axes."""
    import matplotlib.pyplot as plt

    feature = np.linspace(0.0, 1.0, 8 * 10).reshape(8, 10)
    fig = plot_voxel_texture_slice(feature, mode="feature_only", colorbar=False)
    assert len(fig.axes) == 1
    assert len(fig.axes[0].images) == 1
    plt.close(fig)


def test_plot_voxel_texture_slice_colorbar_kwargs() -> None:
    """A mapping spec sets the voxel-texture colorbar label."""
    import matplotlib.pyplot as plt

    feature = np.linspace(0.0, 1.0, 8 * 10).reshape(8, 10)
    fig = plot_voxel_texture_slice(
        feature,
        mode="feature_only",
        colorbar={"label": "Entropy", "shrink": 0.6},
    )
    cbar_axes = _cbar_axes(fig)
    assert len(cbar_axes) == 1
    assert "Entropy" in cbar_axes[0].get_ylabel()
    plt.close(fig)


def test_plot_msi_matrix_colorbar_false() -> None:
    """MSI heatmap can drop the colorbar."""
    import matplotlib.pyplot as plt

    matrix = np.array([[10.0, 2.0], [2.0, 8.0]], dtype=np.float64)
    fig = plot_msi_matrix(matrix, habitat_ids=(1,), colorbar=False)
    assert len(_cbar_axes(fig)) == 0
    assert len(fig.axes[0].images) == 1
    plt.close(fig)


def test_plot_confusion_matrix_colorbar_false_and_kwargs() -> None:
    """Confusion-matrix colorbar can be hidden or labelled."""
    import matplotlib.pyplot as plt

    y_true = np.array([0, 0, 1, 1], dtype=int)
    y_pred = np.array([0, 1, 1, 1], dtype=int)
    fig_off = plot_confusion_matrix(y_true, y_pred, colorbar=False)
    assert len(_cbar_axes(fig_off)) == 0
    plt.close(fig_off)

    fig = plot_confusion_matrix(
        y_true, y_pred, colorbar={"label": "Count", "shrink": 0.6}
    )
    labels = " ".join(ax.get_ylabel() for ax in _cbar_axes(fig))
    assert "Count" in labels
    plt.close(fig)


def test_discrete_habitat_mappable_skips_background_and_centres_ticks() -> None:
    """Background 0 is omitted; ticks sit on equal colour blocks labelled by ID."""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    colors = ((0.0, 0.45, 0.70), (0.90, 0.60, 0.00), (0.00, 0.62, 0.45))
    mappable, ticks, ticklabels = discrete_habitat_mappable((0, 1, 3, 5), colors)
    assert ticks == [1.0, 2.0, 3.0]
    assert ticklabels == ["1", "3", "5"]
    assert isinstance(mappable.cmap, ListedColormap)
    assert isinstance(mappable.norm, BoundaryNorm)
    assert mappable.cmap.N == 3


def test_add_discrete_habitat_colorbar_false_and_empty() -> None:
    """Disabled spec or no positive IDs skip the bar."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow(np.ones((4, 4)))
    assert add_discrete_habitat_colorbar(ax, (1, 2), ["#0072B2"], colorbar=False) is None
    assert add_discrete_habitat_colorbar(ax, (0, 0), ["#0072B2"]) is None
    assert len(fig.axes) == 1
    plt.close(fig)
