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
"""Tests for display-only ROI/label cropping and the teaching diagrams."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from habit.exceptions import HABITAPIError
from habit.viz import (
    plot_directory_ingest,
    plot_feature_preprocessing_chain,
    plot_habitat_label_compare,
    plot_habitat_overlay,
    plot_intensity_slice,
    plot_numpy_ingest,
    plot_simpleitk_ingest,
    plot_voxel_texture_slice,
)
from habit.viz._crop import bbox_slices, validate_crop_to

pytestmark = pytest.mark.unit


def _synthetic_volume(shape=(24, 40, 44), seed: int = 0):
    """Return (image, labels) with two small off-centre habitat blobs."""
    rng = np.random.RandomState(seed)
    image = rng.normal(loc=100.0, scale=20.0, size=shape).astype(np.float32)
    labels = np.zeros(shape, dtype=np.int32)
    labels[10:14, 12:20, 14:22] = 1
    labels[12:16, 24:30, 26:34] = 2
    return image, labels


def _close(fig: Figure) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


# ---------------------------------------------------------------------------
# _crop helpers
# ---------------------------------------------------------------------------


def test_bbox_slices_cover_foreground_with_pad() -> None:
    """The box wraps foreground voxels plus the pad, clamped at edges."""
    mask = np.zeros((10, 20, 30), dtype=np.int32)
    mask[2:5, 4:8, 10:20] = 1
    slices = bbox_slices(mask, pad=2, caller="test", mask_name="mask")
    assert slices == (slice(0, 7), slice(2, 10), slice(8, 22))


def test_bbox_slices_reject_empty_mask() -> None:
    """An all-background mask has nothing to zoom to."""
    with pytest.raises(HABITAPIError, match="no foreground"):
        bbox_slices(np.zeros((8, 8, 8), dtype=np.int32), 1, caller="test", mask_name="m")


def test_validate_crop_to_rejects_unknown_mode() -> None:
    with pytest.raises(HABITAPIError, match="crop_to"):
        validate_crop_to("everything", allowed=("none", "labels"), caller="test")


# ---------------------------------------------------------------------------
# crop_to on the plotters
# ---------------------------------------------------------------------------


def test_overlay_crop_to_labels_shrinks_drawn_slice() -> None:
    """crop_to='labels' zooms the panel to the habitat bounding box."""
    image, labels = _synthetic_volume()
    fig = plot_habitat_overlay(image, labels, axis=0, crop_to="labels", crop_pad=2)
    assert isinstance(fig, Figure)
    drawn = fig.axes[0].images[0].get_array()
    # Axis-0 slice of the padded union box: y 12..30 + pad -> 22 voxels,
    # x 14..34 + pad -> 24 voxels (orientation may transpose the panel).
    assert sorted(drawn.shape[:2]) == [22, 24]
    _close(fig)


def test_overlay_crop_rejects_bad_mode() -> None:
    image, labels = _synthetic_volume()
    with pytest.raises(HABITAPIError, match="crop_to"):
        plot_habitat_overlay(image, labels, crop_to="roi")
    _close_all()


def _close_all() -> None:
    import matplotlib.pyplot as plt

    plt.close("all")


def test_intensity_crop_requires_roi_mask() -> None:
    image, labels = _synthetic_volume()
    with pytest.raises(HABITAPIError, match="requires roi_mask"):
        plot_intensity_slice(image, crop_to="roi")
    _close_all()


def test_intensity_crop_to_roi_shrinks_drawn_slice() -> None:
    image, labels = _synthetic_volume()
    roi = (labels > 0).astype(np.int32)
    fig = plot_intensity_slice(image, roi_mask=roi, crop_to="roi", crop_pad=2)
    assert isinstance(fig, Figure)
    drawn = fig.axes[0].images[0].get_array()
    assert sorted(drawn.shape[:2]) == [22, 24]
    _close(fig)


def test_voxel_texture_crop_to_roi_shrinks_drawn_slice() -> None:
    image, labels = _synthetic_volume()
    roi = (labels > 0).astype(np.int32)
    feature = np.where(roi > 0, image, np.nan)
    fig = plot_voxel_texture_slice(
        feature, anatomy=image, roi_mask=roi, axis=0, crop_to="roi", crop_pad=2
    )
    assert isinstance(fig, Figure)
    drawn = fig.axes[0].images[0].get_array()
    assert sorted(drawn.shape[:2]) == [22, 24]
    _close(fig)


def test_label_compare_crop_to_labels_shrinks_panels() -> None:
    image, labels = _synthetic_volume()
    moved = np.roll(labels, shift=2, axis=1)
    fig = plot_habitat_label_compare(
        image, labels, moved, align_labels=False, crop_to="labels", crop_pad=2
    )
    assert isinstance(fig, Figure)
    drawn = fig.axes[0].images[0].get_array()
    # Union of both maps along axis 1 grows the box by the 2-voxel shift.
    assert sorted(drawn.shape[:2]) == [24, 24]
    _close(fig)


# ---------------------------------------------------------------------------
# Teaching diagrams
# ---------------------------------------------------------------------------


def test_directory_ingest_returns_ascii_figure() -> None:
    image, labels = _synthetic_volume()
    roi = (labels > 0).astype(np.int32)
    fig = plot_directory_ingest(image, roi)
    assert isinstance(fig, Figure)
    fig.canvas.draw()
    texts = []
    for ax in fig.axes:
        texts.extend(text.get_text() for text in ax.texts)
        if ax.get_title():
            texts.append(ax.get_title())
    assert texts and all(text.isascii() for text in texts)
    assert any("DATA/" in text for text in texts)
    _close(fig)


def test_simpleitk_ingest_returns_figure_with_badge_caption() -> None:
    image, labels = _synthetic_volume()
    fig = plot_simpleitk_ingest(image, labels)
    assert isinstance(fig, Figure)
    fig.canvas.draw()
    texts = [
        text.get_text()
        for ax in fig.axes
        for text in ax.texts
    ]
    assert any("SimpleITK" in text for text in texts)
    _close(fig)


def test_numpy_ingest_requires_foreground() -> None:
    image, _ = _synthetic_volume()
    with pytest.raises(HABITAPIError, match="roi_mask or labels"):
        plot_numpy_ingest(image)
    _close_all()


def test_numpy_ingest_with_roi_returns_figure() -> None:
    image, labels = _synthetic_volume()
    roi = (labels > 0).astype(np.int32)
    fig = plot_numpy_ingest(image, roi_mask=roi)
    assert isinstance(fig, Figure)
    _close(fig)


def test_feature_chain_validates_arguments() -> None:
    image, labels = _synthetic_volume()
    with pytest.raises(HABITAPIError, match="must not be empty"):
        plot_feature_preprocessing_chain(())
    with pytest.raises(HABITAPIError, match="given together"):
        plot_feature_preprocessing_chain(["raw"], image=image)
    _close_all()


def test_feature_chain_with_data_panel_returns_figure() -> None:
    image, labels = _synthetic_volume()
    fig = plot_feature_preprocessing_chain(
        ("raw", "winsorize", "minmax"), image=image, labels=labels
    )
    assert isinstance(fig, Figure)
    fig.canvas.draw()
    texts = [
        text.get_text()
        for ax in fig.axes
        for text in ax.texts
    ]
    assert any("winsorize" in text for text in texts)
    _close(fig)
