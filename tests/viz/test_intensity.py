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
"""Tests for greyscale intensity-slice figures in ``habit.viz``."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from habit.exceptions import HABITAPIError
from habit.viz import plot_intensity_slice

pytestmark = pytest.mark.unit


def _synthetic_volume(shape=(12, 16, 14), seed: int = 0):
    """Return (image, mask) with a central blob of higher intensity."""
    rng = np.random.RandomState(seed)
    image = rng.normal(loc=80.0, scale=10.0, size=shape).astype(np.float32)
    image[4:9, 5:12, 4:11] += 40.0
    mask = np.zeros(shape, dtype=np.uint8)
    mask[5:8, 6:11, 5:10] = 1
    return image, mask


def test_plot_intensity_slice_single_panel_is_gray() -> None:
    """One volume draws a single greyscale panel (no ROI crop)."""
    image, _mask = _synthetic_volume()
    fig = plot_intensity_slice(image, axis=0, image_label="Resampled T1")
    assert isinstance(fig, Figure)
    image_axes = [ax for ax in fig.axes if ax.images]
    assert len(image_axes) == 1
    cmap_name = image_axes[0].images[0].cmap.name
    assert cmap_name == "gray"
    texts = [image_axes[0].get_title()]
    if fig._suptitle is not None:
        texts.append(fig._suptitle.get_text())
    joined = " ".join(texts)
    assert joined.isascii(), joined
    assert "ROI" not in joined

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_intensity_slice_picks_slice_from_before_anatomy() -> None:
    """Z-score |z| of air must not steal the plane; original anatomy does."""
    image = np.zeros((6, 12, 10), dtype=np.float32)
    image[3, :, :] = 80.0
    processed = (image - float(image.mean())) / (float(image.std()) + 1e-6)
    fig = plot_intensity_slice(
        processed,
        before=image,
        axis=0,
        image_label="Z-scored",
        before_label="Original",
    )
    image_axes = [ax for ax in fig.axes if ax.images]
    displayed = np.asarray(image_axes[0].images[0].get_array())
    # Plane 3 is filled anatomy; air planes are 0. After z-score those
    # zeros become a non-zero |z| and would win mass-based selection.
    assert float(np.median(displayed)) > 40.0

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_intensity_slice_before_after_two_panels() -> None:
    """Matching grids yield original | processed, both greyscale."""
    image, _mask = _synthetic_volume()
    processed = (image - float(image.mean())) / (float(image.std()) + 1e-6)
    fig = plot_intensity_slice(
        processed,
        before=image,
        axis=0,
        image_label="Z-scored T1",
        title="Original | Z-scored T1",
        colorbar_label="Z-score",
        before_colorbar_label="Intensity",
    )
    image_axes = [ax for ax in fig.axes if ax.images]
    assert len(image_axes) == 2
    for ax in image_axes:
        assert ax.images[0].cmap.name == "gray"
        displayed = np.asarray(ax.images[0].get_array())
        assert displayed.shape == (16, 14) or displayed.ndim == 2
        # Full FOV: not a masked-to-ROI island.
        assert not np.ma.is_masked(displayed) or np.mean(np.ma.getmaskarray(displayed)) < 0.05
    joined = " ".join(ax.get_title() for ax in image_axes)
    assert "ROI" not in joined
    assert "Z-scored" in joined

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_intensity_slice_independent_colorbars_native_units() -> None:
    """Z-score before/after colorbars must show native units, not [0, 1]."""
    image, _mask = _synthetic_volume()
    processed = (image - float(image.mean())) / (float(image.std()) + 1e-6)
    fig = plot_intensity_slice(
        processed,
        before=image,
        axis=0,
        image_label="Z-scored T1",
        colorbar_label="Z-score",
        before_colorbar_label="Intensity",
    )
    image_axes = [ax for ax in fig.axes if ax.images]
    assert len(image_axes) == 2
    before_clim = image_axes[0].images[0].get_clim()
    after_clim = image_axes[1].images[0].get_clim()
    # Original synthetic intensities are around loc=80, not a unit interval.
    assert before_clim[0] > 20.0
    assert before_clim[1] > before_clim[0]
    # Z-score window sits near N(0, 1); must not reuse the raw clim.
    assert after_clim[1] < 10.0
    assert after_clim[0] < 0.0 < after_clim[1]
    assert abs(before_clim[1] - after_clim[1]) > 10.0
    displayed_after = np.asarray(image_axes[1].images[0].get_array())
    assert abs(float(np.nanmean(displayed_after))) < 5.0
    cbar_axes = [ax for ax in fig.axes if ax not in image_axes]
    assert len(cbar_axes) == 2
    labels = " ".join(ax.get_ylabel() for ax in cbar_axes)
    assert "Intensity" in labels
    assert "Z-score" in labels
    assert labels.isascii(), labels

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_intensity_slice_windows_past_bright_tail() -> None:
    """Air + dim tissue + rare hot voxels must not set vmax to the hot tail."""
    rng = np.random.RandomState(1)
    image = np.zeros((8, 80, 80), dtype=np.float32)
    image[:, 15:65, 15:65] = rng.normal(100.0, 12.0, size=(8, 50, 50))
    image[:, 36:40, 36:40] = 2500.0
    fig = plot_intensity_slice(image, axis=0, index=4)
    image_axes = [ax for ax in fig.axes if ax.images]
    vmin, vmax = image_axes[0].images[0].get_clim()
    assert vmax < 400.0
    assert vmin < 90.0
    assert vmax > 80.0
    assert vmax > vmin

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_intensity_slice_symmetric_clim_is_signed() -> None:
    """symmetric_clim windows the processed panel about zero."""
    image, _mask = _synthetic_volume()
    processed = (image - float(image.mean())) / (float(image.std()) + 1e-6)
    fig = plot_intensity_slice(
        processed,
        before=image,
        axis=0,
        cmap="RdBu_r",
        before_cmap="gray",
        symmetric_clim=True,
        colorbar_label="Z-score",
    )
    image_axes = [ax for ax in fig.axes if ax.images]
    after_clim = image_axes[1].images[0].get_clim()
    assert after_clim[0] < 0.0 < after_clim[1]
    assert abs(after_clim[0] + after_clim[1]) < 1e-6

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_intensity_slice_colorbar_false_has_no_cbar_axes() -> None:
    """colorbar=False stays a single image axes."""
    image, _mask = _synthetic_volume()
    fig = plot_intensity_slice(image, axis=0, colorbar=False)
    image_axes = [ax for ax in fig.axes if ax.images]
    assert len(image_axes) == 1
    assert len(fig.axes) == 1

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_intensity_slice_rejects_shape_mismatch() -> None:
    """before= must share the processed grid."""
    image, _mask = _synthetic_volume()
    with pytest.raises(HABITAPIError, match="same shape"):
        plot_intensity_slice(image, before=image[:-1], axis=0)


def test_plot_intensity_slice_roi_contour_does_not_crop() -> None:
    """Contour overlay keeps outside-ROI voxels visible."""
    image, mask = _synthetic_volume()
    fig = plot_intensity_slice(
        image,
        axis=0,
        index=6,
        roi_mask=mask,
        roi_contour=True,
        image_label="Registered T1",
    )
    image_axes = [ax for ax in fig.axes if ax.images]
    assert len(image_axes) == 1
    displayed = np.asarray(image_axes[0].images[0].get_array())
    assert displayed.shape[0] == 16
    assert displayed.shape[1] == 14
    # Anatomy outside the small blob must still be drawn (not white-masked).
    assert float(np.mean(displayed)) > 0.05

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_intensity_slice_never_calls_show(monkeypatch) -> None:
    """The viz function stays pure: no display hooks."""
    import matplotlib.pyplot as plt

    calls = {"show": 0}
    original_show = plt.show

    def counting_show(*args, **kwargs):
        calls["show"] += 1
        return original_show(*args, **kwargs)

    monkeypatch.setattr(plt, "show", counting_show)
    image, _mask = _synthetic_volume()
    plot_intensity_slice(image, axis=0)
    assert calls["show"] == 0
    plt.close("all")
