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
"""Tests for habitat-on-image overlay figures in ``habit.viz``."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from habit.exceptions import HABITAPIError
from habit.viz import plot_habitat_overlay

pytestmark = pytest.mark.unit


def _synthetic_volume(shape=(12, 16, 14), seed: int = 0):
    """Return (image, labels) with three habitats inside a blob."""
    rng = np.random.RandomState(seed)
    image = rng.normal(loc=100.0, scale=20.0, size=shape).astype(np.float32)
    labels = np.zeros(shape, dtype=np.int32)
    labels[3:9, 4:12, 3:11] = 1
    labels[5:8, 6:10, 5:9] = 2
    labels[6:7, 7:9, 6:8] = 3
    return image, labels


def test_plot_habitat_overlay_3d_returns_three_panel_figure() -> None:
    """Default 3D path draws three orthogonal mid-slice overlays."""
    image, labels = _synthetic_volume()
    fig = plot_habitat_overlay(image, labels, title="Habitat overlay demo")
    assert isinstance(fig, Figure)
    image_axes = [ax for ax in fig.axes if ax.images]
    cbar_axes = [ax for ax in fig.axes if not ax.images]
    assert len(image_axes) == 3
    assert len(cbar_axes) == 1
    fig.canvas.draw()
    assert cbar_axes[0].get_ylabel() == "Habitat"
    # Discrete bar: one centred tick per habitat ID (1, 2, 3), not a smear.
    tick_labels = [text.get_text() for text in cbar_axes[0].get_yticklabels()]
    assert "1" in tick_labels and "2" in tick_labels and "3" in tick_labels

    texts = [fig._suptitle.get_text()] if fig._suptitle is not None else []
    for ax in image_axes:
        texts.append(ax.get_title())
    joined = " ".join(texts)
    assert joined.isascii(), joined
    assert "Habitat" in joined

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_habitat_overlay_single_axis() -> None:
    """Passing axis=0 yields a single-panel figure."""
    image, labels = _synthetic_volume()
    fig = plot_habitat_overlay(image, labels, axis=0, index=6)
    assert isinstance(fig, Figure)
    image_axes = [ax for ax in fig.axes if ax.images]
    cbar_axes = [ax for ax in fig.axes if not ax.images]
    assert len(image_axes) == 1
    assert len(cbar_axes) == 1

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_habitat_overlay_rejects_shape_mismatch() -> None:
    """Image and labels must share a shape."""
    image, labels = _synthetic_volume()
    with pytest.raises(HABITAPIError, match="same shape"):
        plot_habitat_overlay(image, labels[:-1])


def test_plot_habitat_overlay_never_calls_show(monkeypatch) -> None:
    """The viz function stays pure: no display hooks."""
    import matplotlib.pyplot as plt

    calls = {"show": 0}
    original_show = plt.show

    def counting_show(*args, **kwargs):
        calls["show"] += 1
        return original_show(*args, **kwargs)

    monkeypatch.setattr(plt, "show", counting_show)
    image, labels = _synthetic_volume()
    plot_habitat_overlay(image, labels)
    assert calls["show"] == 0


def test_plot_habitat_overlay_auto_slice_follows_habitat_mass() -> None:
    """Default slice picks the axis plane with the most habitat voxels."""
    image = np.zeros((20, 20, 20), dtype=np.float32)
    image[:] = 50.0
    labels = np.zeros((20, 20, 20), dtype=np.int32)
    # Off-centre: slice 4 on axis 0 has the largest habitat footprint.
    labels[3, 14:16, 2:4] = 1
    labels[4, 14:18, 2:5] = 1
    labels[5, 14:16, 2:4] = 1
    fig = plot_habitat_overlay(image, labels, axis=0)
    assert "index=4" in fig.axes[0].get_title()

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_habitat_overlay_ras_axial_puts_anterior_up() -> None:
    """RAS (z,y,x) axial panel: high-y (anterior) lands in the top image half."""
    from habit.viz.habitat_overlay import _orient_slice_for_display

    # RAS direction matching demo NIfTI via SimpleITK.
    ras = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
    direction = np.asarray(ras, dtype=np.float64).reshape(3, 3)
    # Rows = y (anterior↑ index), cols = x (right↑ index).
    slice_2d = np.zeros((10, 10), dtype=np.float32)
    slice_2d[-1, 5] = 1.0  # max y = anterior marker
    oriented = _orient_slice_for_display(slice_2d, slice_axis=0, direction=direction)
    row, col = np.argwhere(oriented == 1.0)[0]
    assert row < 5, "anterior marker should be in the upper half after orient"
    # Patient right (max x under RAS) should move to the viewer's left.
    slice_lr = np.zeros((10, 10), dtype=np.float32)
    slice_lr[5, -1] = 1.0
    oriented_lr = _orient_slice_for_display(slice_lr, slice_axis=0, direction=direction)
    _, col_r = np.argwhere(oriented_lr == 1.0)[0]
    assert col_r < 5, "patient-right marker should be on the viewer's left"


def test_plot_habitat_overlay_ras_coronal_and_sagittal_flips() -> None:
    """RAS coronal/sagittal: superior up; coronal R-left; sagittal A-left."""
    from habit.viz.habitat_overlay import _orient_slice_for_display

    ras = np.asarray(
        (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0),
        dtype=np.float64,
    ).reshape(3, 3)

    # Coronal-like (axis 1): rows=z superior↑, cols=x right↑
    coronal = np.zeros((10, 10), dtype=np.float32)
    coronal[-1, 5] = 1.0  # superior
    out = _orient_slice_for_display(coronal, slice_axis=1, direction=ras)
    assert np.argwhere(out == 1.0)[0, 0] < 5
    coronal_r = np.zeros((10, 10), dtype=np.float32)
    coronal_r[5, -1] = 1.0  # patient right
    out_r = _orient_slice_for_display(coronal_r, slice_axis=1, direction=ras)
    assert np.argwhere(out_r == 1.0)[0, 1] < 5

    # Sagittal-like (axis 2): rows=z superior↑, cols=y anterior↑
    sagittal = np.zeros((10, 10), dtype=np.float32)
    sagittal[-1, 5] = 1.0  # superior
    out_s = _orient_slice_for_display(sagittal, slice_axis=2, direction=ras)
    assert np.argwhere(out_s == 1.0)[0, 0] < 5
    sagittal_a = np.zeros((10, 10), dtype=np.float32)
    sagittal_a[5, -1] = 1.0  # anterior
    out_a = _orient_slice_for_display(sagittal_a, slice_axis=2, direction=ras)
    assert np.argwhere(out_a == 1.0)[0, 1] < 5


def test_plot_habitat_overlay_lps_identity_keeps_axial_unflipped() -> None:
    """LPS identity axial: no in-plane AP/LR flip (coronal/sagittal still flip S-I)."""
    from habit.viz.habitat_overlay import _orient_slice_for_display
    from habit.viz.orientation import radiological_array_axis_flips

    lps = np.eye(3, dtype=np.float64)
    slice_2d = np.arange(100, dtype=np.float32).reshape(10, 10)
    oriented = _orient_slice_for_display(slice_2d, slice_axis=0, direction=lps)
    np.testing.assert_array_equal(oriented, slice_2d)
    # Full-volume radiological remap still flips z for orthogonal S-up panels.
    assert radiological_array_axis_flips(lps) == (True, False, False)


def test_imshow_aspect_uses_physical_spacing() -> None:
    """Thick z-spacing stretches coronal/sagittal (aspect = row/col spacing)."""
    from habit.viz.habitat_overlay import _imshow_aspect, _imshow_physical_extent

    # SimpleITK (x, y, z) = (1, 1, 5) mm — thick slices.
    spacing = (1.0, 1.0, 5.0)
    assert _imshow_aspect(spacing, slice_axis=0, ndim=3) == pytest.approx(1.0)
    assert _imshow_aspect(spacing, slice_axis=1, ndim=3) == pytest.approx(5.0)
    assert _imshow_aspect(spacing, slice_axis=2, ndim=3) == pytest.approx(5.0)

    # Coronal-like plane (n_z=30, n_x=40): physical height must exceed width
    # when z-spacing is thick — extent top-bottom spans n_z * 5 mm.
    left, right, bottom, top = _imshow_physical_extent(
        (30, 40), spacing, slice_axis=1, ndim=3
    )
    assert right - left == pytest.approx(40.0)  # 40 * 1 mm
    assert top - bottom == pytest.approx(150.0)  # 30 * 5 mm
    assert bottom < top, "non-inverted ylim so aspect='equal' cannot flip S-I"
    assert (top - bottom) / (right - left) == pytest.approx(150.0 / 40.0)


def test_plot_habitat_overlay_anisotropic_spacing_sets_panel_aspects() -> None:
    """3D panels use mm extent + equal aspect; thick axis is longer on screen."""
    image, labels = _synthetic_volume(shape=(8, 16, 16))
    # Thick slices: z=4 mm, in-plane 1 mm.
    spacing = (1.0, 1.0, 4.0)
    ras = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
    fig = plot_habitat_overlay(
        image, labels, spacing=spacing, direction=ras, title="Anisotropic"
    )
    image_axes = [ax for ax in fig.axes if ax.images]
    assert len(image_axes) == 3
    fig.canvas.draw()

    for axis_id, ax in enumerate(image_axes):
        # Matplotlib reports aspect='equal' as the numeric value 1.0.
        assert float(ax.get_aspect()) == pytest.approx(1.0)
        images = ax.get_images()
        assert images, f"panel {axis_id} missing AxesImage"
        left, right, bottom, top = images[0].get_extent()
        phys_w = abs(right - left)
        phys_h = abs(bottom - top)
        bb = ax.get_window_extent()
        # Screen box must match physical FOV ratio (thick z → taller coronal/sag).
        assert bb.height / bb.width == pytest.approx(phys_h / phys_w, rel=1e-3)

    # Coronal / sagittal FOV height = n_z * 4 mm; width = 16 * 1 mm.
    cor_ext = fig.axes[1].get_images()[0].get_extent()
    assert abs(cor_ext[2] - cor_ext[3]) == pytest.approx(8 * 4.0)
    assert abs(cor_ext[1] - cor_ext[0]) == pytest.approx(16 * 1.0)

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_habitat_overlay_rejects_bad_spacing() -> None:
    """Spacing must match ndim and be positive."""
    image, labels = _synthetic_volume()
    with pytest.raises(HABITAPIError, match="spacing"):
        plot_habitat_overlay(image, labels, spacing=(1.0, 1.0))
    with pytest.raises(HABITAPIError, match="spacing"):
        plot_habitat_overlay(image, labels, spacing=(1.0, 1.0, 0.0))


def _marker_frac_from_bottom(
    ax, row: int, col: int, nrows: int, ncols: int
) -> float:
    """Fraction of axes height from the bottom for an array (row, col) marker."""
    left, right, bottom, top = ax.get_images()[0].get_extent()
    x = left + (col + 0.5) / ncols * (right - left)
    # origin='upper': row 0 sits at ``top`` in data coordinates.
    y = top + (row + 0.5) / nrows * (bottom - top)
    disp_x, disp_y = ax.transData.transform((x, y))
    bbox = ax.get_window_extent()
    return float((disp_y - bbox.y0) / bbox.height)


def test_coronal_sagittal_superior_stays_upper_half_on_screen() -> None:
    """After canvas.draw(), superior markers must sit in the upper axes half.

    This catches the matplotlib ``extent`` + ``aspect='equal'`` trap that
    flipped coronal/sagittal up-down on screen while axial still looked
    right (array-only tests cannot see that).
    """
    import matplotlib.pyplot as plt

    ras = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
    # Thick slices so coronal/sagittal panels are tall (the failure mode).
    spacing = (1.0, 1.0, 5.0)
    image = np.zeros((20, 16, 16), dtype=np.float32)
    image[:] = 50.0
    labels = np.zeros((20, 16, 16), dtype=np.int32)
    # Superior-most slab (max z) is habitat 1 — should be TOP of coronal/sag.
    labels[-1, 6:10, 6:10] = 1
    # Inferior-most slab is habitat 2 — should be BOTTOM.
    labels[0, 6:10, 6:10] = 2

    fig = plot_habitat_overlay(
        image, labels, direction=ras, spacing=spacing, title="S-I screen test"
    )
    fig.canvas.draw()

    # Panel 1 = coronal-like (axis 1): rows = z after orient (superior → row 0).
    coronal_ax = fig.axes[1]
    # After orient_slice, superior marker is in the upper array half; confirm
    # it is also in the upper *screen* half (frac_from_bottom > 0.5).
    from habit.viz.habitat_overlay import _orient_slice_for_display

    direction = np.asarray(ras, dtype=np.float64).reshape(3, 3)
    mid_y = labels.shape[1] // 2
    coronal_raw = labels[:, mid_y, :]
    oriented = _orient_slice_for_display(
        coronal_raw.astype(np.float32), slice_axis=1, direction=direction
    )
    sup_row, sup_col = np.argwhere(oriented == 1)[0]
    inf_row, inf_col = np.argwhere(oriented == 2)[0]
    nrows, ncols = oriented.shape
    sup_frac = _marker_frac_from_bottom(
        coronal_ax, int(sup_row), int(sup_col), nrows, ncols
    )
    inf_frac = _marker_frac_from_bottom(
        coronal_ax, int(inf_row), int(inf_col), nrows, ncols
    )
    assert sup_frac > 0.5, f"superior should be upper half; frac={sup_frac:.3f}"
    assert inf_frac < 0.5, f"inferior should be lower half; frac={inf_frac:.3f}"
    assert sup_frac > inf_frac

    sagittal_ax = fig.axes[2]
    mid_x = labels.shape[2] // 2
    sag_raw = labels[:, :, mid_x]
    sag_oriented = _orient_slice_for_display(
        sag_raw.astype(np.float32), slice_axis=2, direction=direction
    )
    s_row, s_col = np.argwhere(sag_oriented == 1)[0]
    i_row, i_col = np.argwhere(sag_oriented == 2)[0]
    s_frac = _marker_frac_from_bottom(
        sagittal_ax, int(s_row), int(s_col), *sag_oriented.shape
    )
    i_frac = _marker_frac_from_bottom(
        sagittal_ax, int(i_row), int(i_col), *sag_oriented.shape
    )
    assert s_frac > 0.5 and i_frac < 0.5
    plt.close(fig)


def test_plot_habitat_overlay_reads_geometry_from_image_volume() -> None:
    """Passing ImageVolume (not .data) must pick up RAS direction automatically."""
    from habit.api.image import ImageVolume

    ras = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
    image = np.zeros((8, 10, 10), dtype=np.float32)
    labels = np.zeros((8, 10, 10), dtype=np.int32)
    labels[4, 8, 5] = 1  # high y = anterior under RAS
    volume = ImageVolume.from_array(
        image, spacing=(1.0, 1.0, 1.0), direction=ras
    )
    fig = plot_habitat_overlay(volume, labels, axis=0, index=4)
    # Axial RAS: anterior (max y) must land in the upper array half after orient.
    from habit.viz.habitat_overlay import _orient_slice_for_display

    direction = np.asarray(ras, dtype=np.float64).reshape(3, 3)
    oriented = _orient_slice_for_display(
        labels[4].astype(np.float32), slice_axis=0, direction=direction
    )
    row, _ = np.argwhere(oriented == 1.0)[0]
    assert row < 5
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_lps_coronal_flips_superior_to_top() -> None:
    """LPS identity coronal: +z is superior, so flipud puts superior at row 0."""
    from habit.viz.habitat_overlay import _orient_slice_for_display

    lps = np.eye(3, dtype=np.float64)
    coronal = np.zeros((10, 10), dtype=np.float32)
    coronal[-1, 5] = 1.0  # max z = superior
    out = _orient_slice_for_display(coronal, slice_axis=1, direction=lps)
    assert np.argwhere(out == 1.0)[0, 0] < 5


def test_plot_habitat_overlay_colorbar_false_skips_bar() -> None:
    """colorbar=False keeps only the image panel(s)."""
    import matplotlib.pyplot as plt

    image, labels = _synthetic_volume()
    fig = plot_habitat_overlay(image, labels, axis=0, colorbar=False)
    assert len([ax for ax in fig.axes if ax.images]) == 1
    assert len([ax for ax in fig.axes if not ax.images]) == 0
    plt.close(fig)
