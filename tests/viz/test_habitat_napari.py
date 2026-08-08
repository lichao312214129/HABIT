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
"""Unit tests for the optional napari habitat viewer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import numpy as np
import pytest

from habit.exceptions import HABITAPIError, OptionalDependencyError
from habit.viz.habitat_napari import napari_radiological_flips, view_habitat_napari
from habit.viz.orientation import (
    DEFAULT_RAS_DIRECTION,
    apply_radiological_flips,
    orient_slice_for_display,
)

pytestmark = pytest.mark.unit

#: SimpleITK RAS direction (demo NIfTI via sitk) — same as matplotlib overlay tests.
_RAS = DEFAULT_RAS_DIRECTION


class _FakeImageLayer:
    def __init__(
        self,
        data: np.ndarray,
        *,
        name: str,
        colormap: str,
        scale: Any = None,
    ) -> None:
        self.data = data
        self.name = name
        self.colormap = colormap
        self.scale = scale


class _FakeLabelsLayer:
    def __init__(
        self,
        data: np.ndarray,
        *,
        name: str,
        opacity: float,
        scale: Any = None,
    ) -> None:
        self.data = data
        self.name = name
        self.opacity = opacity
        self.scale = scale


class _FakeViewer:
    """Minimal stand-in for napari.Viewer used when napari is unavailable."""

    def __init__(self, *, title: str = "", show: bool = True) -> None:
        self.title = title
        self.show_flag = show
        self.layers: List[Any] = []
        self.closed = False

    def add_image(
        self,
        data: np.ndarray,
        *,
        name: str,
        colormap: str,
        scale: Any = None,
    ) -> _FakeImageLayer:
        layer = _FakeImageLayer(
            data, name=name, colormap=colormap, scale=scale
        )
        self.layers.append(layer)
        return layer

    def add_labels(
        self,
        data: np.ndarray,
        *,
        name: str,
        opacity: float,
        scale: Any = None,
    ) -> _FakeLabelsLayer:
        layer = _FakeLabelsLayer(
            data, name=name, opacity=opacity, scale=scale
        )
        self.layers.append(layer)
        return layer

    def close(self) -> None:
        self.closed = True


def test_napari_radiological_flips_ras_in_plane_preserves_z() -> None:
    """RAS: napari flips y/x for radiological A-P/L-R but keeps axial index."""
    assert napari_radiological_flips(_RAS, ndim=3) == (False, True, True)
    # Omitted direction defaults to LPS identity (same as ImageVolume / overlay).
    assert napari_radiological_flips(None, ndim=3) == (False, False, False)
    # LPS identity: no in-plane flips; z also preserved (slider = file order).
    # Full-volume radiological remap would flip z; napari opts out by default.
    lps = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    assert napari_radiological_flips(lps, ndim=3) == (False, False, False)
    assert napari_radiological_flips(
        lps, ndim=3, preserve_axial_index=False
    ) == (True, False, False)
    assert napari_radiological_flips(None, ndim=2) == (False, False)


def test_napari_ras_volume_flip_matches_overlay_axial() -> None:
    """In-plane RAS flips match per-slice matplotlib orient on an axial plane."""
    ras_matrix = np.asarray(_RAS, dtype=np.float64).reshape(3, 3)
    volume = np.zeros((4, 10, 10), dtype=np.float32)
    volume[2, -1, 5] = 1.0  # anterior marker (max y)
    volume[2, 5, -1] = 2.0  # patient-right marker (max x)

    flips = napari_radiological_flips(_RAS, ndim=3)
    assert flips == (False, True, True)
    # AP/LR come from y/x flips (same as overlay); z index unchanged.
    axial = apply_radiological_flips(volume[2], (flips[1], flips[2]))

    expected = orient_slice_for_display(
        volume[2], slice_axis=0, direction=ras_matrix
    )
    np.testing.assert_array_equal(axial, expected)
    row_a, _ = np.argwhere(axial == 1.0)[0]
    assert row_a < 5, "anterior marker should be in the upper half"
    _, col_r = np.argwhere(axial == 2.0)[0]
    assert col_r < 5, "patient-right marker should be on the viewer's left"


def test_view_habitat_napari_applies_ras_flips_to_layers(monkeypatch) -> None:
    """Image and labels layers receive the same radiological flip for RAS."""
    fake_napari = SimpleNamespace(Viewer=_FakeViewer)
    monkeypatch.setattr(
        "habit.viz.habitat_napari.require",
        lambda module, *, extra, purpose, **kwargs: fake_napari,
    )

    image = np.arange(4 * 6 * 6, dtype=np.float32).reshape(4, 6, 6)
    labels = np.zeros((4, 6, 6), dtype=np.int32)
    labels[1, -1, -1] = 3

    viewer = view_habitat_napari(
        image,
        labels,
        opacity=0.4,
        show=False,
        direction=_RAS,
    )
    assert viewer._habit_radiological_flips == (False, True, True)
    expected_image = np.ascontiguousarray(
        apply_radiological_flips(image, (False, True, True)), dtype=np.float32
    )
    expected_labels = np.ascontiguousarray(
        apply_radiological_flips(labels, (False, True, True)), dtype=np.int32
    )
    np.testing.assert_array_equal(viewer.layers[0].data, expected_image)
    np.testing.assert_array_equal(viewer.layers[1].data, expected_labels)
    # z preserved; flip y/x only: label at [1,-1,-1] → [1,0,0] for shape (4,6,6).
    assert viewer.layers[1].data[1, 0, 0] == 3


def test_view_habitat_napari_builds_image_and_labels_layers(monkeypatch) -> None:
    """Array API adds greyscale image + habitat labels with the requested opacity."""
    fake_napari = SimpleNamespace(Viewer=_FakeViewer)
    monkeypatch.setattr(
        "habit.viz.habitat_napari.require",
        lambda module, *, extra, purpose, **kwargs: fake_napari,
    )

    image = np.ones((4, 6, 6), dtype=np.float32)
    labels = np.zeros((4, 6, 6), dtype=np.int32)
    labels[1:3, 2:5, 2:5] = 2

    # LPS identity + preserve axial index: no volume flips; data order unchanged.
    lps = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    viewer = view_habitat_napari(
        image, labels, opacity=0.4, show=False, title="demo", direction=lps
    )
    assert isinstance(viewer, _FakeViewer)
    assert viewer.show_flag is False
    assert [layer.name for layer in viewer.layers] == ["image", "habitats"]
    assert viewer.layers[1].opacity == pytest.approx(0.4)
    assert viewer.layers[1].data.dtype == np.int32
    assert viewer._habit_radiological_flips == (False, False, False)
    np.testing.assert_array_equal(viewer.layers[0].data, image)
    np.testing.assert_array_equal(viewer.layers[1].data, labels)


def test_view_habitat_napari_show_true_calls_napari_run(monkeypatch) -> None:
    """show=True must call napari.run() so scripts do not flash-close."""
    run_calls: List[dict] = []

    def _fake_run(**kwargs: Any) -> None:
        run_calls.append(dict(kwargs))

    fake_napari = SimpleNamespace(Viewer=_FakeViewer, run=_fake_run)
    monkeypatch.setattr(
        "habit.viz.habitat_napari.require",
        lambda module, *, extra, purpose, **kwargs: fake_napari,
    )

    image = np.ones((3, 4, 4), dtype=np.float32)
    labels = np.zeros((3, 4, 4), dtype=np.int32)
    labels[1, 1:3, 1:3] = 1
    lps = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    viewer = view_habitat_napari(image, labels, show=True, direction=lps)
    assert isinstance(viewer, _FakeViewer)
    assert viewer.show_flag is True
    assert len(run_calls) == 1


def test_view_habitat_napari_show_false_skips_napari_run(monkeypatch) -> None:
    """show=False must not start the Qt event loop."""
    run_calls: List[int] = []

    fake_napari = SimpleNamespace(
        Viewer=_FakeViewer,
        run=lambda **kwargs: run_calls.append(1),
    )
    monkeypatch.setattr(
        "habit.viz.habitat_napari.require",
        lambda module, *, extra, purpose, **kwargs: fake_napari,
    )

    image = np.ones((2, 3, 3), dtype=np.float32)
    labels = np.zeros((2, 3, 3), dtype=np.int32)
    viewer = view_habitat_napari(image, labels, show=False)
    assert isinstance(viewer, _FakeViewer)
    assert run_calls == []


def test_view_habitat_napari_multi_image_layers(monkeypatch) -> None:
    """Sequence of images adds one image layer each plus a single labels layer."""
    fake_napari = SimpleNamespace(Viewer=_FakeViewer)
    monkeypatch.setattr(
        "habit.viz.habitat_napari.require",
        lambda module, *, extra, purpose, **kwargs: fake_napari,
    )

    shape = (3, 5, 5)
    img_a = np.ones(shape, dtype=np.float32)
    img_b = np.full(shape, 2.0, dtype=np.float32)
    labels = np.zeros(shape, dtype=np.int32)
    labels[1, 1:4, 1:4] = 1

    lps = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    viewer = view_habitat_napari(
        [img_a, img_b],
        labels,
        opacity=0.5,
        show=False,
        image_names=["T1", "T2"],
        spacing=(1.0, 1.0, 3.0),  # SimpleITK (x, y, z)
        direction=lps,
    )
    assert [layer.name for layer in viewer.layers] == ["T1", "T2", "habitats"]
    # Napari scale is array order (z, y, x).
    assert viewer.layers[0].scale == (3.0, 1.0, 1.0)
    assert viewer.layers[2].scale == (3.0, 1.0, 1.0)
    assert viewer.layers[2].opacity == pytest.approx(0.5)


def test_view_habitat_napari_rejects_shape_mismatch(monkeypatch) -> None:
    """Mismatched image / label shapes raise HABITAPIError before opening Qt."""
    fake_napari = SimpleNamespace(Viewer=_FakeViewer)
    monkeypatch.setattr(
        "habit.viz.habitat_napari.require",
        lambda module, *, extra, purpose, **kwargs: fake_napari,
    )
    with pytest.raises(HABITAPIError, match="same shape"):
        view_habitat_napari(
            np.zeros((4, 4, 4), dtype=np.float32),
            np.zeros((3, 4, 4), dtype=np.int32),
            show=False,
        )


def test_view_habitat_napari_missing_extra_message(monkeypatch) -> None:
    """Missing napari surfaces OptionalDependencyError with the [view] hint."""

    def _boom(module: str, *, extra: str, purpose: str, **kwargs: Any):
        raise OptionalDependencyError(
            f"{module} is required for {purpose}, but it is not installed.\n"
            f'pip install "habitat-analysis[{extra}]"'
        )

    monkeypatch.setattr("habit.viz.habitat_napari.require", _boom)
    with pytest.raises(OptionalDependencyError, match=r"habitat-analysis\[view\]"):
        view_habitat_napari(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.int32),
            show=False,
        )


def test_view_habitat_napari_incomplete_install_message(monkeypatch) -> None:
    """Namespace-only napari (no Viewer) raises OptionalDependencyError, not AttributeError."""

    def _stub(module: str, *, extra: str, purpose: str, **kwargs: Any):
        # Mimic a broken uninstall: importable module, empty public API,
        # no __file__ (namespace package leftover).
        return SimpleNamespace(__file__=None)

    monkeypatch.setattr("habit.viz.habitat_napari.require", _stub)
    with pytest.raises(OptionalDependencyError, match=r"incomplete|no Viewer"):
        view_habitat_napari(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.int32),
            show=False,
        )


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["find_spec"]).find_spec("napari") is None,
    reason="napari optional extra not installed",
)
def test_view_habitat_napari_real_viewer_show_false() -> None:
    """When napari is installed, Viewer(show=False) can host both layers."""
    image = np.ones((5, 8, 8), dtype=np.float32)
    labels = np.zeros((5, 8, 8), dtype=np.int32)
    labels[2:4, 2:6, 2:6] = 1
    viewer = view_habitat_napari(
        image, labels, opacity=0.45, show=False, direction=_RAS
    )
    try:
        names = [layer.name for layer in viewer.layers]
        assert names == ["image", "habitats"]
        assert viewer.layers["habitats"].opacity == pytest.approx(0.45)
        assert viewer._habit_radiological_flips == (False, True, True)
    finally:
        viewer.close()


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["find_spec"]).find_spec("napari") is None,
    reason="napari optional extra not installed",
)
def test_view_habitat_napari_demo_paths_smoke() -> None:
    """Smoke: build viewer on demo IMAGE+HABITAT paths when files exist."""
    root = Path(__file__).resolve().parents[2]
    image_path = (
        root
        / "demo_data"
        / "preprocessed"
        / "processed_images"
        / "images"
        / "subj001"
        / "delay2"
        / "delay2.nii.gz"
    )
    habitat_path = (
        root / "demo_data" / "results" / "habitat_two_step" / "subj001_habitats.nrrd"
    )
    if not image_path.is_file() or not habitat_path.is_file():
        pytest.skip("demo_data habitat view paths not present")

    from habit.api.image import read_image, read_mask

    image_vol = read_image(image_path)
    label_vol = read_mask(habitat_path)
    viewer = view_habitat_napari(
        image_vol.data,
        label_vol.data,
        opacity=0.45,
        show=False,
        spacing=image_vol.spacing,
        direction=image_vol.direction,
    )
    try:
        assert len(viewer.layers) == 2
        assert viewer.layers[0].data.shape == viewer.layers[1].data.shape
        # Demo NIfTI via SimpleITK is RAS → in-plane flips; z index preserved.
        assert viewer._habit_radiological_flips == (False, True, True)
    finally:
        viewer.close()
