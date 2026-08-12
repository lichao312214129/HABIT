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
"""Tests for voxel texture / feature-map figures in ``habit.viz``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from habit.contracts.geometry import Geometry
from habit.contracts.habitat import VoxelFeatureField
from habit.contracts.provenance import Provenance
from habit.exceptions import HABITAPIError, OptionalDependencyError
from habit.kernels.voxel_texture import local_entropy_map
from habit.viz import dense_voxel_feature_map, plot_voxel_texture_slice, use_style

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_IMAGE = (
    REPO_ROOT
    / "demo_data"
    / "preprocessed"
    / "images"
    / "subj001"
    / "LAP"
    / "WATER__WATER__Ax_Dyn_LAVA_Flex+C_Series0009.nrrd"
)
DEMO_MASK = (
    REPO_ROOT
    / "demo_data"
    / "preprocessed"
    / "masks"
    / "subj001"
    / "LAP"
    / "WATER__BH_Ax_LAVA_Flex_10min_Series0017_mask.nrrd"
)


def _synthetic_volume() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a small anatomy / ROI / entropy stack for unit tests.

    Returns:
        Tuple of ``(anatomy, roi_mask, entropy_map)`` with shape ``(12, 24, 24)``.
    """
    rng = np.random.default_rng(0)
    anatomy = rng.normal(loc=100.0, scale=25.0, size=(12, 24, 24)).astype(np.float32)
    roi = np.zeros((12, 24, 24), dtype=np.uint8)
    roi[4:9, 6:18, 6:18] = 1
    # Structured texture inside the ROI so entropy is non-trivial.
    anatomy[roi > 0] += rng.normal(0.0, 40.0, size=int(roi.sum())).astype(np.float32)
    entropy = local_entropy_map(anatomy, kernel_size=3, bins=16)
    return anatomy, roi, entropy


def _synthetic_field(
    entropy: np.ndarray,
    roi: np.ndarray,
) -> VoxelFeatureField:
    """
    Pack ROI entropy values into a :class:`VoxelFeatureField`.

    Args:
        entropy: Dense entropy volume.
        roi: Binary ROI mask matching ``entropy`` shape.

    Returns:
        Sparse field with one ``local_entropy-demo`` column.
    """
    inside = roi > 0
    index = np.column_stack(np.nonzero(inside)).astype(np.int64)
    values = entropy[inside].astype(np.float64).reshape(-1, 1)
    geometry = Geometry(
        shape=tuple(int(v) for v in entropy.shape),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    return VoxelFeatureField(
        subject_id="synth",
        feature_names=("local_entropy-demo",),
        values=values,
        voxel_index=index,
        geometry=geometry,
        provenance=Provenance(
            produced_by="tests.viz.test_voxel_texture",
            spec_fingerprint="synthetic-local-entropy",
        ),
    )


def test_plot_voxel_texture_slice_returns_figure_and_saves(tmp_path) -> None:
    """Dense map + anatomy side-by-side returns a Figure and writes a PNG."""
    anatomy, roi, entropy = _synthetic_volume()
    with use_style("radiology"):
        fig = plot_voxel_texture_slice(
            entropy,
            anatomy=anatomy,
            roi_mask=roi,
            axis=0,
            mode="side_by_side",
            feature_label="Local entropy",
        )
    assert isinstance(fig, Figure)

    output_path = tmp_path / "voxel_texture_slice.png"
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    assert output_path.is_file()
    assert output_path.stat().st_size > 0

    joined = " ".join(
        [fig._suptitle.get_text() if fig._suptitle is not None else ""]
        + [ax.get_title() for ax in fig.axes]
    )
    assert joined.isascii()
    assert "entropy" in joined.lower() or "Anatomy" in joined
    # side_by_side draws ROI as contour collections (not a filled alpha mask).
    assert any(getattr(ax, "collections", None) for ax in fig.axes)

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_voxel_texture_slice_accepts_voxel_feature_field(tmp_path) -> None:
    """VoxelFeatureField densifies and plots with the feature name as label."""
    anatomy, roi, entropy = _synthetic_volume()
    field = _synthetic_field(entropy, roi)
    dense = dense_voxel_feature_map(field, "local_entropy-demo")
    assert dense.shape == entropy.shape
    assert np.isnan(dense[roi == 0]).all()
    np.testing.assert_allclose(dense[roi > 0], entropy[roi > 0], rtol=0.0, atol=0.0)

    with use_style("nature"):
        fig = plot_voxel_texture_slice(
            field,
            anatomy=anatomy,
            roi_mask=roi,
            feature="local_entropy-demo",
            axis=0,
            mode="overlay",
        )
    assert isinstance(fig, Figure)
    titles = " ".join(ax.get_title() for ax in fig.axes)
    assert titles.isascii()
    assert "local_entropy" in titles

    fig.savefig(tmp_path / "field_overlay.png", dpi=100, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_voxel_texture_slice_triptych_default() -> None:
    """3D volumes without axis produce three orthogonal panels."""
    anatomy, roi, entropy = _synthetic_volume()
    fig = plot_voxel_texture_slice(
        entropy,
        anatomy=anatomy,
        roi_mask=roi,
        mode="feature_only",
    )
    assert isinstance(fig, Figure)
    # Three feature panels (+ colorbars may add axes); at least three images.
    assert len(fig.axes) >= 3
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_voxel_texture_slice_shape_mismatch_raises() -> None:
    """Mismatched anatomy shape raises HABITAPIError."""
    _anatomy, roi, entropy = _synthetic_volume()
    with pytest.raises(HABITAPIError, match="same shape"):
        plot_voxel_texture_slice(
            entropy,
            anatomy=np.zeros((8, 8, 8), dtype=np.float32),
            roi_mask=roi,
            axis=0,
        )


def test_dense_voxel_feature_map_requires_feature_when_multi_column() -> None:
    """Multi-column fields must name the feature to densify."""
    _anatomy, roi, entropy = _synthetic_volume()
    field = _synthetic_field(entropy, roi)
    wide = VoxelFeatureField(
        subject_id=field.subject_id,
        feature_names=("a", "b"),
        values=np.column_stack([field.values[:, 0], field.values[:, 0]]),
        voxel_index=field.voxel_index,
        geometry=field.geometry,
        provenance=field.provenance,
    )
    with pytest.raises(HABITAPIError, match="feature must be set"):
        dense_voxel_feature_map(wide)


@pytest.mark.skipif(not DEMO_IMAGE.is_file() or not DEMO_MASK.is_file(), reason="demo_data missing")
def test_plot_voxel_texture_slice_demo_data_smoke(tmp_path) -> None:
    """Smoke: local entropy on demo_data LAP + mask produces a PNG."""
    import SimpleITK as sitk

    image = sitk.ReadImage(str(DEMO_IMAGE))
    mask = sitk.ReadImage(str(DEMO_MASK))
    anatomy = np.asarray(sitk.GetArrayFromImage(image), dtype=np.float32)
    roi = np.asarray(sitk.GetArrayFromImage(mask), dtype=np.uint8)
    # Crop to padded ROI bbox so the smoke stays fast on full-size demo volumes.
    coords = np.column_stack(np.nonzero(roi > 0))
    assert coords.size > 0
    z0, y0, x0 = np.maximum(coords.min(axis=0) - 2, 0)
    z1, y1, x1 = np.minimum(coords.max(axis=0) + 3, np.array(roi.shape))
    anatomy_c = anatomy[z0:z1, y0:y1, x0:x1]
    roi_c = roi[z0:z1, y0:y1, x0:x1]
    entropy = local_entropy_map(anatomy_c, kernel_size=3, bins=16)
    spacing = tuple(float(v) for v in image.GetSpacing())

    with use_style("radiology"):
        fig = plot_voxel_texture_slice(
            entropy,
            anatomy=anatomy_c,
            roi_mask=roi_c,
            axis=0,
            mode="side_by_side",
            spacing=spacing,
            feature_label="Local entropy",
            title="Demo subj001 LAP local entropy",
        )
    out = tmp_path / "demo_voxel_texture.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    assert out.is_file() and out.stat().st_size > 0
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_voxel_texture_slice_optional_deps_message(monkeypatch) -> None:
    """Missing matplotlib surfaces OptionalDependencyError via require()."""
    import habit.viz.voxel_texture as voxel_texture_mod

    def _boom(name: str, *args, **kwargs):
        raise OptionalDependencyError(
            f"{name} is required for voxel texture figures.\n"
            "Install with: pip install 'habit[viz]'"
        )

    monkeypatch.setattr(voxel_texture_mod, "require", _boom)
    anatomy, roi, entropy = _synthetic_volume()
    with pytest.raises(OptionalDependencyError, match="habit\\[viz\\]"):
        plot_voxel_texture_slice(entropy, anatomy=anatomy, roi_mask=roi, axis=0)
