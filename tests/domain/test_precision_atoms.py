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
"""Focused tests for the volume-level precision atoms."""

from __future__ import annotations

import numpy as np
import pytest

from habit.contracts import VoxelFeatureField
from habit.voxel_features import extract_voxel_texture
from habit.precision import perturb_image, precision_panel
from habit.exceptions import ComponentNotFoundError, HABITAPIError

from .conftest import make_subject


class TestPerturbImage:
    def test_gaussian_noise_same_grid_and_reproducible(self) -> None:
        """Image in, ImageVolume out; same seed repeats; no Cohort."""
        subject = make_subject("P1")
        image = subject.image("T1")
        original = np.asarray(image.data).copy()
        first = perturb_image(image, method="gaussian_noise", sigma=0.5, seed=7)
        second = perturb_image(image, method="gaussian_noise", sigma=0.5, seed=7)
        assert first.data.shape == original.shape
        assert first.spacing == image.spacing
        assert first.origin == image.origin
        assert first.direction == image.direction
        assert not np.allclose(first.data, original)
        np.testing.assert_array_equal(np.asarray(image.data), original)
        np.testing.assert_array_equal(first.data, second.data)
        assert first.metadata["perturbation"]["name"] == "gaussian_noise"

    def test_translation_uses_method_params(self) -> None:
        """Constructor kwargs reach the registered translation step."""
        subject = make_subject("P1", shape=(12, 12, 12))
        image = subject.image("T1")
        mask = subject.mask("tumor")
        moved = perturb_image(
            image,
            method="translation",
            mask=mask,
            shift_voxels=(0.5, 0.0, 0.0),
            random_signs=False,
            seed=0,
        )
        assert moved.data.shape == image.data.shape
        assert not np.allclose(moved.data, image.data)

    def test_unknown_method_raises(self) -> None:
        """An unregistered name is an honest lookup failure."""
        image = make_subject("P1").image("T1")
        with pytest.raises(ComponentNotFoundError):
            perturb_image(image, method="not_a_method")

    def test_empty_method_raises(self) -> None:
        """A blank method name is rejected before the registry."""
        image = make_subject("P1").image("T1")
        with pytest.raises(HABITAPIError, match="method"):
            perturb_image(image, method="  ")


class TestExtractVoxelTexture:
    def test_returns_field_and_records_grid_point(self) -> None:
        """Image + mask + knobs yield a VoxelFeatureField; no Cohort."""
        pytest.importorskip("radiomics")
        subject = make_subject("P1", shape=(12, 12, 12))
        image = subject.image("T1")
        mask = subject.mask("tumor")
        field = extract_voxel_texture(
            image,
            mask,
            kernel_radius=1,
            bin_width=12.0,
            feature_classes={"firstorder": ["Mean"]},
        )
        assert isinstance(field, VoxelFeatureField)
        assert field.values.ndim == 2
        assert field.values.shape[0] == int(np.count_nonzero(np.asarray(mask.data) > 0))
        assert field.values.shape[1] >= 1
        assert field.provenance.produced_by == "voxel_feature_extractor.voxel_radiomics"
        assert field.provenance.spec_fingerprint

    def test_kernel_radius_is_a_first_class_knob(self) -> None:
        """R1 vs R3 are two calls; provenance fingerprints differ."""
        pytest.importorskip("radiomics")
        subject = make_subject("P1", shape=(12, 12, 12))
        image = subject.image("T1")
        mask = subject.mask("tumor")
        kwargs = {
            "bin_width": 12.0,
            "feature_classes": {"firstorder": ["Mean"]},
        }
        r1 = extract_voxel_texture(image, mask, kernel_radius=1, **kwargs)
        r3 = extract_voxel_texture(image, mask, kernel_radius=3, **kwargs)
        assert r1.feature_names == r3.feature_names
        assert r1.provenance.spec_fingerprint != r3.provenance.spec_fingerprint

    def test_rejects_feature_classes_and_params_together(self) -> None:
        """The two settings forms are mutually exclusive."""
        subject = make_subject("P1")
        with pytest.raises(HABITAPIError, match="not both"):
            extract_voxel_texture(
                subject.image("T1"),
                subject.mask("tumor"),
                feature_classes={"firstorder": ["Mean"]},
                params={"setting": {"binWidth": 12}},
            )

    def test_repeatability_panel_from_atoms(self) -> None:
        """perturb + extract + precision_panel is the paper repeatability combo."""
        pytest.importorskip("radiomics")
        subject = make_subject("P1", shape=(12, 12, 12))
        image = subject.image("T1")
        mask = subject.mask("tumor")
        perturbed = perturb_image(image, method="gaussian_noise", sigma=0.4, seed=3)
        kwargs = {
            "kernel_radius": 1,
            "bin_width": 12.0,
            "feature_classes": {"firstorder": ["Mean"]},
        }
        original_field = extract_voxel_texture(image, mask, **kwargs)
        perturbed_field = extract_voxel_texture(perturbed, mask, **kwargs)
        panel = precision_panel(
            {"original": original_field, "perturbed": perturbed_field},
            agreement="absolute",
            min_voxels=8,
        )
        assert "value" in panel.columns
        assert "lcl" in panel.columns
        assert len(panel) == len(original_field.feature_names)
