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
"""Whole-habitat radiomics features (PyRadiomics of the habitat map itself)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union


from habit.contracts.habitat import HabitatMap
from habit.contracts.subject import Subject
from habit.contracts.table import FeatureTable
from habit.habitat_features._base import single_subject_table
from habit.radiomics._domain import (
    DEFAULT_USE_TORCH_RADIOMICS,
    binarized_habitat_mask,
    build_pyradiomics_extractor,
    execute_radiomics,
    sitk_image_from_contract,
)
from habit.habitat_features.registry import HabitatFeatureExtractorRegistry
from habit.spec.specs import Spec

__all__ = ["WholeHabitatRadiomicsFeatures"]


@HabitatFeatureExtractorRegistry.register("whole_habitat")
class WholeHabitatRadiomicsFeatures:
    """
    PyRadiomics features of the habitat map treated as the image itself.

    The multi-label habitat map plays BOTH roles: it is the intensity image
    (habitat ids as grey values) and, binarised, the ROI mask. This is the
    v1 form of the v0.1 ``whole_habitat`` feature type, replicating
    ``HabitatRadiomicsExtractor.extract_radiomics_features_for_whole_habitat``;
    it is the family that quantifies the SHAPE of the habitat partition
    (sphericity, surface area, ...) rather than the underlying intensity.

    Column names are the bare PyRadiomics feature names with ``diagnostic``
    entries dropped, matching the v0.1 ``whole_habitat_radiomics.csv``.
    """

    def __init__(
        self,
        params_file: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        use_torch_radiomics: Union[str, bool] = DEFAULT_USE_TORCH_RADIOMICS,
        torch_device: str = "auto",
        torch_dtype: str = "float32",
    ) -> None:
        self.params_file = params_file
        self.params = dict(params) if params is not None else None
        self.use_torch_radiomics = use_torch_radiomics
        self.torch_device = str(torch_device)
        self.torch_dtype = str(torch_dtype)
        self._params_file = self.params_file
        self._params = self.params
        self._use_torch_radiomics = self.use_torch_radiomics
        self._torch_device = self.torch_device
        self._torch_dtype = self.torch_dtype

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name="whole_habitat",
            params={
                "params_file": self._params_file,
                "params": self._params,
                "use_torch_radiomics": self._use_torch_radiomics,
                "torch_device": self._torch_device,
                "torch_dtype": self._torch_dtype,
            },
        )

    def __call__(self, subject: Subject, habitat_map: HabitatMap) -> FeatureTable:
        """
        Compute the whole-habitat radiomics family for one subject.

        Args:
            subject: Owning subject (labels suffice; intensities unused).
            habitat_map: Habitat labels; used as image and, binarised, mask.

        Returns:
            One-row table of PyRadiomics features of the habitat map.
        """
        owner = f"habitat_feature_extractor.{self.spec.name}"
        extractor = build_pyradiomics_extractor(self._params_file, self._params, owner=owner)
        habitat_sitk = sitk_image_from_contract(habitat_map.label_array, habitat_map.geometry)
        mask_sitk = binarized_habitat_mask(habitat_sitk)
        features = execute_radiomics(
            extractor,
            habitat_sitk,
            mask_sitk,
            label=1,
            use_torch_radiomics=self._use_torch_radiomics,
            torch_device=self._torch_device,
            torch_dtype=self._torch_dtype,
            subject_id=subject.subject_id,
        )
        return single_subject_table(
            subject_id=subject.subject_id,
            features=features,
            habitat_map=habitat_map,
            spec=self.spec,
        )

