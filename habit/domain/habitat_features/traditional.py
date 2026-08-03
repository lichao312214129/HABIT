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
"""Traditional (whole-ROI) radiomics habitat features."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from pydantic import BaseModel

from habit.contracts.habitat import HabitatMap
from habit.contracts.subject import Subject
from habit.contracts.table import FeatureTable
from habit.domain.habitat_features._base import single_subject_table
from habit.domain.habitat_features._radiomics import (
    binarized_habitat_mask,
    build_pyradiomics_extractor,
    execute_radiomics,
    harmonize_mask_geometry,
    resolve_modalities,
    sitk_image_from_contract,
)
from habit.domain.habitat_features.registry import HabitatFeatureExtractorRegistry
from habit.spec.specs import Spec

__all__ = ["TraditionalRadiomicsHabitatFeatures", "TraditionalRadiomicsHabitatFeaturesParams"]


class TraditionalRadiomicsHabitatFeaturesParams(BaseModel):
    """Constructor parameters for :class:`TraditionalRadiomicsHabitatFeatures`."""

    #: Path to the PyRadiomics parameter YAML for the raw image; ``None``
    #: selects PyRadiomics defaults (the v0.1 no-file behaviour).
    params_file: Optional[str] = None
    #: Inline PyRadiomics parameter mapping (mutually exclusive with
    #: ``params_file``), for API users holding settings in memory.
    params: Optional[Dict[str, Any]] = None
    #: Modalities to extract from; ``None`` uses every subject image.
    modalities: Optional[Sequence[str]] = None


@HabitatFeatureExtractorRegistry.register("traditional")
class TraditionalRadiomicsHabitatFeatures:
    """
    PyRadiomics features of the raw image(s) within the whole ROI.

    This is the v1 form of the v0.1 ``traditional`` feature type (and of the
    standalone ``habit radiomics`` path): the habitat map is binarised into
    a single ROI mask and PyRadiomics runs on each raw modality within it.
    The mask construction, the mask-metadata harmonisation and the
    PyRadiomics invocation replicate the v0.1
    ``HabitatRadiomicsExtractor.extract_tranditional_radiomics`` exactly, so
    extracted numbers stay comparable with previously published results.

    Column names keep the v0.1 CSV scheme ``{feature}_of_{modality}`` with
    ``diagnostic`` entries dropped. A per-subject failure (e.g. an unreadable
    modality) raises instead of yielding a silently empty row -- the
    execution layer's failure policy decides whether the cohort run
    continues, which is where that decision belongs in v1.
    """

    def __init__(
        self,
        params_file: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        modalities: Optional[Sequence[str]] = None,
    ) -> None:
        self._params_file = params_file
        self._params = dict(params) if params is not None else None
        self._modalities = tuple(modalities) if modalities is not None else None

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        return Spec(
            name="traditional",
            params={
                "params_file": self._params_file,
                "params": self._params,
                "modalities": self._modalities,
            },
        )

    def __call__(self, subject: Subject, habitat_map: HabitatMap) -> FeatureTable:
        """
        Compute the traditional-radiomics family for one subject.

        Args:
            subject: Owning subject; every selected modality is extracted.
            habitat_map: Habitat labels; binarised into the ROI mask.

        Returns:
            One-row table of ``{feature}_of_{modality}`` columns.
        """
        owner = f"habitat_feature_extractor.{self.spec.name}"
        modalities = resolve_modalities(subject, self._modalities, owner=owner)
        extractor = build_pyradiomics_extractor(self._params_file, self._params, owner=owner)
        habitat_sitk = sitk_image_from_contract(habitat_map.label_array, habitat_map.geometry)
        mask_sitk = binarized_habitat_mask(habitat_sitk)

        features: Dict[str, float] = {}
        for modality in modalities:
            volume = subject.image(modality)
            image_sitk = sitk_image_from_contract(volume.load(), volume.geometry)
            # v0.1 semantics: the mask adopts the raw image's metadata.
            harmonize_mask_geometry(image_sitk, mask_sitk)
            for key, value in execute_radiomics(extractor, image_sitk, mask_sitk, label=1).items():
                features[f"{key}_of_{modality}"] = value
        return single_subject_table(
            subject_id=subject.subject_id,
            features=features,
            habitat_map=habitat_map,
            spec=self.spec,
        )


HabitatFeatureExtractorRegistry.register_params_model(
    "traditional", TraditionalRadiomicsHabitatFeaturesParams
)
