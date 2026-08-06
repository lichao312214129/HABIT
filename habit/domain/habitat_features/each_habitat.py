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
"""Per-habitat radiomics features (PyRadiomics within each habitat label)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from habit.contracts.habitat import HabitatMap
from habit.contracts.subject import Subject
from habit.contracts.table import FeatureTable
from habit.domain.habitat_features._base import single_subject_table
from habit.domain.habitat_features._radiomics import (
    build_pyradiomics_extractor,
    execute_radiomics,
    harmonize_mask_geometry,
    resolve_modalities,
    sitk_image_from_contract,
)
from habit.domain.habitat_features.registry import HabitatFeatureExtractorRegistry
from habit.spec.specs import Spec

__all__ = ["EachHabitatRadiomicsFeatures", "EachHabitatRadiomicsFeaturesParams"]


class EachHabitatRadiomicsFeaturesParams(BaseModel):
    """Constructor parameters for :class:`EachHabitatRadiomicsFeatures`."""

    model_config = ConfigDict(extra="forbid")
    #: Path to the PyRadiomics parameter YAML for the raw image; ``None``
    #: selects PyRadiomics defaults (the v0.1 no-file behaviour).
    params_file: Optional[str] = None
    #: Inline PyRadiomics parameter mapping (mutually exclusive with
    #: ``params_file``), for API users holding settings in memory.
    params: Optional[Dict[str, Any]] = None
    #: Modalities to extract from; ``None`` uses every subject image.
    modalities: Optional[Sequence[str]] = None


@HabitatFeatureExtractorRegistry.register("each_habitat")
class EachHabitatRadiomicsFeatures:
    """
    PyRadiomics features of the raw image(s) within each habitat label.

    This is the v1 form of the v0.1 ``each_habitat`` feature type: for every
    habitat id the model can assign, PyRadiomics runs on each raw modality
    with the multi-label habitat map as mask and the habitat id as label,
    replicating
    ``HabitatRadiomicsExtractor.extract_radiomics_features_from_each_habitat``
    (mask-metadata harmonisation included).

    Where v0.1 wrote one CSV per habitat plus a ``habitat_count.csv``, the
    v1 single-row table carries everything at once:

    * ``has_habitat_{id}`` -- 1.0 when the habitat is present in this
      subject, else 0.0 (the v0.1 habitat-count semantics);
    * ``habitat_{id}_{feature}_of_{modality}`` -- one column per PyRadiomics
      feature, ``NaN`` when the habitat is absent (``NaN`` is the honest
      "not measured"; zero would be a fabricated measurement).

    Columns and their order are canonical for a given extractor
    configuration -- every subject of the same model yields the same layout
    regardless of which habitats it contains. A subject whose map has no
    habitat label at all yields only the ``has_habitat_*`` columns.
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
            name="each_habitat",
            params={
                "params_file": self._params_file,
                "params": self._params,
                "modalities": self._modalities,
            },
        )

    def __call__(self, subject: Subject, habitat_map: HabitatMap) -> FeatureTable:
        """
        Compute the per-habitat radiomics family for one subject.

        Args:
            subject: Owning subject; every selected modality is extracted.
            habitat_map: Habitat labels; used as the multi-label mask.

        Returns:
            One-row table of per-habitat radiomics plus presence indicators.
        """
        owner = f"habitat_feature_extractor.{self.spec.name}"
        modalities = resolve_modalities(subject, self._modalities, owner=owner)
        extractor = build_pyradiomics_extractor(self._params_file, self._params, owner=owner)

        labels = np.asarray(habitat_map.label_array)
        present = {int(v) for v in np.unique(labels) if v != 0}
        # Extraction runs only for ids the model can assign; labels outside
        # the model's id set indicate a contract violation and are ignored
        # (consistent with the MSI family's use of habitat_ids).
        measured = [h for h in habitat_map.habitat_ids if h in present]

        # Pass 1: extract every present habitat x modality, remembering the
        # flattened ``{feature}_of_{modality}`` names per habitat.
        per_habitat: Dict[int, Dict[str, float]] = {}
        base_names: List[str] = []
        for modality in modalities:
            volume = subject.image(modality)
            image_sitk = sitk_image_from_contract(volume.load(), volume.geometry)
            mask_sitk = sitk_image_from_contract(labels, habitat_map.geometry)
            # v0.1 semantics: the mask adopts the raw image's metadata.
            harmonize_mask_geometry(image_sitk, mask_sitk)
            for habitat_id in measured:
                habitat_features = per_habitat.setdefault(habitat_id, {})
                for key, value in execute_radiomics(
                    extractor, image_sitk, mask_sitk, label=habitat_id
                ).items():
                    base_name = f"{key}_of_{modality}"
                    if base_name not in base_names:
                        base_names.append(base_name)
                    habitat_features[base_name] = value

        # Pass 2: emit the canonical column layout for every model habitat,
        # NaN-filling the ones absent from this subject.
        features: Dict[str, float] = {}
        for habitat_id in habitat_map.habitat_ids:
            features[f"has_habitat_{habitat_id}"] = 1.0 if habitat_id in per_habitat else 0.0
            habitat_features = per_habitat.get(habitat_id, {})
            for base_name in base_names:
                features[f"habitat_{habitat_id}_{base_name}"] = habitat_features.get(
                    base_name, float("nan")
                )
        return single_subject_table(
            subject_id=subject.subject_id,
            features=features,
            habitat_map=habitat_map,
            spec=self.spec,
        )


HabitatFeatureExtractorRegistry.register_params_model(
    "each_habitat", EachHabitatRadiomicsFeaturesParams
)
