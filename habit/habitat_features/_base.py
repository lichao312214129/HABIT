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
"""Shared machinery for the built-in habitat feature extractors."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from habit.contracts.habitat import HabitatMap
from habit.contracts.table import FeatureTable
from habit.spec.specs import Spec

#: Identifier column every per-subject habitat feature table carries, so
#: tables from different families join cleanly on it.
SUBJECT_ID_COLUMN = "subject"


def single_subject_table(
    *,
    subject_id: str,
    features: Dict[str, float],
    habitat_map: HabitatMap,
    spec: Spec,
) -> FeatureTable:
    """
    Assemble the one-row-per-subject table every family produces.

    Keeping the assembly in one place guarantees the column-role contract
    (identifier vs model input) stays uniform across feature families and
    that provenance chains back to the habitat map the features describe.

    Args:
        subject_id: Owning subject.
        features: Feature name to value mapping for this family.
        habitat_map: Habitat labels the features were computed from.
        spec: The extractor's specification (provenance fingerprint).

    Returns:
        A single-row table keyed by :data:`SUBJECT_ID_COLUMN`.
    """
    frame = pd.DataFrame([{SUBJECT_ID_COLUMN: subject_id, **features}])
    provenance = habitat_map.provenance.derive(
        produced_by=f"habitat_feature_extractor.{spec.name}",
        spec_fingerprint=spec.fingerprint(),
        random_seed=habitat_map.provenance.random_seed,
    )
    return FeatureTable(
        frame=frame,
        id_columns=(SUBJECT_ID_COLUMN,),
        feature_columns=tuple(features.keys()),
        provenance=provenance,
    )
