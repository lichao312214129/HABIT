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
"""Shared machinery for the built-in habitat model fitters.

Pooling, feature-name validation, cohort fingerprinting and model assembly
are identical for every population-level fitter; keeping them here means the
k-means and GMM fitters only differ in the actual clustering call.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from habit.api.exceptions import CompatibilityError, HABITAPIError
from habit.contracts.habitat import HabitatModel, Supervoxelization
from habit.contracts.provenance import Provenance, software_fingerprint
from habit.contracts.subject import Cohort, CohortFingerprint
from habit.spec.specs import Spec


def pool_supervoxel_features(
    units: Sequence[Supervoxelization],
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """
    Concatenate every unit's supervoxel features in the given order.

    Order is part of the contract because clustering can be order-sensitive;
    this helper never sorts or shuffles.

    Args:
        units: Supervoxelizations in cohort order.

    Returns:
        ``(matrix, feature_names)`` where matrix has one row per supervoxel
        across all subjects.

    Raises:
        HABITAPIError: If ``units`` is empty.
        CompatibilityError: If feature columns differ between subjects.
    """
    if not units:
        raise HABITAPIError("Habitat model fitting requires at least one unit.")
    feature_names = tuple(str(c) for c in units[0].features.columns)
    frames: List[pd.DataFrame] = []
    for unit in units:
        current = tuple(str(c) for c in unit.features.columns)
        if current != feature_names:
            raise CompatibilityError(
                f"Subject {unit.subject_id!r} provides features {current}, "
                f"but the cohort expects {feature_names}."
            )
        frames.append(unit.features)
    matrix = pd.concat(frames, axis=0).to_numpy(dtype=np.float64)
    return matrix, feature_names


def fingerprint_from_units(units: Sequence[Supervoxelization]) -> CohortFingerprint:
    """
    Derive a non-identifiable cohort description from the units alone.

    Used when the fitter is called without the originating cohort; the
    digest still proves two runs used the same subject set, in the same
    order, without revealing identifiers.

    Args:
        units: Supervoxelizations in cohort order.

    Returns:
        A fingerprint with no modality information (unknowable here).
    """
    digest = hashlib.sha256(
        ("habit-cohort-v1" + "\n" + "\n".join(u.subject_id for u in units)).encode(
            "utf-8"
        )
    ).hexdigest()
    return CohortFingerprint(
        n_subjects=len(units),
        modalities=(),
        subject_id_digest=digest,
    )


def build_habitat_model(
    *,
    fitter_name: str,
    spec: Spec,
    centroids: np.ndarray,
    feature_names: Tuple[str, ...],
    units: Sequence[Supervoxelization],
    cohort: Optional[Cohort],
    random_seed: int,
    preprocessing_state: Optional[dict] = None,
) -> HabitatModel:
    """
    Assemble the fitted :class:`HabitatModel` with full provenance.

    The model id binds the specification fingerprint to the cohort digest,
    so models fitted from different cohorts or different specs can never be
    confused; provenance chains every unit's record into the model.

    Args:
        fitter_name: Registered fitter name, e.g. ``"kmeans"``.
        spec: The fitter's specification.
        centroids: Population cluster centres.
        feature_names: Feature order of the centroid columns.
        units: Supervoxelizations used for fitting (provenance inputs).
        cohort: Originating cohort, when available.
        random_seed: Seed the fitter ran with.
        preprocessing_state: Optional state learned at fit time.

    Returns:
        The self-contained habitat model.
    """
    fingerprint = cohort.summarize() if cohort is not None else fingerprint_from_units(units)
    model_id = hashlib.sha256(
        f"{fitter_name}:{spec.fingerprint()}:{fingerprint.subject_id_digest}".encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    provenance = Provenance(
        produced_by=f"habitat_model_fitter.{fitter_name}",
        spec_fingerprint=spec.fingerprint(),
        inputs=tuple(unit.provenance for unit in units),
        software=software_fingerprint(),
        random_seed=random_seed,
        created_at=None,
    )
    return HabitatModel(
        model_id=f"{fitter_name}-{model_id}",
        n_habitats=int(centroids.shape[0]),
        feature_names=tuple(feature_names),
        centroids=centroids,
        preprocessing_state=preprocessing_state or {},
        spec_payload={f"habitat_model_fitter": spec.to_dict()},
        cohort_fingerprint=fingerprint,
        provenance=provenance,
    )
