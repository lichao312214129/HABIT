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
"""Precision analysis: which voxel features deserve to define habitats.

Voxel-wise radiomics is noisy; clustering features that do not survive a
simulated re-acquisition produces habitats nobody can reproduce. This
package implements the precision screen of Prior et al. (Radiol Artif
Intell 2024;6(2):e230118) as composable domain components:

* :class:`ImagePerturbation` implementations (noise, translation, rotation)
  and the :class:`PerturbationChain` composing one simulated retest;
* :func:`precision_panel` / :func:`aggregate_panels` computing per-subject
  and cohort-level ICC tables on voxel feature fields;
* :func:`identify_precise_features` applying the LCL screen across
  experiments and returning the serialisable :class:`PreciseFeatureSet`;
* :func:`habitat_stability` scoring habitat maps under perturbation.
"""

from __future__ import annotations

from habit.domain.precision.analysis import (
    aggregate_panels,
    identify_precise_features,
    precision_panel,
)
from habit.domain.precision.chain import PerturbationChain
from habit.domain.precision.perturbations import (
    GaussianNoisePerturbation,
    GaussianNoisePerturbationParams,
    RotationPerturbation,
    RotationPerturbationParams,
    TranslationPerturbation,
    TranslationPerturbationParams,
)
from habit.domain.precision.precise_set import PreciseFeatureSet
from habit.domain.precision.registry import ImagePerturbationRegistry
from habit.domain.precision.stability import habitat_stability

__all__ = [
    "GaussianNoisePerturbation",
    "GaussianNoisePerturbationParams",
    "ImagePerturbationRegistry",
    "PerturbationChain",
    "PreciseFeatureSet",
    "RotationPerturbation",
    "RotationPerturbationParams",
    "TranslationPerturbation",
    "TranslationPerturbationParams",
    "aggregate_panels",
    "habitat_stability",
    "identify_precise_features",
    "precision_panel",
]
