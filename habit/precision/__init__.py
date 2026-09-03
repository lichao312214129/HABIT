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

* :class:`ImagePerturbation` implementations (noise, translation, rotation,
  optional MONAI ``bspline_deform``) and the :class:`PerturbationChain`
  composing one simulated retest;
* :func:`precision_panel` / :func:`aggregate_panels` computing per-subject
  and cohort-level ICC tables on voxel feature fields;
* :func:`identify_precise_features` applying the LCL screen across
  experiments and returning the serialisable :class:`PreciseFeatureSet`;
* :func:`habitat_stability` scoring habitat maps under perturbation;
* :func:`align_habitat_map` remapping independently clustered labels onto
  a reference (centroid / test-retest matcher, or overlap Hungarian).
"""

from __future__ import annotations

from habit.precision.analysis import (
    aggregate_panels,
    identify_precise_features,
    precision_panel,
)
from habit.precision.atoms import perturb_image
from habit.precision.chain import PerturbationChain
from habit.precision.perturbations import (
    BSplineDeformPerturbation,
    GaussianNoisePerturbation,
    GradientWeightedPerturbation,
    MorphologicalPerturbation,
    RigidPerturbation,
    RotationPerturbation,
    SliceExtentPerturbation,
    TranslationPerturbation,
    prior2024_retest_perturbation,
)
from habit.precision.precise_set import PreciseFeatureSet
from habit.precision.registry import ImagePerturbationRegistry
from habit.precision.stability import align_habitat_map, habitat_stability

from habit._protocols import ImagePerturbation

__all__ = [
    "ImagePerturbation",
    "BSplineDeformPerturbation",
    "GaussianNoisePerturbation",
    "GradientWeightedPerturbation",
    "ImagePerturbationRegistry",
    "MorphologicalPerturbation",
    "PerturbationChain",
    "PreciseFeatureSet",
    "RigidPerturbation",
    "RotationPerturbation",
    "SliceExtentPerturbation",
    "TranslationPerturbation",
    "aggregate_panels",
    "align_habitat_map",
    "habitat_stability",
    "identify_precise_features",
    "perturb_image",
    "precision_panel",
    "prior2024_retest_perturbation",
]
