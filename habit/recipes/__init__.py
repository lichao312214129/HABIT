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
"""L4 recipes: named study designs, assembled from L3 components.

A recipe is the shortest path from "I want a two-step habitat analysis" to a
result object, and nothing more. It contains no engine of its own: no run
loop, no resume, no configuration parsing, no output directory. Those live in
the execution backends (L1/L2), the writers (L1) and the CLI (L5)
respectively, which is what keeps a recipe callable from a notebook, a web
service or someone else's pipeline.
"""

from __future__ import annotations

from habit.recipes.auxiliary import dice, dicom_info, merge_tables
from habit.recipes.comparison import compare_models, pairwise_delong_test
from habit.recipes.features import extract_habitat_features, traditional_radiomics
from habit.recipes.habitat import (
    apply_habitat_model,
    direct_pooling,
    one_step,
    two_step,
)
from habit.recipes.icc import icc_analysis
from habit.recipes.precision import (
    identify_precise_voxel_features,
    voxel_radiomics_factory,
)
from habit.recipes.modeling import (
    CVResult,
    ModelResult,
    PredictionResult,
    cross_validate,
    predict_model,
    train_model,
)
from habit.recipes.preprocess import (
    preprocess_image,
    preprocess_images,
    preprocess_subject,
)
from habit.recipes.sort_dicom import sort_dicom
from habit.recipes.result import StudyResult
from habit.recipes.study import (
    Study,
    direct_pooling_habitat,
    one_step_habitat,
    two_step_habitat,
)
from habit.recipes.test_retest import test_retest_analysis
from habit.recipes.yaml_runner import run_from_yaml

__all__ = [
    "CVResult",
    "ModelResult",
    "PredictionResult",
    "Study",
    "StudyResult",
    "apply_habitat_model",
    "compare_models",
    "cross_validate",
    "dice",
    "dicom_info",
    "direct_pooling",
    "direct_pooling_habitat",
    "extract_habitat_features",
    "icc_analysis",
    "identify_precise_voxel_features",
    "merge_tables",
    "one_step",
    "one_step_habitat",
    "pairwise_delong_test",
    "predict_model",
    "preprocess_image",
    "preprocess_images",
    "preprocess_subject",
    "run_from_yaml",
    "sort_dicom",
    "test_retest_analysis",
    "traditional_radiomics",
    "train_model",
    "two_step",
    "two_step_habitat",
    "voxel_radiomics_factory",
]
