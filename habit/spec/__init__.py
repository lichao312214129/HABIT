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
"""The Spec / DataSource / RunPolicy tripartition.

``habit.spec`` holds algorithm specifications (what to compute), run
policies (how to execute), the YAML isomorphism, and the legacy-config
translation. Data location is deliberately absent: that is the
:class:`~habit.contracts.ops.DataSource` contract's concern.
"""

from __future__ import annotations

from habit.spec.document import (
    build_habitat_document,
    load_habitat_config,
    save_habitat_config,
)
from habit.spec.expressions import parse_feature_expression
from habit.spec.legacy import (
    LegacyConfigAdapter,
    LegacyTranslation,
    MigrationReport,
    detect_yaml_version,
    migrate_yaml,
    validate_v1_document,
)
from habit.spec.policy import RunPolicy
from habit.spec.specs import HabitatSpec, MLSpec, Spec, Stage, coerce_spec
from habit.spec.stages import (
    POOL_COMPONENT_NAME,
    ROLE_ASSIGN,
    ROLE_EXTRACT_SUPERVOXEL_FEATURES,
    ROLE_EXTRACT_VOXEL_FEATURES,
    ROLE_FIT,
    ROLE_PARTITION,
    ROLE_POOL,
    ROLE_POSTPROCESS_HABITAT,
    ROLE_POSTPROCESS_SUPERVOXEL,
    ROLE_PREPROCESS,
    ROLE_QUANTIFY,
)
from habit.spec.yaml_io import (
    load_habitat_spec,
    load_run_policy,
    save_habitat_spec,
    save_run_policy,
)

__all__ = [
    "Spec",
    "Stage",
    "HabitatSpec",
    "MLSpec",
    "RunPolicy",
    "coerce_spec",
    "POOL_COMPONENT_NAME",
    "ROLE_EXTRACT_VOXEL_FEATURES",
    "ROLE_PREPROCESS",
    "ROLE_PARTITION",
    "ROLE_EXTRACT_SUPERVOXEL_FEATURES",
    "ROLE_POOL",
    "ROLE_FIT",
    "ROLE_ASSIGN",
    "ROLE_QUANTIFY",
    "ROLE_POSTPROCESS_SUPERVOXEL",
    "ROLE_POSTPROCESS_HABITAT",
    "parse_feature_expression",
    "load_habitat_spec",
    "save_habitat_spec",
    "load_run_policy",
    "save_run_policy",
    "build_habitat_document",
    "save_habitat_config",
    "load_habitat_config",
    "LegacyConfigAdapter",
    "LegacyTranslation",
    "MigrationReport",
    "detect_yaml_version",
    "migrate_yaml",
    "validate_v1_document",
]
