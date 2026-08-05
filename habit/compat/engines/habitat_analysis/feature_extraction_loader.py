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
"""Backward-compatible re-export of the L1 feature-extraction loader."""

from __future__ import annotations

from habit.compat.feature_extraction_loader import (
    load_feature_extraction_config_from_file,
    parse_feature_extraction_config,
    plugin_configs_for_feature_types,
)

__all__ = [
    "load_feature_extraction_config_from_file",
    "parse_feature_extraction_config",
    "plugin_configs_for_feature_types",
]
