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
"""Configuration primitives: schema base, I/O, validation."""

from .base import BaseConfig, ConfigAccessor, ConfigValidationError
from .loader import (
    load_config,
    load_config_with_paths,
    resolve_config_paths,
    save_config,
    validate_config,
)
from .validator import ConfigValidator, load_and_validate_config

__all__ = [
    'BaseConfig',
    'ConfigAccessor',
    'ConfigValidationError',
    'load_config',
    'load_config_with_paths',
    'resolve_config_paths',
    'save_config',
    'validate_config',
    'ConfigValidator',
    'load_and_validate_config',
]
