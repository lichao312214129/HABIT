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
"""
Utilities module for HABIT package.

Provides common utilities including:
- parallel_utils: Parallel processing with unified interface
- log_utils: Centralized logging management
- progress_utils: Progress bar utilities
- config_utils: Configuration loading and validation
- io_utils: Input/output operations
"""

from .parallel_utils import (
    parallel_map,
    parallel_map_simple,
    ParallelProcessor,
    ProcessingResult,
)

__all__ = [
    "parallel_map",
    "parallel_map_simple", 
    "ParallelProcessor",
    "ProcessingResult",
]
