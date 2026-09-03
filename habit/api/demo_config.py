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
Public API for materializing bundled demo YAML configs.

Thin facade over :mod:`habit.utils.demo_config_utils` so third parties can::

    from habit.api.demo_config import copy_demo_config
    copy_demo_config(r"D:/my_habit_work")

without depending on CLI or a repository clone. ``demo_data/`` is never part
of the wheel; download it into the same work directory separately.
"""

from __future__ import annotations

from habit.utils.demo_config_utils import (
    copy_demo_config,
    demo_config_root,
    iter_demo_config_files,
)

__all__ = [
    "copy_demo_config",
    "demo_config_root",
    "iter_demo_config_files",
]
