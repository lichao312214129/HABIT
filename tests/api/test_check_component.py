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
"""Tests for ``habit.check_component``."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_check_component_resolves_builtin_names() -> None:
    """Known built-in components resolve; unknown names and domains do not."""
    import habit

    assert habit.check_component("kmeans", domain="habitat_model_fitter") is True
    assert habit.check_component("slic", domain="supervoxelizer") is True
    assert (
        habit.check_component("not_a_real_plugin", domain="habitat_model_fitter")
        is False
    )
    assert habit.check_component("kmeans", domain="not_a_real_domain") is False
