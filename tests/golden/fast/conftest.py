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
"""Pytest helpers for the synthetic fast golden gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from tests.golden.fast._runner import (
    FAST_GOLDEN_CASES,
    baseline_path,
    compare_fast_records,
    run_case,
)

__all__ = [
    "FAST_GOLDEN_CASES",
    "baseline_path",
    "compare_fast_records",
    "load_fast_baseline",
    "run_case",
]


def load_fast_baseline(case_name: str) -> Dict[str, Any]:
    """
    Load one committed fast baseline, skipping when absent.

    Args:
        case_name: Case identifier, e.g. ``habitat_two_step``.

    Returns:
        Parsed baseline JSON.
    """
    path = baseline_path(case_name)
    if not path.is_file():
        pytest.skip(
            f"No fast golden baseline for '{case_name}'. "
            "Generate it locally with: python scripts/make_fast_golden_baseline.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))
