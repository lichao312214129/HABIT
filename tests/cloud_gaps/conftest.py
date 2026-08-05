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
"""Shared fixtures for cloud gap tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.cloud_gaps.synth_data import DEFAULT_DATA_ROOT, write_synthetic_demo_dataset

#: Repository root (``tests/cloud_gaps/conftest.py`` -> parents[2]).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def demo_data_root(repo_root: Path) -> Path:
    """
    Ensure the synthetic demo cohort exists at the path demo configs reference.

    The dataset is written once per session and left untracked (gitignored).
    """
    return write_synthetic_demo_dataset(DEFAULT_DATA_ROOT)


@pytest.fixture
def cwd_repo_root(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> Path:
    """
    Run CLI/YAML tests with cwd at the repository root.

    v1 YAML paths resolve against the current working directory.
    """
    monkeypatch.chdir(repo_root)
    return repo_root
