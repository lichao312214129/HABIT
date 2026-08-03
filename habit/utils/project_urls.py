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
"""Canonical project URLs for the public HABIT release."""

from __future__ import annotations

DOCS_BASE_URL: str = "https://lichao312214129.github.io/HABIT/"
GITHUB_REPO_URL: str = "https://github.com/lichao312214129/HABIT"
GITHUB_REPO_SLUG: str = "lichao312214129/HABIT"
DOCS_SITE_LABEL: str = "HABIT"


def docs_page(relative_path: str) -> str:
    """Build an absolute documentation URL from a path under the docs site root."""
    normalized: str = relative_path.lstrip("/")
    return f"{DOCS_BASE_URL}{normalized}"


def github_issues_url() -> str:
    """Return the GitHub Issues URL for the active repository."""
    return f"{GITHUB_REPO_URL}/issues"
