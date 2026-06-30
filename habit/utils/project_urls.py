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
