#!/usr/bin/env python3
"""Fail when private graph feature artifacts appear in the public HABIT tree."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if len(sys.argv) > 1:
    ROOT = Path(sys.argv[1]).resolve()

FORBIDDEN_PATH_PARTS = {
    "graph_features",
}

FORBIDDEN_FILE_NAMES = {
    "test_habitat_graph_features.py",
}

FORBIDDEN_DOC_PREFIX = Path("docs/source/reference/features/graph")

TEXT_PATTERNS = [
    re.compile(r"habitat_features[/\\]graph_features"),
    re.compile(
        r"from habit\.core\.habitat_analysis\.habitat_features\.graph_features"
    ),
    re.compile(r"from \.graph_features import"),
    re.compile(r"HabitatGraphFeatureExtractor"),
    re.compile(r"GraphHabitatFeaturePlugin"),
    re.compile(r"@register_habitat_feature\(\s*['\"]graph['\"]"),
]

ALLOWLIST_FILES = {
    ROOT / "scripts" / "check_public_leak.py",
    ROOT / "scripts" / "sync_v1_to_habit_project.py",
    ROOT / "habit" / "core" / "habitat_analysis" / "feature_extraction_loader.py",
    ROOT / "habit" / "core" / "habitat_analysis" / "feature_registry.py",
}


def _iter_text_files(base: Path):
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if path in ALLOWLIST_FILES:
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".nrrd", ".parquet"}:
            continue
        if "docs" in path.parts and path.suffix.lower() in {".rst", ".md"}:
            if str(path.relative_to(base)).startswith(str(FORBIDDEN_DOC_PREFIX).replace("\\", "/")):
                yield path
                continue
        yield path


def main() -> int:
    errors: list[str] = []

    for part in FORBIDDEN_PATH_PARTS:
        matches = list(ROOT.rglob(part))
        for match in matches:
            if match.is_dir():
                errors.append(f"forbidden directory: {match.relative_to(ROOT)}")

    for name in FORBIDDEN_FILE_NAMES:
        for match in ROOT.rglob(name):
            errors.append(f"forbidden file: {match.relative_to(ROOT)}")

    if (ROOT / FORBIDDEN_DOC_PREFIX).exists():
        errors.append(f"forbidden docs tree: {FORBIDDEN_DOC_PREFIX}")

    pyproject = ROOT / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"(?im)^networkx\s*=", text):
            errors.append("networkx dependency found in pyproject.toml")
        if re.search(r"(?im)^graph\s*=\s*\[\s*[\"']networkx[\"']\s*\]", text):
            errors.append("graph extra with networkx found in pyproject.toml")

    for path in _iter_text_files(ROOT):
        rel = path.relative_to(ROOT)
        rel_posix = rel.as_posix()
        if rel_posix.startswith(FORBIDDEN_DOC_PREFIX.as_posix()):
            errors.append(f"forbidden graph doc: {rel_posix}")
            continue
        if "graph_features" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern in TEXT_PATTERNS:
            if pattern.search(text):
                errors.append(f"pattern {pattern.pattern!r} in {rel_posix}")
                break

    if errors:
        print("Public leak check FAILED:")
        for item in errors:
            print(f"  - {item}")
        return 1

    print(f"Public leak check passed for {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
