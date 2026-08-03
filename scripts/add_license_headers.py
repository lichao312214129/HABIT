"""
Maintain the HABIT license header on all source files.

The script is idempotent and also performs the v0.1 -> v1.0 relicensing
migration. Files carrying any legacy "HABIT Software License" header variant are
rewritten to the Apache-2.0 header, files already carrying the Apache-2.0 header
are left untouched, and files with no header get one.

Legacy headers were written by several earlier passes and exist in a handful of
slightly different shapes, so they are detected structurally -- a leading
comment block anchored at the HABIT copyright line and mentioning the legacy
license -- rather than by matching each variant verbatim.

Files that vendor third-party code carry an additional upstream attribution
block below the HABIT header. Those blocks are part of the file body and are
never removed, because Apache-2.0 section 4(c) requires retaining them.

Run from the repository root:
    python scripts/add_license_headers.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]

# Directories walked for source files, plus individual files at the repo root.
_SOURCE_DIRS: Tuple[Path, ...] = (
    _ROOT / "habit",
    _ROOT / "tests",
    _ROOT / "scripts",
)
_SOURCE_FILES: Tuple[Path, ...] = (_ROOT / "setup.py",)

# This script stores header text as data, so it must never rewrite itself.
_SELF = Path(__file__).resolve()

_COPYRIGHT = "Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors."

_PY_HEADER = f"""\
# {_COPYRIGHT}
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

_C_HEADER = f"""\
/*
 * {_COPYRIGHT}
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
"""

_LEGACY_MARKER = "HABIT Software License"
_CURRENT_MARKER = "Licensed under the Apache License, Version 2.0"


def _strip_legacy_hash_block(content: str) -> Optional[str]:
    """
    Remove a leading ``#`` comment block carrying a legacy license header.

    Anchoring on the copyright line guarantees the file does not begin with a
    shebang or encoding declaration, so the whole leading comment run can be
    removed safely.

    Args:
        content: Full file text.

    Returns:
        The remaining file body, or ``None`` when no legacy block is present.
    """
    if not content.startswith(f"# {_COPYRIGHT}"):
        return None

    lines = content.splitlines(keepends=True)
    end = 0
    for index, line in enumerate(lines):
        if not line.startswith("#"):
            break
        end = index + 1

    if _LEGACY_MARKER not in "".join(lines[:end]):
        return None

    body = lines[end:]
    # Drop one blank separator so the new header sits directly above the code.
    if body and not body[0].strip():
        body = body[1:]
    return "".join(body)


def _strip_legacy_c_block(content: str) -> Optional[str]:
    """
    Remove a leading ``/* ... */`` block carrying a legacy license header.

    Args:
        content: Full file text.

    Returns:
        The remaining file body, or ``None`` when no legacy block is present.
    """
    if not content.startswith("/*"):
        return None

    close = content.find("*/")
    if close == -1 or _LEGACY_MARKER not in content[:close]:
        return None

    body = content[close + 2:]
    return body.lstrip("\n")


def _apply_header(path: Path, header: str, is_c_style: bool) -> str:
    """
    Ensure a single source file carries the current license header.

    Args:
        path: Target source file.
        header: Current header text, ending with a newline.
        is_c_style: Whether the file uses ``/* */`` rather than ``#`` comments.

    Returns:
        One of ``"unchanged"``, ``"relicensed"``, ``"added"``, or ``"review"``.
        ``"review"`` means the file mentions a license but does not start with a
        recognisable header block, so a human must inspect it.

    Raises:
        OSError: When the file cannot be read or written.
    """
    content = path.read_text(encoding="utf-8")

    if content.startswith(header):
        return "unchanged"

    stripper = _strip_legacy_c_block if is_c_style else _strip_legacy_hash_block
    body = stripper(content)
    if body is not None:
        path.write_text(header + body, encoding="utf-8", newline="\n")
        return "relicensed"

    # A stale mention anywhere in the file means the migration is incomplete;
    # failing loudly is safer than leaving contradictory license terms behind.
    if _LEGACY_MARKER in content or _CURRENT_MARKER in content:
        return "review"

    path.write_text(header + content, encoding="utf-8", newline="\n")
    return "added"


def _iter_sources() -> List[Tuple[Path, str, bool]]:
    """
    Collect every source file that must carry a license header.

    Returns:
        Tuples of ``(path, header, is_c_style)``.
    """
    collected: List[Tuple[Path, str, bool]] = []

    for directory in _SOURCE_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            if path.resolve() != _SELF:
                collected.append((path, _PY_HEADER, False))
        for pattern in ("*.c", "*.h"):
            for path in sorted(directory.rglob(pattern)):
                collected.append((path, _C_HEADER, True))

    for path in _SOURCE_FILES:
        if path.is_file():
            collected.append((path, _PY_HEADER, False))

    return collected


def main() -> None:
    """Apply the current license header across the repository's source tree."""
    counts = {"unchanged": 0, "relicensed": 0, "added": 0, "review": 0}
    review: List[str] = []
    failures: List[str] = []

    for path, header, is_c_style in _iter_sources():
        try:
            outcome = _apply_header(path, header, is_c_style)
        except OSError as exc:
            failures.append(f"{path}: {exc}")
            continue
        counts[outcome] += 1
        if outcome == "review":
            review.append(str(path.relative_to(_ROOT)))

    print(
        f"relicensed={counts['relicensed']} added={counts['added']} "
        f"unchanged={counts['unchanged']} review={counts['review']}"
    )
    if review:
        print("Manual review required (unexpected license text):")
        for item in review:
            print(f"  {item}")
    if failures:
        print(f"Failed on {len(failures)} file(s):")
        for item in failures:
            print(f"  {item}")


if __name__ == "__main__":
    main()
