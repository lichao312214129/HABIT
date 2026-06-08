"""
One-off utility to prepend HABIT license headers to all core source files.

Run from the repository root:
    python scripts/add_license_headers.py
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_HABIT_DIR = _ROOT / "habit"

_PY_HEADER = """\
# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
"""

_C_HEADER = """\
/*
 * Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
 *
 * This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
 * Use is governed by the HABIT Software License — see the LICENSE file in the
 * project root for the full text. Summary:
 *
 *   - Non-commercial use (academic, research, education, personal) is permitted
 *     provided that copyright notices are retained and HABIT usage is
 *     acknowledged in publications, reports, or documentation.
 *   - Commercial use requires prior written consent from the copyright holder
 *     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
 *     product documentation or user-facing materials.
 *   - Unauthorized commercial use or removal of attribution is prohibited.
 */
"""

_MARKER = "HABIT Software License"


def _already_has_header(content: str) -> bool:
    """Return True when the file already contains the license header marker."""
    return _MARKER in content[:2000]


def _prepend_header(path: Path, header: str) -> bool:
    """
    Prepend license header to a source file.

    Parameters
    ----------
    path : Path
        Target file path.
    header : str
        Header text to prepend (must end with a newline).

    Returns
    -------
    bool
        True if the file was modified, False if skipped.

    Raises
    ------
    OSError
        When the file cannot be read or written.
    """
    content = path.read_text(encoding="utf-8")
    if _already_has_header(content):
        return False
    path.write_text(header + content, encoding="utf-8", newline="\n")
    return True


def main() -> None:
    """Walk habit/ and add license headers to Python and C/C++ core sources."""
    py_modified = 0
    c_modified = 0
    failures: list[str] = []

    for path in sorted(_HABIT_DIR.rglob("*.py")):
        try:
            if _prepend_header(path, _PY_HEADER):
                py_modified += 1
        except OSError as exc:
            failures.append(f"{path}: {exc}")

    for pattern in ("*.c", "*.h"):
        for path in sorted(_HABIT_DIR.rglob(pattern)):
            try:
                if _prepend_header(path, _C_HEADER + "\n"):
                    c_modified += 1
            except OSError as exc:
                failures.append(f"{path}: {exc}")

    print(f"Updated {py_modified} Python file(s) and {c_modified} C/C++ file(s).")
    if failures:
        print(f"Failed on {len(failures)} file(s):")
        for item in failures:
            print(f"  {item}")


if __name__ == "__main__":
    main()
