#!/usr/bin/env python3
"""
Normalize *.sh files to Unix LF line endings (strip CR).

Run before bash scripts on WSL/Linux if you see: $'\\r': command not found

Usage:
    python developer/fix_sh_line_endings.py
    python developer/fix_sh_line_endings.py developer docker
"""

from __future__ import annotations

import sys
from pathlib import Path


def normalize_lf(path: Path) -> bool:
    """
    Rewrite file with LF-only newlines.

    Args:
        path: Target shell script path.

    Returns:
        bool: True if the file was modified.
    """
    raw: bytes = path.read_bytes()
    try:
        text: str = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    fixed: str = text.replace("\r\n", "\n").replace("\r", "\n")
    if fixed == text:
        return False
    path.write_bytes(fixed.encode("utf-8"))
    return True


def collect_sh_files(roots: list[Path]) -> list[Path]:
    """
    Collect *.sh under given directories (non-recursive permission errors skipped).

    Args:
        roots: Directory paths to scan.

    Returns:
        list[Path]: Sorted unique shell script paths.
    """
    found: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.sh"):
            if path.is_file():
                found.add(path.resolve())
    return sorted(found)


def main(argv: list[str]) -> int:
    """
    Entry point.

    Args:
        argv: CLI args; optional directory roots (default: developer, docker).

    Returns:
        int: Exit code 0 on success.
    """
    repo_root: Path = Path(__file__).resolve().parent.parent
    if len(argv) > 1:
        roots = [(repo_root / arg).resolve() for arg in argv[1:]]
    else:
        roots = [repo_root / "developer", repo_root / "docker"]

    changed: list[Path] = []
    for sh_path in collect_sh_files(roots):
        try:
            if normalize_lf(sh_path):
                changed.append(sh_path)
        except OSError as exc:
            print(f"[skip] {sh_path}: {exc}", file=sys.stderr)

    if changed:
        print(f"[fix_sh_line_endings] Normalized {len(changed)} file(s) to LF:")
        for path in changed:
            print(f"  {path.relative_to(repo_root)}")
    else:
        print("[fix_sh_line_endings] All scanned *.sh files already use LF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
