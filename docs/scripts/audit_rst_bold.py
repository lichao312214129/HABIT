"""Find reST strong markup with invalid boundary characters (docutils rules)."""
from __future__ import annotations

import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "source"
ALLOWED_BEFORE = set(" \t([{<:.,;!?\\-/\\'\"`")
ALLOWED_AFTER = set(" \t)]}:.,;!?\\-/\\'\"`")


def is_opening_marker(line: str, pos: int) -> bool:
    """True if ``**`` at pos is likely opening strong emphasis."""
    if pos + 2 > len(line) or line[pos : pos + 2] != "**":
        return False
    if pos > 0 and line[pos - 1] == "*":
        return False
    before = line[:pos]
    opens = len(re.findall(r"(?<!\*)\*\*(?!\*)", before))
    return opens % 2 == 0


def audit_line(line: str) -> list[str]:
    if line.lstrip().startswith(".. "):
        return []
    issues: list[str] = []
    pos = 0
    while True:
        idx = line.find("**", pos)
        if idx < 0:
            break
        if is_opening_marker(line, idx):
            if idx > 0:
                prev = line[idx - 1]
                if prev not in ALLOWED_BEFORE:
                    issues.append(f"bad-before-{prev!r}@{idx}")
        else:
            end = idx + 2
            if end < len(line):
                nxt = line[end]
                if nxt not in ALLOWED_AFTER:
                    issues.append(f"bad-after-{nxt!r}@{end}")
        pos = idx + 2
    return issues


def main() -> None:
    rows: list[tuple[str, int, list[str], str]] = []
    for path in sorted(SOURCE.rglob("*.rst")):
        for num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "**" not in line:
                continue
            issues = audit_line(line)
            if issues:
                rows.append((str(path.relative_to(SOURCE)), num, issues, line.strip()[:100]))
    print(f"issues: {len(rows)}")
    for rel, num, issues, preview in rows:
        safe = preview.encode("ascii", "backslashreplace").decode("ascii")
        print(f"{rel}:{num} {issues} | {safe}")


if __name__ == "__main__":
    main()
