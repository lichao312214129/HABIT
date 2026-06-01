"""Insert spaces so docutils recognizes **strong** markup next to CJK/Latin text."""
from __future__ import annotations

import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "source"

ALLOWED_BEFORE = set(" \t([{<:.,;!?\\-/\\'\"`）】》」』、，。（；")
ALLOWED_AFTER = set(" \t)]}:.,;!?\\-/\\'\"`（【「『《、，。（；）")
HAN = re.compile(r"[\u4e00-\u9fff]")
QUAD_STAR = re.compile(r"\*\*\*\*")
TRIPLE_BOLD = re.compile(r"\*\*\*([^*]+?) \*\*")
STAR_TRIPLE_SUFFIX = re.compile(r"(\*\*[^*\n]+?)\*\*\*+\s*$")
BOLD_AROUND_LIT = re.compile(r"\*\*`([^`]+)`\*\*")


def _is_opening(line: str, pos: int) -> bool:
    if pos + 2 > len(line) or line[pos : pos + 2] != "**":
        return False
    if pos > 0 and line[pos - 1] == "*":
        return False
    before = line[:pos]
    opens = len(re.findall(r"(?<!\*)\*\*(?!\*)", before))
    return opens % 2 == 0


def fix_bold_boundaries(line: str) -> str:
    """Add spaces only at true strong-emphasis open/close boundaries."""
    if line.lstrip().startswith(".. "):
        return line

    chars: list[str] = []
    i = 0
    while i < len(line):
        if line.startswith("**", i):
            opening = _is_opening(line, i)
            if opening:
                if chars and chars[-1] not in ALLOWED_BEFORE:
                    chars.append(" ")
                chars.extend(["*", "*"])
                i += 2
                continue
            # closing
            chars.extend(["*", "*"])
            i += 2
            if i < len(line) and line[i] not in ALLOWED_AFTER:
                chars.append(" ")
            continue
        chars.append(line[i])
        i += 1
    return "".join(chars)


def normalize_inner_bold_spaces(line: str) -> str:
    """Collapse mistaken spaces inside **strong** spans."""
    while True:
        new = re.sub(r"\*\* +([^*]+?) +\*\*", r"**\1**", line)
        new = re.sub(r"\*\* +([^*]+?)\*\*", r"**\1**", new)
        new = re.sub(r"\*\*([^*]+?) +\*\*", r"**\1**", new)
        if new == line:
            return line
        line = new


def fix_line(line: str) -> str:
    if line.lstrip().startswith(".. "):
        return line
    line = QUAD_STAR.sub("**", line)
    line = TRIPLE_BOLD.sub(r"**\1**", line)
    line = STAR_TRIPLE_SUFFIX.sub(r"**\1**", line)
    line = BOLD_AROUND_LIT.sub(r"``\1``", line)
    line = normalize_inner_bold_spaces(line)
    return fix_bold_boundaries(line)


def main() -> None:
    total = 0
    for path in sorted(SOURCE.rglob("*.rst")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        out: list[str] = []
        changed = 0
        for line in lines:
            fixed = fix_line(line.rstrip("\n"))
            if fixed != line.rstrip("\n"):
                changed += 1
            out.append(fixed + ("\n" if line.endswith("\n") else ""))
        if changed:
            path.write_text("".join(out), encoding="utf-8")
            print(f"{path.relative_to(SOURCE)}: {changed}")
            total += changed
    print(f"done, {total} lines")


if __name__ == "__main__":
    main()
