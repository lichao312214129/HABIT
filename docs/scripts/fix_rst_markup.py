"""Fix reST bold/literal spacing for Chinese HABIT docs (single safe pass)."""
from __future__ import annotations

import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "source"

TARGETS = sorted(SOURCE.rglob("*.rst"))

BOLD_FW = re.compile(r"\*\*([^*]+)\*\*([：，。（；）（])")
BOLD_THEN_CJK = re.compile(r"\*\*([^*]+)\*\*([\u4e00-\u9fff])")
DASH_BOLD = re.compile(r"([-\d\.])(\*\*)")
ARROW_BOLD = re.compile(r"(→)(\*\*)")
BOLD_COLON_LIT = re.compile(r"(\*\*[^*]+\*\* ：)``")
COLON_LIT = re.compile(r"(：)``")
# Adjacent bold segments: **目录**或**路径** → **目录** 或 **路径**
BOLD_CJK_BOLD = re.compile(r"(\*\*[^*]+?\*\*)([\u4e00-\u9fff])(\*\*)")


def fix_line(line: str) -> str:
    if line.lstrip().startswith(".. "):
        return line
    line = BOLD_FW.sub(r"**\1** \2", line)
    line = DASH_BOLD.sub(r"\1 \2", line)
    line = ARROW_BOLD.sub(r"\1 \2", line)
    line = BOLD_THEN_CJK.sub(r"**\1** \2", line)
    line = BOLD_CJK_BOLD.sub(r"\1 \2 \3", line)
    line = BOLD_COLON_LIT.sub(r"\1 ``", line)
    line = COLON_LIT.sub(r"\1 ``", line)
    return line


def main() -> None:
    total = 0
    for path in TARGETS:
        if not path.exists():
            continue
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
