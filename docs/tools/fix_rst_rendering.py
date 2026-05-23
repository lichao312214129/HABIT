"""Conservative RST rendering fixes for HABIT docs."""

from __future__ import annotations

import re
from pathlib import Path


DOCS_ROOT = Path(__file__).resolve().parents[1] / "source"

TIGHT_LITERAL = (
    re.compile(r"``([^`\n]+)``([（(])"),
    re.compile(r"``([^`\n]+)``([、，])"),
    re.compile(r"``([^`\n]+)``([：:])"),
)
INLINE_LINK = re.compile(r"`([^<`\n]+)\s*<(https?://[^>]+)>`__")
DOC_TIGHT = re.compile(r"(:doc:`[^`]+`)([「『《])")
CODE_DOC = re.compile(r"(``[^`]+``)([（(]:doc:`)")
CLOSED_BOLD_LITERAL = re.compile(r"\*\*([^*\n]+?)\*\*``")
OPEN_BOLD_LITERAL = re.compile(r"\*\*``([^`]+)``([^*]{0,24}?)\*\*")

BOLD_LITERAL_REPLACEMENTS = {
    "**``type_of_transform``、``metric``、``optimizer`` 仅对 ``ants`` 与 ``simpleitk`` 有意义**": (
        "``type_of_transform``、``metric``、``optimizer`` 仅对 ``ants`` 与 ``simpleitk`` 有意义"
    ),
    "**``ants`` / ``simpleitk`` / ``elastix``**": "``ants`` / ``simpleitk`` / ``elastix``",
    "**``elastix``**": "``elastix``",
    "**``backend: elastix`` 时会被接收但不传入 elastix 可执行文件**": (
        "``backend: elastix`` 时会被接收但不传入 elastix 可执行文件"
    ),
    "**``.txt`` 模板宜按模态与任务从 Model Zoo 选取**": (
        "``.txt`` 模板宜按模态与任务从 Model Zoo 选取"
    ),
    "**``type_of_transform`` / ``metric`` / ``optimizer``**": (
        "``type_of_transform`` / ``metric`` / ``optimizer``"
    ),
    "**``elastix`` 专属**": "**elastix 专属**",
    "**``habit dicom-info`` 主要参数**": "**habit dicom-info 主要参数** (``habit dicom-info``)",
}


def fix_line(line: str) -> str:
    if line.strip().startswith(".. ") or ".. code-block::" in line:
        return line

    updated = line
    for old, new in BOLD_LITERAL_REPLACEMENTS.items():
        updated = updated.replace(old, new)

    for pattern in TIGHT_LITERAL:
        updated = pattern.sub(r"``\1`` \2", updated)

    updated = INLINE_LINK.sub(r"`\1 <\2>`_", updated)
    updated = DOC_TIGHT.sub(r"\1 \2", updated)
    updated = CODE_DOC.sub(r"\1 \2", updated)
    updated = CLOSED_BOLD_LITERAL.sub(r"**\1** ``", updated)
    updated = OPEN_BOLD_LITERAL.sub(r"**\2** (``\1``)", updated)
    return updated


def process_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    lines = [fix_line(line) for line in original.splitlines(keepends=True)]
    updated = "".join(lines)
    if updated != original:
        path.write_text(updated, encoding="utf-8", newline="\n")
        return True
    return False


def main() -> None:
    changed = [p for p in sorted(DOCS_ROOT.rglob("*.rst")) if process_file(p)]
    print(f"Updated {len(changed)} file(s)")


if __name__ == "__main__":
    main()
