"""Build one page and assert no raw ** in body paragraphs."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
OUT = DOCS / "build_bold_check"
PAGES = [
    "getting_started/installation_zh.html",
    "configuration_zh.html",
    "design_philosophy_zh.html",
]


def main() -> int:
    subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", "-q", "source", str(OUT)],
        cwd=DOCS,
        check=True,
    )
    bad = 0
    for rel in PAGES:
        html = (OUT / rel).read_text(encoding="utf-8", errors="replace")
        # Raw ** in paragraph text (not in code/script).
        for m in re.finditer(r"<p>[^<]*\*\*[^<]*</p>", html):
            bad += 1
            print(f"{rel}: {m.group(0)[:120]}")
    print("raw ** in <p>:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
