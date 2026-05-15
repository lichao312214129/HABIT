#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

PROCESSED = Path(r"F:\work\habit_project\demo_data\preprocessed1")
TEMPLATE, COUNT = "subj001", 500

for side in ("images", "masks"):
    src_root = PROCESSED / side / TEMPLATE
    pad = max(3, len(str(COUNT)))
    for i in range(1, COUNT + 1):
        dest_root = PROCESSED / side / f"subj{i:0{pad}d}"
        if dest_root.resolve() == src_root.resolve():
            continue
        for root_str, _dirs, files in os.walk(src_root):
            root = Path(root_str)
            rel = root.relative_to(src_root)
            out = dest_root / rel
            out.mkdir(parents=True, exist_ok=True)
            for name in files:
                shutil.copy2(root / name, out / name)
