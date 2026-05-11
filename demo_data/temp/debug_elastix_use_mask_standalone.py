# -*- coding: utf-8 -*-
"""Minimal elastix: no mask vs mask. Edit ROOT/SUBJ if needed."""

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(r"F:\work\habit_project\demo_data\preprocessed\resample_02")
SUBJ = "subj001"
FIXED, MOVING = "delay2", "delay3"
PARAM = Path(r"F:\work\habit_project\demo_data\Par0040affine.txt")

def vol(d: Path):
    return next(d.glob("*.nii*"))

f, m = vol(ROOT / "images" / SUBJ / FIXED), vol(ROOT / "images" / SUBJ / MOVING)
f_mask, m_mask = vol(ROOT / "masks" / SUBJ / FIXED), vol(ROOT / "masks" / SUBJ / MOVING)

out0, out1 = Path(tempfile.mkdtemp(prefix="elx_")), Path(tempfile.mkdtemp(prefix="elxm_"))

cmd0 = ["elastix", "-f", str(f), "-m", str(m), "-out", str(out0), "-p", str(PARAM)]
cmd1 = cmd0[:-4] + ["-fMask", str(f_mask), "-mMask", str(m_mask), "-out", str(out1), "-p", str(PARAM)]

print("no mask:", subprocess.run(cmd0).returncode, out0)
print("with mask:", subprocess.run(cmd1).returncode, out1)
