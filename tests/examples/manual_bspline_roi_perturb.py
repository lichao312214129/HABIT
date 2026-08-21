#!/usr/bin/env python
# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
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
morphological（形态学膨胀/腐蚀）ROI 轮廓扰动的手动调参脚本。

【场景说明】本脚本模拟的科学场景是"同一张影像、不同医生勾画 ROI"：
只有轮廓边缘允许变化，原始图像完全不参与扰动。因此使用 mask-only 的
morphological 算子（只改掩膜，图像一个体素都不动），而不是 test-retest
场景的 bspline_deform（那个会把图像和轮廓一起歪，属于另一种问题）。
文件名沿用历史名称 manual_bspline_roi_perturb.py 未改，仅为避免牵连其它
引用；实际扰动算子已换成 morphological。

扁平脚本（不定义任何函数）：修改下方"参数块"后重新运行，观察打印的 Dice
与保存的 PNG 叠加图（青色实线 = 原始轮廓，橙红实线 = 扰动后轮廓，放射学
方位），凭肉眼判断轮廓变化是否自然、是否"像另一位医生勾画的"。

直观原理：把 ROI 当成一个实心形状，沿它的边界均匀地向外"长"一层（膨胀）
或向内"削"一层（腐蚀），厚度以毫米计。这对应观察者间一致性的系统分量：
一位医生习惯描得稍大，另一位习惯描得稍小。

本文件不会被 pytest 收集（tests/pytest.ini 中 python_files = test_*.py）。
从仓库根目录用 py310 解释器直接运行：

    E:\\conda\\mconda\\envs\\py310\\python.exe tests/examples/manual_bspline_roi_perturb.py
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from habit import ImagePerturbationRegistry, cohort_from_directory
from habit.viz import plot_intensity_slice, sanitize_label
from habit.viz.orientation import direction_matrix, orient_slice_for_display, resolve_display_geometry

# 注意：不要在这里调用 matplotlib.use("Agg")——Agg 是无头后端，会让末尾的
# plt.show() 弹不出图窗。无显示环境下做验证时，改用环境变量 MPLBACKEND=Agg
# （优先级高于代码外的默认后端），此时 show() 自动变为 no-op。

# =============================================================================
# 参数块 —— 改这里，然后重新运行脚本。
# =============================================================================
# HABIT 预处理数据根目录：其下应有 images/<受试者>/<模态> 与 masks/<受试者>/<ROI>
# 子目录。路径锚定在仓库根目录，因此从别的工作目录启动也能找到数据。
DATA: Path = Path(__file__).resolve().parents[2] / "demo_data" / "preprocessed"
SUBJECT_ID: str = "subj001"  # 要扰动的受试者 id
MODALITY: str = "LAP"        # 解剖图像（强度图）的模态名；只用于显示，不参与扰动
ROI: str = "LAP"             # 肿瘤 ROI 掩膜名
# 随机种子：仅当 GROW_MM=None 时有用，决定本次随机抽到膨胀还是腐蚀、多厚。
SEED: int = 7
# grow_mm：固定的膨胀/腐蚀厚度，单位毫米（物理距离，已按体素间距换算）。
# 正数 = 沿边界向外长（膨胀），负数 = 向内削（腐蚀），0 = 不动。
# 设为 None 时，每次运行从 ±MAX_GROW_MM 之间均匀随机抽一个有符号厚度，
# 更像"随机换一位医生"。
GROW_MM: Optional[float] = 1.0
# max_grow_mm：GROW_MM=None 时的随机采样上限（毫米）。注意毫米数会按体素
# 间距换算成整数次形态学迭代（至少 1 次），demo LAP 体素 1 mm，所以
# 0 < |grow_mm| <= 1.5 都是膨胀/腐蚀 1 体素，效果等价。
MAX_GROW_MM: float = 1.0
# connectivity：形态学结构元的连通性，可选 1/2/3。1 = 6 连通（MIRP 风格
# 默认，最保守）；3 = 26 连通，同样毫米数下动得更多。
CONNECTIVITY: int = 1
OUT_DIR: Path = DATA.parent / "results" / "examples" / "bspline_roi_perturb"  # PNG 输出目录
# =============================================================================

if not DATA.is_dir():
    raise SystemExit(f"DATA directory not found: {DATA}")

# 构造扰动器。mask-only：只替换掩膜，原始图像完全不参与扰动；
# roi=ROI 表示只动这一个掩膜（本队列也只有这一个 ROI）。
deform = ImagePerturbationRegistry.create(
    "morphological", grow_mm=GROW_MM, max_grow_mm=MAX_GROW_MM,
    roi=ROI, connectivity=CONNECTIVITY,
)

# 从目录加载队列，并取出目标受试者（找不到就直接报错退出）。
cohort = cohort_from_directory(DATA, modalities=(MODALITY,), roi=ROI)
if SUBJECT_ID not in list(cohort.subject_ids):
    raise SystemExit(f"Subject {SUBJECT_ID!r} not found in {list(cohort.subject_ids)}")
subject = next(item for item in cohort if item.subject_id == SUBJECT_ID)
image, mask = subject.image(MODALITY), subject.mask(ROI)
mask_arr: np.ndarray = np.asarray(mask.data)
print(f"Subject {subject.subject_id}  grid (z,y,x)={tuple(int(s) for s in mask_arr.shape)}"
      f"  spacing(x,y,z)={tuple(float(v) for v in mask.spacing)}")
print(f"SEED={SEED}  grow_mm={GROW_MM}  max_grow_mm={MAX_GROW_MM}  connectivity={CONNECTIVITY}")

# 施加一次轮廓扰动：只有掩膜被替换，图像体素保持原样。
print("Applying morphological (mask-only) perturbation...", flush=True)
warped_mask_arr: np.ndarray = np.asarray(deform(subject, rng=np.random.default_rng(SEED)).mask(ROI).data)

# 计算新旧掩膜的二值 Dice（前景 = 非零体素）；Dice 越接近 1 说明重叠越好、
# 轮廓差异越小。同时打印前景体素数与异或（不一致）体素数，便于量化对比。
orig_fg: np.ndarray = mask_arr != 0
warp_fg: np.ndarray = warped_mask_arr != 0
n_orig: int = int(np.count_nonzero(orig_fg))
n_warp: int = int(np.count_nonzero(warp_fg))
n_xor: int = int(np.count_nonzero(orig_fg != warp_fg))
dice: float = 1.0 if n_orig + n_warp == 0 else float(2.0 * np.count_nonzero(orig_fg & warp_fg) / (n_orig + n_warp))
print(f"Dice (original vs perturbed ROI): {dice:.4f}")
print(f"Foreground voxels: original={n_orig}  perturbed={n_warp}  xor={n_xor}")

# 画 PNG 叠加图：选原掩膜前景体素最多的那一层轴位切片。
slice_z: int = int(np.argmax(np.sum(orig_fg, axis=(1, 2))))
direction, spacing = resolve_display_geometry(image, mask)
fig = plot_intensity_slice(
    image, roi_mask=mask, axis=0, index=slice_z,
    title=f"Original vs morphological ROI | z={slice_z} (0-based)",
    image_label="Anatomy + ROI contours", display_convention="radiological",
    roi_contour=False, colorbar=True, direction=direction, spacing=spacing,
)
ax = next(axis for axis in fig.axes if axis.images)
matrix = direction_matrix(direction, ndim=3)
# 下面这步方向校正必不可少：demo LAP 的图像与掩膜的 direction 元数据不一致，
# 如果直接 imshow 原始 NumPy 数组，前后（A-P）/左右（L-R）方向会被翻转。
contours = [(orig_fg, "#00E5FF", "Original ROI"), (warp_fg, "#D55E00", "Perturbed ROI")]  # 青色 / 橙红
for foreground, color, _name in contours:
    slice_2d: np.ndarray = orient_slice_for_display(
        np.take(foreground.astype(np.float64), slice_z, axis=0),
        slice_axis=0, direction=matrix, convention="radiological",
    )
    if np.any(slice_2d > 0):
        ax.contour(slice_2d, levels=[0.5], colors=[color], linewidths=0.5,
                   linestyles="solid", origin="upper", extent=ax.images[0].get_extent())
ax.legend(handles=[Line2D([0], [0], color=c, lw=0.5, ls="solid", label=sanitize_label(n))
                   for _f, c, n in contours], loc="lower right", frameon=True, fontsize=8)
OUT_DIR.mkdir(parents=True, exist_ok=True)
png_path: Path = OUT_DIR / f"{SUBJECT_ID}_{ROI}_morph_overlay_z{slice_z:03d}.png"
fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
plt.show()  # 本地交互后端下弹出图窗；MPLBACKEND=Agg 时为 no-op
plt.close(fig)
print(f"Overlay PNG: {png_path}\nDone. Edit GROW_MM / MAX_GROW_MM and re-run.")
