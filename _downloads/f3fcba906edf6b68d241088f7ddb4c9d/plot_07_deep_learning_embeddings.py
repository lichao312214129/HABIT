"""
Deep-learning habitat embeddings
================================

Bridge HABIT habitat maps to deep-learning pipelines: extract a binary
mask per habitat and apply **masked spatial average pooling** on a 3-D
feature tensor (simulated here with NumPy; swap in a MONAI / PyTorch
encoder output with the same ``(C, z, y, x)`` layout).
"""

# sphinx_gallery_thumbnail_number = 1

# %%
# Habitat map from HABIT, synthetic 64-channel feature volume from a DL model.
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from habit.contracts import cohort_from_directory
from habit.datasets import fetch_demo
from habit.recipes import one_step_habitat


def masked_spatial_average_pooling(
    feature_map: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Average-pool a 4-D feature tensor over voxels where ``mask > 0``.

    Parameters
    ----------
    feature_map:
        Deep feature volume, shape ``(C, z, y, x)``.
    mask:
        Binary or integer habitat mask, shape ``(z, y, x)``.

    Returns
    -------
    np.ndarray
        Per-channel mean, shape ``(C,)``.
    """
    if feature_map.ndim != 4:
        raise ValueError(
            f"feature_map must be 4-D (C, z, y, x); got shape {feature_map.shape}"
        )
    if mask.shape != feature_map.shape[1:]:
        raise ValueError(
            f"mask shape {mask.shape} must match spatial dims {feature_map.shape[1:]}"
        )
    region = mask > 0
    if not np.any(region):
        return np.full(feature_map.shape[0], np.nan, dtype=np.float64)
    pooled = feature_map[:, region].mean(axis=1)
    return np.asarray(pooled, dtype=np.float64)


def habitat_embedding_table(
    feature_map: np.ndarray,
    label_array: np.ndarray,
    habitat_ids: Tuple[int, ...],
) -> pd.DataFrame:
    """
    Build a table of masked-pooled embeddings, one row per habitat.

    Parameters
    ----------
    feature_map:
        Deep feature volume, shape ``(C, z, y, x)``.
    label_array:
        Integer habitat labels, shape ``(z, y, x)``.
    habitat_ids:
        Habitat ids to embed (model order).

    Returns
    -------
    pd.DataFrame
        Rows indexed by habitat id; columns ``emb_0`` … ``emb_{C-1}``.
    """
    rows: Dict[int, np.ndarray] = {}
    for hid in habitat_ids:
        mask = label_array == hid
        rows[hid] = masked_spatial_average_pooling(feature_map, mask)
    n_channels = int(feature_map.shape[0])
    columns = [f"emb_{i}" for i in range(n_channels)]
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    frame.index.name = "habitat_id"
    return frame


DATA = fetch_demo()
MODALITIES = ("LAP",)
ROI = "LAP"
cohort = cohort_from_directory(DATA, modalities=MODALITIES, roi=ROI)[:1]
result = one_step_habitat(
    modalities=MODALITIES, n_habitats=3, random_seed=0, roi=ROI
).fit_predict(cohort)
habitat_map = result.habitat_maps[0]
labels = np.asarray(habitat_map.label_array)
spatial = labels.shape
rng = np.random.default_rng(0)

# Simulate a 3-D encoder output: 64 channels, same (z, y, x) as the ROI.
n_channels = 64
feature_map = rng.standard_normal((n_channels,) + spatial, dtype=np.float32)

# Optional: torch users can replace the array with ``tensor.detach().cpu().numpy()``.
embeddings = habitat_embedding_table(
    feature_map,
    labels,
    habitat_map.habitat_ids,
)
print(f"Embedding shape: {embeddings.shape} (habitats x {n_channels}-D)")
print(embeddings.iloc[:, :6].round(3))
embeddings.iloc[:, :6]

# %%
# Heatmap of the first 16 embedding dimensions per habitat (English labels only).
Path("out").mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 3))
subset = embeddings.iloc[:, :16].to_numpy()
im = ax.imshow(subset, aspect="auto", cmap="viridis")
ax.set_xlabel("Embedding dimension")
ax.set_ylabel("Habitat id")
ax.set_yticks(range(len(embeddings)))
ax.set_yticklabels([f"H{idx}" for idx in embeddings.index])
ax.set_title("Masked spatial average pooling (64-D, first 16 shown)")
fig.colorbar(im, ax=ax, label="Pooled activation")
fig.savefig("out/dl_habitat_embeddings.png", dpi=150, bbox_inches="tight")
plt.show()
