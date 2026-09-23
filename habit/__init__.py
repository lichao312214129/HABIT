# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""HABIT (Habitat Analysis: Biomedical Imaging Toolkit).

Habitat imaging partitions a tumor into subregions (habitats) on voxels,
then quantifies spatial heterogeneity. Import a capability from its own
package, for example ``from habit.voxel_features import RawVoxelFeatures``.
The root package exports only version metadata in v2.
"""

from habit._version import __version__

__all__ = ["__version__"]
