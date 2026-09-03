# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""HABIT package metadata and capability-package discovery.

Import public components from their capability package, for example
``from habit.voxel_features import RawVoxelFeatures``.  The root package
intentionally exports only version metadata in v2.
"""

from habit._version import __version__

__all__ = ["__version__"]
