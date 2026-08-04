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
"""L1 adapters: the only layer (besides recipe writers) allowed to touch the
filesystem.

Each adapter turns one external data convention -- HABIT's own directory
layout, a DataFrame, in-memory arrays, or an nnU-Net dataset -- into the L2
:class:`~habit.contracts.subject.Cohort` contract.
"""

from __future__ import annotations

from habit.adapters.directory import DirectoryDataSource
from habit.adapters.image_refs import FileImageRef
from habit.adapters.writers import DirectoryResultWriter

__all__ = ["DirectoryDataSource", "DirectoryResultWriter", "FileImageRef"]
