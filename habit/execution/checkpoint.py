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
"""Checkpoint persistence for resumable subject-level results.

Kept orthogonal to both algorithms and backends so that resume behaviour is
testable on its own and can be disabled entirely in notebook usage.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Optional, Union

__all__ = ["CheckpointStore"]


class CheckpointStore:
    """
    File-based persistence for subject-level results, keyed by cache key.

    Each entry is one pickle file named by a digest of its key, written
    atomically (write-then-rename) so an interrupted run never leaves a
    half-written checkpoint that a resumed run would mistake for a valid
    result.

    Args:
        root: Directory holding the checkpoint files. Created on first
            write.
    """

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)

    def _path_for(self, key: str) -> Path:
        """Map an arbitrary cache key to a safe, deterministic file path."""
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.pkl"

    def get(self, key: str) -> Optional[Any]:
        """
        Return a previously stored result, or ``None`` when absent.

        A corrupt entry is treated as absent (and removed) rather than
        failing the run: a checkpoint is a cache, never the source of truth.

        Args:
            key: Cache key produced by the operator's ``cache_key``.
        """
        path = self._path_for(key)
        if not path.is_file():
            return None
        try:
            with path.open("rb") as handle:
                return pickle.load(handle)
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def put(self, key: str, value: Any) -> None:
        """
        Store a result under ``key`` atomically.

        Args:
            key: Cache key produced by the operator's ``cache_key``.
            value: Picklable result value.
        """
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
