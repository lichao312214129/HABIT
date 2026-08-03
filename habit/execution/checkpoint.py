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

The store tracks SUCCESSES and FAILURES separately: a success is a pickled
result value, a failure is a pickled ``{"key", "message"}`` record. The
split is what makes the v0.1 resume rule -- "failed checkpoint subjects are
skipped unless ``retry_failed_subjects`` or listed in
``force_rerun_subjects``" -- implementable by any backend without leaking
checkpoint policy into the algorithms.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Optional, Tuple, Union

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

    #: Suffix of success entries.
    _SUCCESS_SUFFIX = ".pkl"
    #: Suffix of failure records.
    _FAILURE_SUFFIX = ".failed"

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)

    def _digest(self, key: str) -> str:
        """Return the filesystem-safe digest of an arbitrary cache key."""
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _path_for(self, key: str) -> Path:
        """Map a cache key to its success-entry path."""
        return self.root / f"{self._digest(key)}{self._SUCCESS_SUFFIX}"

    def _failure_path_for(self, key: str) -> Path:
        """Map a cache key to its failure-record path."""
        return self.root / f"{self._digest(key)}{self._FAILURE_SUFFIX}"

    def _atomic_dump(self, payload: Any, path: Path) -> None:
        """Write ``payload`` to ``path`` atomically (write-then-rename)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)

    # ------------------------------------------------------------------
    # Success entries
    # ------------------------------------------------------------------

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

        Storing a success clears any earlier failure record for the same
        key, so a retried subject that finally succeeds resumes cleanly.

        Args:
            key: Cache key produced by the operator's ``cache_key``.
            value: Picklable result value.
        """
        self._atomic_dump(value, self._path_for(key))
        self._failure_path_for(key).unlink(missing_ok=True)

    def contains(self, key: str) -> bool:
        """
        Return whether a success entry exists for ``key``.

        Args:
            key: Cache key produced by the operator's ``cache_key``.
        """
        return self._path_for(key).is_file()

    def __len__(self) -> int:
        """Return the number of stored success entries."""
        if not self.root.is_dir():
            return 0
        return sum(1 for _ in self.root.glob(f"*{self._SUCCESS_SUFFIX}"))

    # ------------------------------------------------------------------
    # Failure records
    # ------------------------------------------------------------------

    def put_failure(self, key: str, message: str) -> None:
        """
        Record that computing ``key`` failed, with a human-readable cause.

        Only the terminal failure should be recorded (after a backend's
        retry rounds are exhausted); intermediate failures of an in-flight
        retry never reach the store.

        Args:
            key: Cache key produced by the operator's ``cache_key``.
            message: Failure description (exception type and text).
        """
        payload = {"key": key, "message": str(message)}
        self._atomic_dump(payload, self._failure_path_for(key))

    def get_failure(self, key: str) -> Optional[str]:
        """
        Return the recorded failure message for ``key``, or ``None``.

        A corrupt failure record is treated as absent (and removed), on the
        same "a checkpoint is a cache" principle as success entries.

        Args:
            key: Cache key produced by the operator's ``cache_key``.
        """
        path = self._failure_path_for(key)
        if not path.is_file():
            return None
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            return str(payload["message"])
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def discard_failure(self, key: str) -> None:
        """
        Remove any failure record for ``key`` (e.g. before a forced rerun).

        Args:
            key: Cache key produced by the operator's ``cache_key``.
        """
        self._failure_path_for(key).unlink(missing_ok=True)

    def failed_keys(self) -> Tuple[str, ...]:
        """
        Return the original cache keys with a recorded failure, sorted.

        Failure payloads embed the original key because the file name is a
        one-way digest; scanning the (small) failure set is the price of
        keeping file names filesystem-safe.
        """
        if not self.root.is_dir():
            return ()
        keys = []
        for path in self.root.glob(f"*{self._FAILURE_SUFFIX}"):
            try:
                with path.open("rb") as handle:
                    payload = pickle.load(handle)
                keys.append(str(payload["key"]))
            except Exception:
                path.unlink(missing_ok=True)
        return tuple(sorted(keys))

    # ------------------------------------------------------------------
    # Whole-store operations
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove every success entry and failure record in the store."""
        if not self.root.is_dir():
            return
        for suffix in (self._SUCCESS_SUFFIX, self._FAILURE_SUFFIX):
            for path in self.root.glob(f"*{suffix}"):
                path.unlink(missing_ok=True)
