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
"""Unit tests for fail-fast write probes and atomic replace helpers."""

from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from habit.exceptions import HABITAPIError
from habit.utils.write_access import (
    is_filesystem_permission_error,
    probe_writable_directory,
    unwritable_destination_message,
    write_via_temp_then_replace,
)


@pytest.fixture
def writable_dir(tmp_path: Path) -> Path:
    """Return an empty writable temporary directory."""
    destination = tmp_path / "out"
    destination.mkdir()
    return destination


@contextmanager
def _make_read_only(path: Path) -> Iterator[None]:
    """
    Clear the write bit on ``path`` for the duration of a ``with`` block.

    Restores owner write permission afterward so tmp cleanup succeeds.
    """
    path.chmod(stat.S_IREAD)
    try:
        yield
    finally:
        path.chmod(stat.S_IREAD | stat.S_IWRITE)


@pytest.mark.unit
def test_probe_writable_directory_ok(writable_dir: Path) -> None:
    """A writable empty directory passes the probe and keeps no leftover probe file."""
    resolved = probe_writable_directory(writable_dir)
    assert resolved == writable_dir.resolve(strict=False)
    leftovers = list(writable_dir.glob(".habit_write_probe_*"))
    assert leftovers == []


@pytest.mark.unit
def test_probe_creates_missing_directory(tmp_path: Path) -> None:
    """Missing ``out_dir`` is created by the probe."""
    destination = tmp_path / "missing" / "nested"
    probe_writable_directory(destination)
    assert destination.is_dir()


@pytest.mark.unit
def test_probe_fails_on_read_only_existing_file(writable_dir: Path) -> None:
    """
    Existing read-only destinations fail fast with path + recovery guidance.

    Args:
        writable_dir: Writable parent directory under pytest's tmp_path.
    """
    target = writable_dir / "subj001_habitats.nrrd"
    target.write_bytes(b"old-bytes")
    with _make_read_only(target):
        with pytest.raises(HABITAPIError) as exc_info:
            probe_writable_directory(writable_dir, existing_paths=[target])
        message = str(exc_info.value)
        assert str(target) in message
        assert "out_dir" in message
        assert "delete" in message.lower() or "rename" in message.lower()
        # Probe must not delete the caller's existing artefact.
        assert target.is_file()
        assert target.read_bytes() == b"old-bytes"


@pytest.mark.unit
def test_probe_directory_permission_error_message(
    writable_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Directory-level PermissionError becomes an actionable HABITAPIError."""
    real_open = open

    def _deny_probe(path: object, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        path_str = str(path)
        if ".habit_write_probe_" in path_str:
            raise PermissionError("simulated denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _deny_probe)
    with pytest.raises(HABITAPIError) as exc_info:
        probe_writable_directory(writable_dir)
    message = str(exc_info.value)
    assert str(writable_dir) in message or writable_dir.name in message
    assert "ACL" in message or "read-only" in message.lower()


@pytest.mark.unit
def test_atomic_replace_leaves_final_file_without_temp(writable_dir: Path) -> None:
    """Successful atomic write leaves the destination and no sibling temp file."""
    destination = writable_dir / "labels.nrrd"
    payload = b"nrrd-bytes-payload"

    def _writer(tmp_path: Path) -> None:
        tmp_path.write_bytes(payload)

    write_via_temp_then_replace(destination, _writer)
    assert destination.read_bytes() == payload
    leaked = [
        path
        for path in writable_dir.iterdir()
        if path != destination and path.name.startswith(".")
    ]
    assert leaked == []


@pytest.mark.unit
def test_atomic_replace_cleans_temp_on_failure(writable_dir: Path) -> None:
    """A failed writer removes the sibling temp and does not create the destination."""
    destination = writable_dir / "labels.nrrd"

    def _writer(tmp_path: Path) -> None:
        tmp_path.write_bytes(b"partial")
        raise RuntimeError("simulated encoder failure")

    with pytest.raises(RuntimeError, match="simulated encoder failure"):
        write_via_temp_then_replace(destination, _writer)
    assert not destination.exists()
    leaked = list(writable_dir.iterdir())
    assert leaked == []


@pytest.mark.unit
def test_atomic_replace_wraps_permission_error(writable_dir: Path) -> None:
    """Permission failures during atomic write become HABITAPIError."""
    destination = writable_dir / "labels.nrrd"

    def _writer(tmp_path: Path) -> None:
        raise PermissionError("Access is denied")

    with pytest.raises(HABITAPIError) as exc_info:
        write_via_temp_then_replace(destination, _writer)
    assert str(destination) in str(exc_info.value)


@pytest.mark.unit
def test_is_filesystem_permission_error_detects_itk_message() -> None:
    """SimpleITK-style RuntimeError messages are treated as permission errors."""
    assert is_filesystem_permission_error(
        RuntimeError("Exception thrown in WriteImage: Permission denied")
    )
    assert not is_filesystem_permission_error(RuntimeError("unsupported format"))


@pytest.mark.unit
def test_unwritable_message_contains_guidance() -> None:
    """The shared message always names the path and recovery options."""
    message = unwritable_destination_message(r"F:\locked\out")
    assert r"F:\locked\out" in message
    assert "out_dir" in message
    assert "Windows" in message


@pytest.mark.unit
def test_writer_habitat_map_atomic_and_permission_wrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    DirectoryResultWriter writes habitat maps atomically and wraps ITK denials.

    Args:
        tmp_path: Pytest temporary root.
        monkeypatch: Used to simulate SimpleITK Permission denied.
    """
    import SimpleITK as sitk

    from habit.adapters.writers import DirectoryResultWriter
    from habit.contracts import Geometry, HabitatMap, Provenance

    habitat_map = HabitatMap(
        subject_id="a",
        label_array=np.ones((2, 2, 2), dtype=np.int32),
        geometry=Geometry.from_array((2, 2, 2)),
        model_id="m",
        habitat_ids=(1,),
        provenance=Provenance.source("test"),
    )
    writer = DirectoryResultWriter(tmp_path / "study")
    path = writer.write_habitat_map(habitat_map)
    assert path is not None
    destination = Path(path)
    assert destination.is_file()
    assert list(destination.parent.glob(".a_habitats*")) == []

    def _deny_write(image: object, filename: str) -> None:
        raise RuntimeError(f"WriteImage failed for {filename}: Permission denied")

    monkeypatch.setattr(sitk, "WriteImage", _deny_write)
    with pytest.raises(HABITAPIError) as exc_info:
        DirectoryResultWriter(tmp_path / "locked").write_habitat_map(habitat_map)
    message = str(exc_info.value)
    assert "Permission denied" in message or "permission denied" in message.lower()
    assert "out_dir" in message


@pytest.mark.unit
def test_study_result_save_probes_before_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ``StudyResult.save`` probes ``out_dir`` before writing artefacts.

    Args:
        tmp_path: Pytest temporary root.
        monkeypatch: Records probe calls.
    """
    from habit.contracts import (
        CohortFingerprint,
        FeatureTable,
        Geometry,
        HabitatMap,
        HabitatModel,
        Provenance,
        RunManifest,
    )
    from habit.recipes import StudyResult
    import pandas as pd

    calls: list[str] = []
    real_probe = probe_writable_directory

    def _spy(directory: object, *, existing_paths: object = None) -> Path:
        calls.append(str(directory))
        return real_probe(directory, existing_paths=existing_paths or [])

    monkeypatch.setattr(
        "habit.utils.write_access.probe_writable_directory", _spy
    )
    monkeypatch.setattr(
        "habit.adapters.writers.probe_writable_directory", _spy
    )

    provenance = Provenance.source("study")
    result = StudyResult(
        habitat_model=HabitatModel(
            model_id="m",
            n_habitats=1,
            feature_names=("f1",),
            centroids=np.zeros((1, 1)),
            preprocessing_state={},
            spec_payload={},
            cohort_fingerprint=CohortFingerprint(
                n_subjects=1, modalities=("T1",), subject_id_digest="d"
            ),
            provenance=provenance,
        ),
        pipeline=object(),
        features=FeatureTable(
            frame=pd.DataFrame({"subject": ["a"], "f1": [1.0]}),
            id_columns=("subject",),
            feature_columns=("f1",),
            provenance=provenance,
        ),
        habitat_maps=(
            HabitatMap(
                subject_id="a",
                label_array=np.ones((2, 2, 2), dtype=np.int32),
                geometry=Geometry.from_array((2, 2, 2)),
                model_id="m",
                habitat_ids=(1,),
                provenance=provenance,
            ),
        ),
        manifest=RunManifest(
            spec_payload={"design": "two_step"},
            provenance=provenance,
            subject_outcomes={"a": "success"},
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:01:00Z",
        ),
    )
    out = tmp_path / "study_out"
    result.save(out)
    assert calls, "StudyResult.save must probe write access"
    assert (out / "a_habitats.nrrd").is_file()


@pytest.mark.unit
@pytest.mark.skipif(os.name != "nt", reason="Windows read-only attribute semantics")
def test_probe_read_only_file_windows(writable_dir: Path) -> None:
    """On Windows, clearing the write bit makes overwrite probe fail."""
    target = writable_dir / "locked.nrrd"
    target.write_bytes(b"x")
    with _make_read_only(target):
        with pytest.raises(HABITAPIError):
            probe_writable_directory(writable_dir, existing_paths=[target])
