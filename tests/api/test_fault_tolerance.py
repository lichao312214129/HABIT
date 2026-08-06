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
"""v1.0 fault-tolerance contracts: geometry, batch radiomics, plugins, cancel.

These tests lock the public resilience behaviours that third-party callers and
CLI users rely on. Process-pool timeout / auto-retry / OOM backoff live in
``tests/execution/``; HabitatModel CompatibilityError lives in
``tests/cloud_gaps/`` and ``tests/contracts/``.
"""

from __future__ import annotations

from typing import Any, Iterator, List
from unittest.mock import patch

import numpy as np
import pytest

from habit.api.exceptions import GeometryError, ProcessingError
from habit.api.plugins import load_plugins
from habit.contracts import Cohort, Subject
from habit.execution import SerialBackend
from habit.image import (
    GeometryPolicy,
    ImageMaskPair,
    ImageVolume,
    MaskVolume,
    align_image_mask,
)
from habit.radiomics import FeatureResult, extract_batch
from habit.utils.job_cancel import (
    JobCancelledError,
    bind_cancel_file,
    clear_cancel_state,
    iter_until_cancelled,
    raise_if_job_cancelled,
    request_job_cancel,
)


def _compatible_pair(*, subject_id: str = "ok") -> ImageMaskPair:
    """Build a geometrically compatible image/mask pair."""
    image = ImageVolume.from_array(
        np.ones((4, 4, 4), dtype=np.float32),
        subject_id=subject_id,
    )
    mask = MaskVolume.from_array(
        np.ones((4, 4, 4), dtype=np.uint8),
        subject_id=subject_id,
    )
    return ImageMaskPair(image, mask)


def _mismatched_pair(*, subject_id: str = "bad") -> ImageMaskPair:
    """Build a pair whose shape/spacing mismatch triggers GeometryError under STRICT."""
    image = ImageVolume.from_array(
        np.ones((4, 4, 4), dtype=np.float32),
        spacing=(1.0, 1.0, 1.0),
        subject_id=subject_id,
    )
    mask = MaskVolume.from_array(
        np.ones((3, 3, 3), dtype=np.uint8),
        spacing=(2.0, 2.0, 2.0),
        subject_id=subject_id,
    )
    return ImageMaskPair(image, mask)


def _fake_feature_result(subject_id: str) -> FeatureResult:
    """Minimal FeatureResult for batch success stubs."""
    from habit.api.image import GeometryReport

    return FeatureResult(
        values={"original_firstorder_Mean": 1.0},
        label=1,
        backend="pyradiomics",
        geometry_report=GeometryReport(compatible=True),
        resolved_params={},
        provenance={"subject_id": subject_id},
    )


@pytest.mark.unit
def test_extract_batch_fail_fast_true_raises_on_first_error() -> None:
    """Default fail_fast=True re-raises the first per-pair exception."""
    with pytest.raises(GeometryError):
        extract_batch([_mismatched_pair(), _compatible_pair()])


@pytest.mark.unit
def test_extract_batch_fail_fast_false_collects_failures() -> None:
    """fail_fast=False keeps successful rows and records failures by subject_id."""
    ok = _compatible_pair(subject_id="ok")
    bad = _mismatched_pair(subject_id="bad")

    def _stub_extract(
        image: Any,
        mask: Any,
        *args: Any,
        **kwargs: Any,
    ) -> FeatureResult:
        sid = getattr(image, "subject_id", None) or "unknown"
        if sid == "bad":
            raise GeometryError("incompatible geometry")
        return _fake_feature_result(str(sid))

    with patch("habit.api.radiomics.extract_features", side_effect=_stub_extract):
        batch = extract_batch([ok, bad], fail_fast=False)

    assert list(batch.table["subject_id"]) == ["ok"]
    assert "bad" in batch.failures
    assert (
        "GeometryError" in batch.failures["bad"]
        or "incompatible" in batch.failures["bad"]
    )
    assert len(batch.results) == 1


@pytest.mark.unit
def test_load_plugins_strict_true_raises_first_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """strict=True aborts plugin discovery on the first broken entry point."""
    from habit.api import plugins

    class BrokenEntryPoint:
        """Minimal entry point whose load() always raises."""

        name = "broken"
        value = "broken_plugin:register"

        @staticmethod
        def load() -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(plugins, "_ENTRY_POINT_GROUPS", {"models": "habit.models"})
    monkeypatch.setattr(
        plugins,
        "_entry_points_for",
        lambda group: (BrokenEntryPoint(),),
    )
    plugins._LOADED_ENTRY_POINTS.clear()

    with pytest.raises(RuntimeError, match="boom"):
        load_plugins(strict=True)


@pytest.mark.unit
def test_cohort_map_aggregates_failures_even_when_backend_continues() -> None:
    """Default Cohort.map raises ProcessingError even when the backend continues.

    Soft failure for recipes / CLI uses ``raise_on_failure=False``.
    """
    cohort = Cohort(
        [
            Subject(subject_id="ok", images={}, masks={}),
            Subject(subject_id="bad", images={}, masks={}),
        ]
    )

    def op(subject: Subject) -> str:
        if subject.subject_id == "bad":
            raise RuntimeError("boom")
        return "fine"

    backend = SerialBackend(on_subject_failure="continue")
    with pytest.raises(ProcessingError, match="1/2 subject"):
        cohort.map(op, backend=backend)

    # Soft-failure path returns SubjectResult slots in cohort order.
    slots = cohort.map(op, backend=backend, raise_on_failure=False)
    assert slots[0].result() == "fine"
    assert isinstance(slots[1].error, RuntimeError)

    # The backend itself still yields both slots when used directly.
    direct = list(backend.map(op, list(cohort)))
    assert direct[0].result() == "fine"
    assert isinstance(direct[1].error, RuntimeError)


@pytest.mark.unit
def test_recipe_soft_map_continues_with_successful_subjects() -> None:
    """Habitat recipes proceed with survivors when some subjects fail."""
    from habit.recipes.habitat import _map_soft

    cohort = Cohort(
        [
            Subject(subject_id="ok", images={}, masks={}),
            Subject(subject_id="bad", images={}, masks={}),
        ]
    )

    def op(subject: Subject) -> str:
        if subject.subject_id == "bad":
            raise RuntimeError("boom")
        return "fine"

    survivors, values, failures = _map_soft(
        cohort,
        op,
        backend=SerialBackend(on_subject_failure="continue"),
        checkpoint=None,
        stage="test",
    )
    assert [s.subject_id for s in survivors] == ["ok"]
    assert values == ["fine"]
    assert "bad" in failures
    assert "RuntimeError" in failures["bad"]


@pytest.mark.unit
def test_geometry_resample_mask_aligns_to_image_grid() -> None:
    """RESAMPLE_MASK regrids the mask onto the image and marks action."""
    pytest.importorskip("SimpleITK")
    image = ImageVolume.from_array(
        np.ones((6, 6, 6), dtype=np.float32),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    mask = MaskVolume.from_array(
        np.ones((3, 3, 3), dtype=np.uint8),
        spacing=(2.0, 2.0, 2.0),
        origin=(0.0, 0.0, 0.0),
    )

    pair = align_image_mask(
        ImageMaskPair(image, mask),
        policy=GeometryPolicy.RESAMPLE_MASK,
    )

    assert pair.geometry_report is not None
    assert pair.geometry_report.compatible
    assert pair.geometry_report.action == GeometryPolicy.RESAMPLE_MASK.value
    assert pair.mask.data.shape == pair.image.data.shape
    assert pair.mask.spacing == pair.image.spacing


@pytest.mark.unit
def test_geometry_resample_image_aligns_to_mask_grid() -> None:
    """RESAMPLE_IMAGE regrids the image onto the mask and marks action."""
    pytest.importorskip("SimpleITK")
    image = ImageVolume.from_array(
        np.ones((6, 6, 6), dtype=np.float32),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    mask = MaskVolume.from_array(
        np.ones((3, 3, 3), dtype=np.uint8),
        spacing=(2.0, 2.0, 2.0),
        origin=(0.0, 0.0, 0.0),
    )

    pair = align_image_mask(
        ImageMaskPair(image, mask),
        policy=GeometryPolicy.RESAMPLE_IMAGE,
    )

    assert pair.geometry_report is not None
    assert pair.geometry_report.compatible
    assert pair.geometry_report.action == GeometryPolicy.RESAMPLE_IMAGE.value
    assert pair.image.data.shape == pair.mask.data.shape
    assert pair.image.spacing == pair.mask.spacing


@pytest.mark.unit
def test_raise_if_job_cancelled_after_request(tmp_path: Any) -> None:
    """GUI cancel flag surfaces as JobCancelledError at cooperative checkpoints."""
    cancel_path = tmp_path / "cancel.flag"
    try:
        bind_cancel_file(cancel_path)
        raise_if_job_cancelled()  # not cancelled yet
        request_job_cancel()
        with pytest.raises(JobCancelledError, match="cancelled"):
            raise_if_job_cancelled()
    finally:
        clear_cancel_state()


@pytest.mark.unit
def test_iter_until_cancelled_stops_mid_stream(tmp_path: Any) -> None:
    """iter_until_cancelled raises after the item that observes the cancel flag."""
    cancel_path = tmp_path / "cancel_iter.flag"
    seen: List[int] = []

    def _source() -> Iterator[int]:
        for value in range(5):
            yield value
            if value == 1:
                request_job_cancel()

    try:
        bind_cancel_file(cancel_path)
        with pytest.raises(JobCancelledError):
            for item in iter_until_cancelled(_source()):
                seen.append(item)
        # Items 0 and 1 are yielded; cancel is checked at the start of the
        # next loop iteration, so 2 never arrives.
        assert seen == [0, 1]
    finally:
        clear_cancel_state()
