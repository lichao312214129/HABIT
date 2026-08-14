#!/usr/bin/env python
"""
Demonstrate v1.0 fault-tolerance entry points on synthetic data.

Covers geometry policies, batch radiomics fail_fast, plugin load reports,
backend continue vs Cohort.map aggregation, and HabitatModel CompatibilityError.

This script accompanies ``docs/source/examples/fault_tolerance.rst``.

Run from the repository root::

    python docs/source/examples/scripts/fault_tolerance_demo.py
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np

from habit import (
    CompatibilityError,
    FeatureResult,
    GeometryError,
    GeometryPolicy,
    GeometryReport,
    ImageMaskPair,
    ImageVolume,
    MaskVolume,
    ProcessingError,
    align_image_mask,
    extract_batch,
    load_plugins,
)
from habit.contracts import Cohort, Subject
from habit.execution import SerialBackend


def _pair(shape: tuple[int, ...], spacing: tuple[float, ...], sid: str) -> ImageMaskPair:
    """Build an image/mask pair with the given grid and subject id."""
    image = ImageVolume.from_array(
        np.ones(shape, dtype=np.float32),
        spacing=spacing,
        subject_id=sid,
    )
    mask = MaskVolume.from_array(
        np.ones(shape, dtype=np.uint8),
        spacing=spacing,
        subject_id=sid,
    )
    return ImageMaskPair(image, mask)


def demo_geometry_policies() -> None:
    """Show STRICT raise vs RESAMPLE_MASK correction."""
    image = ImageVolume.from_array(
        np.ones((6, 6, 6), dtype=np.float32),
        spacing=(1.0, 1.0, 1.0),
    )
    mask = MaskVolume.from_array(
        np.ones((3, 3, 3), dtype=np.uint8),
        spacing=(2.0, 2.0, 2.0),
    )
    pair = ImageMaskPair(image, mask)

    try:
        align_image_mask(pair, policy=GeometryPolicy.STRICT)
        print("STRICT: unexpectedly compatible")
    except GeometryError as exc:
        print(f"STRICT: GeometryError - {exc}")

    aligned = align_image_mask(pair, policy=GeometryPolicy.RESAMPLE_MASK)
    report = aligned.geometry_report
    assert report is not None
    print(
        f"RESAMPLE_MASK: action={report.action!r}, "
        f"mask_shape={aligned.mask.data.shape}, compatible={report.compatible}"
    )
    from pathlib import Path

    from habit.viz import plot_habitat_overlay

    fig = plot_habitat_overlay(
        aligned.image.data,
        aligned.mask.data.astype(np.int32),
        axis=0,
        title="RESAMPLE_MASK: aligned ROI on image grid",
    )
    Path("out").mkdir(exist_ok=True)
    fig.savefig("out/fault_tolerance_align.png", dpi=150, bbox_inches="tight")
    print("Wrote out/fault_tolerance_align.png")
    return fig


def demo_extract_batch_fail_fast() -> None:
    """Contrast fail_fast=True (raise) with fail_fast=False (failures map)."""
    ok = _pair((4, 4, 4), (1.0, 1.0, 1.0), "ok")
    bad_image = ImageVolume.from_array(
        np.ones((4, 4, 4), dtype=np.float32),
        spacing=(1.0, 1.0, 1.0),
        subject_id="bad",
    )
    bad_mask = MaskVolume.from_array(
        np.ones((3, 3, 3), dtype=np.uint8),
        spacing=(2.0, 2.0, 2.0),
        subject_id="bad",
    )
    bad = ImageMaskPair(bad_image, bad_mask)

    def _stub(
        image: ImageVolume,
        mask: MaskVolume,
        *args: object,
        **kwargs: object,
    ) -> FeatureResult:
        sid = image.subject_id or "unknown"
        if sid == "bad":
            raise GeometryError("incompatible geometry")
        return FeatureResult(
            values={"original_firstorder_Mean": 1.0},
            label=1,
            backend="demo",
            geometry_report=GeometryReport(compatible=True),
            resolved_params={},
        )

    with patch("habit.api.radiomics.extract_features", side_effect=_stub):
        try:
            extract_batch([ok, bad], fail_fast=True)
        except GeometryError as exc:
            print(f"extract_batch(fail_fast=True): raised {type(exc).__name__}")

        batch = extract_batch([ok, bad], fail_fast=False)
        print(
            f"extract_batch(fail_fast=False): rows={len(batch.table)}, "
            f"failures={dict(batch.failures)}"
        )


def demo_plugin_load_report() -> None:
    """Default load_plugins returns a report (strict=False swallows EP errors)."""
    report = load_plugins(strict=False)
    print(
        f"load_plugins(strict=False): loaded={len(report.loaded)}, "
        f"failures={len(report.failures)}"
    )


def demo_backend_continue_vs_cohort_map() -> None:
    """Backend continue yields error slots; Cohort.map still aggregates to raise."""
    subjects: List[Subject] = [
        Subject(subject_id="ok", images={}, masks={}),
        Subject(subject_id="bad", images={}, masks={}),
    ]
    cohort = Cohort(subjects)

    def op(subject: Subject) -> str:
        if subject.subject_id == "bad":
            raise RuntimeError("boom")
        return "fine"

    backend = SerialBackend(on_subject_failure="continue")
    slots = list(backend.map(op, subjects))
    print(
        f"backend.map(continue): ok={slots[0].result()!r}, "
        f"bad_error={type(slots[1].error).__name__}"
    )

    try:
        cohort.map(op, backend=backend)
    except ProcessingError as exc:
        print(f"cohort.map(...): ProcessingError - {exc}")

    soft = cohort.map(op, backend=backend, raise_on_failure=False)
    print(
        f"cohort.map(raise_on_failure=False): ok={soft[0].result()!r}, "
        f"bad_error={type(soft[1].error).__name__}"
    )


def demo_habitatmodel_compatibility_error() -> None:
    """A truncated ZIP is rejected with CompatibilityError (never a silent map)."""
    from habit.contracts import HabitatModel

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "broken.habitatmodel"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("manifest.json", '{"format": "not-a-habitat-model"}')
        try:
            HabitatModel.load(path)
        except CompatibilityError as exc:
            print(f"HabitatModel.load(bad): CompatibilityError - {exc}")


def main() -> None:
    """Run all fault-tolerance demos and print English status lines."""
    print("=== GeometryPolicy ===")
    fig = demo_geometry_policies()
    print("\n=== extract_batch fail_fast ===")
    demo_extract_batch_fail_fast()
    print("\n=== PluginLoadReport ===")
    demo_plugin_load_report()
    print("\n=== Backend continue vs Cohort.map ===")
    demo_backend_continue_vs_cohort_map()
    print("\n=== HabitatModel CompatibilityError ===")
    demo_habitatmodel_compatibility_error()
    print("\nDone. See docs/source/examples/fault_tolerance.rst")
    return fig


if __name__ == "__main__":
    import sys
    from pathlib import Path

    fig = main()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _example_roi import save_example_figure

    save_example_figure(fig, "fault_tolerance_align.png")
