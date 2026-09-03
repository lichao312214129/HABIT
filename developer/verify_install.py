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
Reusable verification harness for an installed ``habitat-analysis`` package.

The script proves that HABIT is not merely importable but functional: it
exercises every registered public API symbol, every CLI subcommand entry
point (``python -m habit <cmd> --help``), optional-dependency error paths,
and (at ``--level full``) minimal end-to-end habitat, feature-extraction,
and machine-learning workflows on local demo data.

Designed to run inside a bare virtual environment that only has
``habitat-analysis`` installed — no pytest or dev test suite required.

Example::

    python developer/verify_install.py --level smoke --out .tmp_verify/smoke
    python developer/verify_install.py --level full --out .tmp_verify/full \\
        --demo-data demo_data
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Result bookkeeping
# ---------------------------------------------------------------------------


class CheckStatus(str, Enum):
    """Outcome of a single verification check."""

    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class CheckRecord:
    """
    One atomic verification result.

    Attributes:
        group: Logical bucket (``import``, ``public_api``, ``cli_help``, …).
        name: Human-readable check identifier within the group.
        status: PASS, FAIL, or SKIP.
        seconds: Wall-clock duration of this check.
        reason: Explanation for SKIP or FAIL (empty on PASS).
    """

    group: str
    name: str
    status: CheckStatus
    seconds: float
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for ``results.json``."""
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass
class HarnessReport:
    """Aggregate outcome written to ``results.json`` and printed as a table."""

    level: str
    habit_version: str
    total_seconds: float
    checks: List[CheckRecord] = field(default_factory=list)

    @property
    def summary(self) -> Dict[str, int]:
        """Count checks by status."""
        counts = {status.value: 0 for status in CheckStatus}
        for record in self.checks:
            counts[record.status.value] += 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable report payload."""
        return {
            "level": self.level,
            "habit_version": self.habit_version,
            "total_seconds": round(self.total_seconds, 3),
            "summary": self.summary,
            "checks": [item.to_dict() for item in self.checks],
        }


class CheckRunner:
    """
    Collects timed check results and enforces uniform error handling.

    Each ``run_*`` helper wraps a callable, records elapsed time, and stores
    PASS/FAIL/SKIP without aborting the overall harness (unless the caller
    chooses to stop early — we never do).
    """

    def __init__(self) -> None:
        self.records: List[CheckRecord] = []

    def add_skip(
        self,
        group: str,
        name: str,
        reason: str,
        *,
        seconds: float = 0.0,
    ) -> None:
        """Record an explicit SKIP without executing work."""
        self.records.append(
            CheckRecord(group, name, CheckStatus.SKIP, seconds, reason=reason)
        )

    def run(
        self,
        group: str,
        name: str,
        func: Callable[[], None],
    ) -> None:
        """
        Execute ``func`` and record PASS or FAIL.

        Args:
            group: Logical bucket for reporting.
            name: Check label.
            func: Zero-argument callable that raises on failure.
        """
        started = time.perf_counter()
        reason = ""
        status = CheckStatus.PASS
        try:
            func()
        except SkipCheck as exc:
            status = CheckStatus.SKIP
            reason = str(exc)
        except Exception as exc:  # noqa: BLE001 — harness must continue
            status = CheckStatus.FAIL
            reason = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        self.records.append(CheckRecord(group, name, status, elapsed, reason=reason))

    def extend(self, other: "CheckRunner") -> None:
        """Merge records produced by a nested runner."""
        self.records.extend(other.records)


class SkipCheck(Exception):
    """Raised inside a check callable to mark an intentional SKIP."""


# ---------------------------------------------------------------------------
# Authoritative CLI inventory (derived from habit/cli.py)
# ---------------------------------------------------------------------------

#: Every Click subcommand registered on the root ``cli`` group.
#: ``required_options`` lists options that must be supplied for a real run;
#: smoke level only invokes ``--help``, which needs no options.
CLI_SUBCOMMANDS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "check-config",
        "required_options": ("--config",),
        "optional_options": ("--workflow", "--syntax-only"),
    },
    {
        "name": "migrate-config",
        "required_options": ("--config",),
        "optional_options": ("--output", "--dry-run", "--workflow"),
    },
    {
        "name": "preprocess",
        "required_options": ("--config",),
        "optional_options": (),
    },
    {
        "name": "sort-dicom",
        "required_options": ("--config",),
        "optional_options": (),
    },
    {
        "name": "get-habitat",
        "required_options": ("--config",),
        "optional_options": ("--mode", "--pipeline", "--debug", "--resume"),
    },
    {
        "name": "extract",
        "required_options": ("--config",),
        "optional_options": (),
    },
    {
        "name": "model",
        "required_options": ("--config",),
        "optional_options": ("--mode",),
    },
    {
        "name": "cv",
        "required_options": ("--config",),
        "optional_options": (),
    },
    {
        "name": "compare",
        "required_options": ("--config",),
        "optional_options": (),
    },
    {
        "name": "icc",
        "required_options": ("--config",),
        "optional_options": (),
    },
    {
        "name": "radiomics",
        "required_options": ("--config",),
        "optional_options": (),
    },
    {
        "name": "dice",
        "required_options": ("--input1", "--input2"),
        "optional_options": ("--output", "--mask-keyword", "--label-id"),
    },
    {
        "name": "dicom-info",
        "required_options": ("--input",),
        "optional_options": (
            "--tags",
            "--output",
            "--format",
            "--recursive",
            "--list-tags",
            "--num-samples",
            "--group-by-series",
            "--one-file-per-folder",
            "--dicom-extensions",
            "--include-no-extension",
            "--num-workers",
            "--max-depth",
        ),
    },
    {
        "name": "merge-csv",
        "required_options": ("--output",),
        "optional_options": ("--index-col", "--separator", "--encoding", "--join"),
        "positional": "two or more input CSV/Excel files",
    },
)


# ---------------------------------------------------------------------------
# Optional-extra probes (smoke)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptionalExtraProbe:
    """
    Describe how to verify the absence error path for one optional extra.

    Attributes:
        extra_name: PyPI extra label from ``pyproject.toml`` (informational).
        module_name: Top-level import name checked via ``habit.is_available``.
        description: Short label for reporting.
        trigger: Callable that must raise ``OptionalDependencyError`` when
            ``module_name`` is not importable.
    """

    extra_name: str
    module_name: str
    description: str
    trigger: Callable[[], None]


def _build_optional_probes() -> Tuple[OptionalExtraProbe, ...]:
    """
    Construct optional-dependency probes lazily so ``import habit`` happens
    after the runner starts (keeping import timing visible in reports).
    """
    from habit.api.utils import is_available
    from habit.exceptions import OptionalDependencyError
    from habit.viz import plot_habitat_clustering_pca_3d_interactive

    probes: List[OptionalExtraProbe] = []

    def _probe(
        extra_name: str,
        module_name: str,
        description: str,
        trigger: Callable[[], None],
    ) -> None:
        probes.append(
            OptionalExtraProbe(extra_name, module_name, description, trigger)
        )

    # radiomics extra -------------------------------------------------------
    def _trigger_radiomics() -> None:
        from habit.utils.optional_deps import require_pyradiomics

        require_pyradiomics()

    _probe("radiomics", "radiomics", "PyRadiomics (radiomics extra)", _trigger_radiomics)

    # registration extra ----------------------------------------------------
    def _trigger_ants() -> None:
        import numpy as np
        import SimpleITK as sitk

        from habit.utils.image_converter import ImageConverter

        # itk_2_ants is the earliest antspyx touchpoint with OptionalDependencyError.
        ImageConverter.itk_2_ants(sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.float32)))

    _probe("registration", "ants", "ANTsPy registration backend", _trigger_ants)

    # ml extra — XGBoost classifier -----------------------------------------
    def _trigger_xgboost() -> None:
        from habit.classification.models import XgboostClassifier

        XgboostClassifier()._build_estimator()

    _probe("ml", "xgboost", "XGBoost classifier (ml extra)", _trigger_xgboost)

    # ml extra — mRMR selector ----------------------------------------------
    def _trigger_mrmr() -> None:
        from habit.feature_selection.selectors import MrmrSelector

        MrmrSelector(n_features=3).fit(None)  # type: ignore[arg-type]

    _probe("ml", "mrmr", "mRMR feature selector (ml extra)", _trigger_mrmr)

    # analysis extra — pingouin ICC -----------------------------------------
    def _trigger_pingouin() -> None:
        from habit.evaluation.reliability import _require_pingouin

        _require_pingouin()

    _probe("analysis", "pingouin", "Pingouin ICC (analysis extra)", _trigger_pingouin)

    # analysis extra — plotly interactive PCA -------------------------------
    def _trigger_plotly() -> None:
        import numpy as np

        plot_habitat_clustering_pca_3d_interactive(
            features=np.zeros((4, 2), dtype=np.float64),
            labels=np.array([1, 1, 2, 2], dtype=np.int64),
            centers=np.zeros((2, 2), dtype=np.float64),
            n_clusters=2,
        )

    _probe("analysis", "plotly", "Plotly interactive PCA (analysis extra)", _trigger_plotly)

    # automl extra ----------------------------------------------------------
    def _trigger_autogluon() -> None:
        from habit.classification.autogluon import _lazy_tabular_predictor

        _lazy_tabular_predictor()

    _probe("automl", "autogluon", "AutoGluon classifier (automl extra)", _trigger_autogluon)

    # torch extra — ImageConverter raises ImportError (not OptionalDependencyError).
    def _trigger_torch() -> None:
        import numpy as np

        from habit.utils.image_converter import ImageConverter

        ImageConverter.numpy_to_tensor(np.zeros((4, 4, 4), dtype=np.float32))

    _probe("torch", "torch", "Torch tensor conversion (torch extra)", _trigger_torch)

    return tuple(probes)


# ---------------------------------------------------------------------------
# Public API verification helpers
# ---------------------------------------------------------------------------


def _symbol_is_usable(obj: Any) -> bool:
    """
    Return whether a registry symbol is callable or meaningfully inspectable.

    Constants (``MAXIMIZE``), classes, functions, and exception types all
    qualify; ``None`` does not.
    """
    if obj is None:
        return False
    if callable(obj):
        return True
    if inspect.isclass(obj):
        return True
    # Module-level constants such as SCORE_DIRECTIONS (dict) or direction strs.
    return isinstance(obj, (str, dict, tuple, list, int, float, bool))


def _verify_public_symbol(namespace: str, symbol: str) -> None:
    """
    Resolve one v2 capability symbol through its canonical package.

    Args:
        namespace: Canonical module path (e.g. ``"habit.voxel_features"``).
        symbol: Name listed in ``PUBLIC_NAMESPACES[namespace]``.

    Raises:
        AssertionError: When the symbol is missing or not usable.
        SkipCheck: Never — import failures are real FAILs.
    """
    package = importlib.import_module(namespace)
    obj = getattr(package, symbol)
    assert obj is not None, f"{namespace}.{symbol} resolved to None"
    assert _symbol_is_usable(obj), (
        f"{namespace}.{symbol} is neither callable nor a recognised "
        f"constant/type ({type(obj)!r})"
    )


# ---------------------------------------------------------------------------
# Demo-data layout helpers (full level)
# ---------------------------------------------------------------------------

#: DCE-MRI modality keys in the bundled demo cohort.
_DEMO_MODALITIES: Tuple[str, ...] = ("pre_contrast", "LAP", "PVP", "delay_3min")

#: ROI mask key in the demo layout (prefer arterial phase for overlay/ROI).
_DEMO_ROI: str = "LAP"

#: Fixed habitat count for fast, deterministic full-level habitat runs.
_FULL_N_HABITATS: int = 3


def resolve_demo_data(explicit: Optional[str]) -> Optional[Path]:
    """
    Locate the demo-data root directory.

    Resolution order:
    1. ``--demo-data`` when provided and valid.
    2. ``<repo>/demo_data`` relative to this script (source checkout).
    3. ``None`` when no suitable tree exists (full checks SKIP).

    Args:
        explicit: Optional path from CLI ``--demo-data``.

    Returns:
        Resolved demo-data directory or ``None``.
    """
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.is_dir():
            return candidate
        return None

    script_root = Path(__file__).resolve().parent.parent
    bundled = script_root / "demo_data"
    imaging = bundled / "preprocessed"
    if bundled.is_dir() and (imaging / "images").is_dir():
        return bundled
    return None


def resolve_config_root(demo_root: Optional[Path]) -> Optional[Path]:
    """
    Locate shipped YAML configs for optional CLI full checks.

    When running from a source checkout the ``config/`` tree lives next to
    ``demo_data/``; installed wheels do not ship configs.

    Args:
        demo_root: Resolved demo-data directory (may be ``None``).

    Returns:
        Config directory or ``None``.
    """
    if demo_root is not None:
        candidate = demo_root.parent / "config"
        if candidate.is_dir():
            return candidate
    script_root = Path(__file__).resolve().parent.parent
    candidate = script_root / "config"
    return candidate if candidate.is_dir() else None


def demo_imaging_root(demo_root: Path) -> Path:
    """Return the preprocessed imaging root inside demo data."""
    return demo_root / "preprocessed"


def demo_ml_csv(demo_root: Path) -> Path:
    """Return the primary radiomics CSV used for minimal ML verification."""
    return demo_root / "ml_data" / "breast_cancer_dataset.csv"


def _load_demo_cohort(demo_root: Path) -> Any:
    """
    Build the bundled demo cohort through the public API.

    Args:
        demo_root: Demo-data root directory.

    Returns:
        :class:`~habit.contracts.subject.Cohort` with lazy file references.
    """
    from habit.contracts.subject import cohort_from_directory

    return cohort_from_directory(
        demo_imaging_root(demo_root),
        modalities=_DEMO_MODALITIES,
        roi=_DEMO_ROI,
        name="verify_install_demo",
    )


def _build_fast_habitat_spec(*, two_step: bool) -> Any:
    """
    Minimal :class:`~habit.spec.specs.HabitatSpec` for full-level runs.

    Uses a **fixed** habitat count (no elbow/silhouette search) and a small
    supervoxel budget so runtime stays bounded on real demo volumes.

    Args:
        two_step: When ``True``, include a k-means supervoxel stage; when
            ``False``, direct-pooling mode (``supervoxelizer=None``).

    Returns:
        Fully wired habitat specification object.
    """
    from habit.spec import HabitatSpec, Spec

    return HabitatSpec(
        name="verify_install_habitat",
        voxel_feature_extractor=Spec(
            name="raw",
            params={"modalities": list(_DEMO_MODALITIES)},
        ),
        supervoxelizer=(
            Spec(
                name="kmeans",
                params={"n_supervoxels": 20, "max_iter": 100, "n_init": 3},
            )
            if two_step
            else None
        ),
        habitat_model_fitter=Spec(
            name="kmeans",
            params={"n_habitats": _FULL_N_HABITATS, "n_init": 3},
        ),
        habitat_assigner=Spec(name="nearest_centroid"),
        habitat_features=(Spec(name="volume"), Spec(name="msi")),
        random_seed=42,
    )


def _build_fast_ml_spec() -> Any:
    """
    Minimal tabular ML spec (z-score + correlation + logistic regression).

    Returns:
        :class:`~habit.spec.specs.MLSpec` without optional ml-extra selectors.
    """
    from habit.spec import MLSpec, Spec

    return MLSpec(
        name="verify_install_ml",
        classifier=Spec(
            name="LogisticRegression",
            params={"max_iter": 500, "C": 1.0, "penalty": "l2", "solver": "lbfgs"},
        ),
        table_preprocessors=(Spec(name="zscore"),),
        feature_selectors=(
            Spec(name="variance", params={"threshold": 0.01}),
            Spec(name="correlation", params={"threshold": 0.95, "method": "spearman"}),
        ),
        random_seed=42,
    )


# ---------------------------------------------------------------------------
# Smoke-level checks
# ---------------------------------------------------------------------------


def run_smoke_import(runner: CheckRunner) -> str:
    """
    Import ``habit``, read ``__version__``, and smoke-test utility entry points.

    Returns:
        Detected HABIT version string.
    """
    version = ""

    def _import_version() -> None:
        nonlocal version
        import habit

        version = habit.__version__
        assert isinstance(version, str) and version, "habit.__version__ must be a non-empty str"

    runner.run("import", "habit_version", _import_version)

    def _show_versions() -> None:
        from habit.api.utils import show_versions

        versions = show_versions()
        assert isinstance(versions, dict)
        assert "habit" in versions

    runner.run("import", "show_versions", _show_versions)

    def _list_plugins() -> None:
        from habit.api.plugins import list_plugins

        plugins = list_plugins()
        assert isinstance(plugins, (list, tuple)), "list_plugins must return a sequence"
        assert len(plugins) > 0, "list_plugins returned an empty registry"

    runner.run("import", "list_plugins", _list_plugins)

    return version


def run_smoke_public_api(runner: CheckRunner) -> None:
    """
    Verify every symbol in ``PUBLIC_NAMESPACES`` resolves from its owner package.

    Each symbol is its own timed check so slow lazy imports are visible in
    the report (important when diagnosing optional heavy backends).
    """
    from habit.api.registry import PUBLIC_NAMESPACES

    for namespace, symbols in PUBLIC_NAMESPACES.items():
        for symbol in symbols:
            check_name = f"{namespace.rsplit('.', 1)[-1]}.{symbol}"
            runner.run(
                "public_api",
                check_name,
                lambda ns=namespace, sym=symbol: _verify_public_symbol(ns, sym),
            )


def run_smoke_cli_help(runner: CheckRunner) -> None:
    """
    Invoke ``python -m habit <subcommand> --help`` for every CLI command.

    Confirms the module entry point (``habit.__main__``) and Click wiring.
    """

    def _help_for(subcommand: str) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "habit", subcommand, "--help"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, (
            f"exit {completed.returncode}; stderr={completed.stderr[:500]}"
        )
        assert "Usage:" in completed.stdout or "Usage:" in completed.stderr, (
            "help output missing Usage banner"
        )

    for entry in CLI_SUBCOMMANDS:
        name = str(entry["name"])
        runner.run("cli_help", name, lambda cmd=name: _help_for(cmd))


def run_smoke_optional_deps(runner: CheckRunner, *, skip_extras: bool) -> None:
    """
    When an optional module is **absent**, trigger its code path and expect
    :class:`~habit.exceptions.OptionalDependencyError` with a helpful message.

    When the module is already installed the absence path cannot be exercised
    and the check is marked SKIP (not PASS — we did not prove the error path).
    """
    if skip_extras:
        runner.add_skip(
            "optional_deps",
            "all",
            "--skip-extras requested",
        )
        return

    from habit.api.utils import is_available
    from habit.exceptions import OptionalDependencyError

    for probe in _build_optional_probes():
        if is_available(probe.module_name):
            runner.add_skip(
                "optional_deps",
                probe.description,
                f"{probe.module_name} is installed; absence path not testable",
            )
            continue

        def _exercise(p: OptionalExtraProbe = probe) -> None:
            try:
                p.trigger()
            except OptionalDependencyError as exc:
                message = str(exc)
                assert message.strip(), "OptionalDependencyError must carry guidance text"
                assert (
                    p.extra_name in message
                    or p.module_name in message.lower()
                    or "install" in message.lower()
                    or len(message) > 20
                ), "error message should mention the missing extra or give install guidance"
                return
            except ImportError as exc:
                # Torch paths in habit.utils.image_converter use ImportError today.
                if p.module_name == "torch":
                    message = str(exc)
                    assert "torch" in message.lower(), (
                        "torch ImportError should name the missing package"
                    )
                    return
                raise
            raise AssertionError(
                f"Expected OptionalDependencyError for missing {p.module_name}, "
                "but no error was raised"
            )

        runner.run("optional_deps", probe.description, _exercise)


# ---------------------------------------------------------------------------
# Full-level checks
# ---------------------------------------------------------------------------


def run_full_demo_presence(runner: CheckRunner, demo_root: Optional[Path]) -> bool:
    """
    Verify demo imaging + tabular inputs exist.

    Returns:
        ``True`` when both trees are present; habitat/ML checks depend on this.
    """
    if demo_root is None:
        runner.add_skip(
            "demo_data",
            "root",
            "no demo data directory (pass --demo-data or run from source checkout)",
        )
        return False

    imaging = demo_imaging_root(demo_root)
    ml_csv = demo_ml_csv(demo_root)

    def _imaging() -> None:
        assert imaging.is_dir(), f"missing imaging root: {imaging}"
        subjects = list(imaging.glob("images/*"))
        assert subjects, f"no subjects under {imaging / 'images'}"

    runner.run("demo_data", "imaging_root", _imaging)

    def _ml_table() -> None:
        assert ml_csv.is_file(), f"missing ML CSV: {ml_csv}"

    runner.run("demo_data", "ml_csv", _ml_table)

    imaging_ok = any(
        record.status == CheckStatus.PASS
        for record in runner.records
        if record.group == "demo_data" and record.name == "imaging_root"
    )
    ml_ok = any(
        record.status == CheckStatus.PASS
        for record in runner.records
        if record.group == "demo_data" and record.name == "ml_csv"
    )
    return imaging_ok and ml_ok


def run_full_atomic_api(runner: CheckRunner, demo_root: Path) -> None:
    """
    Exercise subject-level and volume-level preprocessing (API-first contract).

    These operators must work on a single :class:`~habit.contracts.subject.Subject`
    without YAML, cohort backends, or batch directories.
    """

    def _atomic() -> None:
        from habit.api.preprocessing import preprocess_image, preprocess_subject

        cohort = _load_demo_cohort(demo_root)
        subject = cohort[0]
        steps = {
            "resample": {
                "target_spacing": [3.0, 3.0, 3.0],
                "img_mode": "bilinear",
            },
        }
        processed = preprocess_subject(
            subject,
            steps,
            mask_roi=_DEMO_ROI,
            broadcast_mask=True,
        )
        probe = processed.image(_DEMO_MODALITIES[0])
        assert probe.data.shape[0] > 0, "preprocess_subject returned empty volume"

        volume = subject.image(_DEMO_MODALITIES[0])
        mask_vol = subject.mask(_DEMO_ROI)
        single = preprocess_image(
            volume, steps, mask=mask_vol, modality=_DEMO_MODALITIES[0]
        )
        assert single.data.shape == probe.data.shape, "preprocess_image spacing/shape mismatch"

    runner.run("full_atomic", "preprocess_subject_and_image", _atomic)


def run_full_habitat(
    runner: CheckRunner,
    demo_root: Path,
    *,
    mode: str,
) -> Optional[Any]:
    """
    Run one habitat recipe and persist artefacts under the harness output dir.

    Args:
        demo_root: Demo-data root.
        mode: ``two_step`` or ``direct_pooling``.

    Returns:
        In-memory :class:`~habit.recipes.StudyResult` on success, else ``None``.
    """
    import habit.recipes as recipes

    result_box: Dict[str, Any] = {}

    def _run() -> None:
        cohort = _load_demo_cohort(demo_root)
        spec = _build_fast_habitat_spec(two_step=(mode == "two_step"))
        if mode == "two_step":
            result = recipes.Study(spec=spec, design="two_step").fit_predict(cohort)
        elif mode == "direct_pooling":
            result = recipes.Study(
                spec=spec, design="direct_pooling"
            ).fit_predict(cohort)
        else:
            raise ValueError(f"unknown habitat mode: {mode}")
        assert result.habitat_model is not None, "expected a fitted cohort habitat model"
        assert len(result.habitat_maps) == len(cohort), "one habitat map per subject"
        assert result.features.frame.shape[0] == len(cohort), "feature table row count"
        result_box["result"] = result

    runner.run("full_habitat", mode, _run)
    return result_box.get("result")


def run_full_feature_extraction(
    runner: CheckRunner,
    demo_root: Path,
    habitat_result: Optional[Any],
    out_dir: Path,
) -> None:
    """
    Extract non-radiomics habitat features from maps produced by the habitat step.

    Skips PyRadiomics ``traditional`` features when PyRadiomics is absent; MSI
    and volume features remain sufficient to prove the extraction pipeline.
    """
    from habit.api.utils import is_available
    import habit.recipes as recipes

    if habitat_result is None:
        runner.add_skip(
            "full_features",
            "extract_habitat_features",
            "habitat step did not produce a StudyResult",
        )
        return

    maps_dir = out_dir / "habitat_maps_for_extract"
    maps_dir.mkdir(parents=True, exist_ok=True)
    habitat_result.save(maps_dir, write_cluster_plots=False)

    # Use only feature types that do not require PyRadiomics for minimal runtime.
    feature_types = ["non_radiomics", "whole_habitat", "msi", "ith_score"]
    if is_available("radiomics"):
        feature_types.insert(0, "traditional")

    config = {
        "raw_img_folder": str(demo_imaging_root(demo_root)),
        "habitats_map_folder": str(maps_dir),
        "out_dir": str(out_dir / "extract_features"),
        "n_processes": 1,
        "habitat_pattern": "*_habitats.nrrd",
        "feature_types": feature_types,
        "n_habitats": _FULL_N_HABITATS,
    }

    def _extract() -> None:
        result = recipes.extract_habitat_features(config)
        assert result.output_dir.is_dir(), "extract_habitat_features did not write output_dir"
        artefacts = result.artifacts or {}
        assert artefacts, "expected at least one artefact path"

    runner.run("full_features", "extract_habitat_features", _extract)


def run_full_ml(runner: CheckRunner, demo_root: Path) -> None:
    """Train a minimal logistic model on the demo radiomics CSV."""

    def _train() -> None:
        import pandas as pd

        import habit.recipes as recipes
        from habit.contracts import FeatureTable
        from habit.contracts.outcome import BinaryOutcome

        csv_path = demo_ml_csv(demo_root)
        frame = pd.read_csv(csv_path, dtype={"subject_id": str})
        id_col = "subject_id"
        label_col = "label"
        feature_columns = tuple(
            col for col in frame.columns if col not in (id_col, label_col)
        )
        table = FeatureTable(
            frame=frame,
            id_columns=(id_col,),
            feature_columns=feature_columns,
            outcome=BinaryOutcome(column=label_col),
        )
        spec = _build_fast_ml_spec()
        result = recipes.train_model(table, spec, seed=42)
        assert result.train_metrics, "train_model must return training metrics"
        assert result.pipeline is not None, "train_model must return a fitted pipeline"

    runner.run("full_ml", "train_model", _train)


def run_full_cli_pipeline(
    runner: CheckRunner,
    demo_root: Path,
    config_root: Optional[Path],
    out_dir: Path,
) -> None:
    """
    Optionally run ``habit check-config`` and ``habit get-habitat`` on shipped YAML.

    These checks SKIP when the ``config/`` tree is unavailable (typical for
    wheel-only installs). Real compute uses patched absolute paths under
    ``out_dir`` so the harness never mutates repository demo results.
    """
    if config_root is None:
        runner.add_skip(
            "full_cli",
            "check-config",
            "config/ tree not found (wheel install)",
        )
        runner.add_skip(
            "full_cli",
            "get-habitat",
            "config/ tree not found (wheel install)",
        )
        return

    minimal_yaml = config_root / "habitat" / "config_habitat_two_step_minimal.yaml"
    if not minimal_yaml.is_file():
        runner.add_skip(
            "full_cli",
            "check-config",
            f"missing {minimal_yaml.name}",
        )
        runner.add_skip(
            "full_cli",
            "get-habitat",
            f"missing {minimal_yaml.name}",
        )
        return

    patched_yaml = out_dir / "habitat_two_step_patched.yaml"
    text = minimal_yaml.read_text(encoding="utf-8")
    habitat_out = out_dir / "cli_habitat_two_step"
    imaging = demo_imaging_root(demo_root)
    # Replace relative demo paths with absolute ones (Windows-safe forward slashes).
    text = text.replace(
        "../../demo_data/preprocessed",
        imaging.as_posix(),
    )
    text = text.replace(
        "../../demo_data/results/habitat_two_step",
        habitat_out.as_posix(),
    )
    # Speed caps for full-level CLI run.
    text = text.replace("processes: 2", "processes: 1")
    text = text.replace("plot_curves: true", "plot_curves: false")
    patched_yaml.write_text(text, encoding="utf-8")

    def _check_config() -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "habit",
                "check-config",
                "--config",
                str(patched_yaml),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr[:800]

    runner.run("full_cli", "check-config", _check_config)

    def _get_habitat() -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "habit",
                "get-habitat",
                "--config",
                str(patched_yaml),
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert completed.returncode == 0, (
            (completed.stdout or "")[-2000:] + (completed.stderr or "")[-2000:]
        )
        assert habitat_out.is_dir(), "CLI habitat run did not create out_dir"
        assert any(habitat_out.glob("*_habitats.nrrd")), "missing habitat NRRD outputs"

    runner.run("full_cli", "get-habitat", _get_habitat)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary_table(report: HarnessReport) -> None:
    """Print a human-readable PASS/FAIL/SKIP table to stdout."""
    print()
    print(f"HABIT install verification - level={report.level} version={report.habit_version}")
    print(f"Total runtime: {report.total_seconds:.1f}s")
    summary = report.summary
    print(
        f"Summary: PASS={summary['PASS']} FAIL={summary['FAIL']} SKIP={summary['SKIP']}"
    )
    print()
    header = f"{'GROUP':<16} {'CHECK':<42} {'STATUS':<6} {'SEC':>7}  REASON"
    print(header)
    print("-" * len(header))
    for record in report.checks:
        reason = record.reason.replace("\n", " ")[:120]
        print(
            f"{record.group:<16} {record.name:<42} {record.status.value:<6} "
            f"{record.seconds:7.2f}  {reason}"
        )
    print()


def write_results_json(report: HarnessReport, out_dir: Path) -> Path:
    """
    Persist machine-readable results next to other harness artefacts.

    Args:
        report: Completed harness report.
        out_dir: Output directory (created when missing).

    Returns:
        Path to ``results.json``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.json"
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the harness."""
    parser = argparse.ArgumentParser(
        description=(
            "Verify that an installed habitat-analysis package is functional."
        ),
    )
    parser.add_argument(
        "--level",
        choices=("smoke", "full"),
        required=True,
        help="smoke: import/API/CLI-help only; full: additionally run demo pipelines.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Directory for results.json and full-level artefacts.",
    )
    parser.add_argument(
        "--demo-data",
        type=str,
        default=None,
        help="Demo data root (default: auto-detect demo_data/ next to this script).",
    )
    parser.add_argument(
        "--skip-extras",
        action="store_true",
        help="Skip optional-dependency absence-path checks in smoke level.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Run the verification harness.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 when no FAIL results, 1 otherwise.
    """
    args = parse_args(argv)
    out_dir = Path(args.out).expanduser().resolve()
    demo_root = resolve_demo_data(args.demo_data)
    config_root = resolve_config_root(demo_root)

    runner = CheckRunner()
    started = time.perf_counter()

    version = run_smoke_import(runner)
    run_smoke_public_api(runner)
    run_smoke_cli_help(runner)
    run_smoke_optional_deps(runner, skip_extras=args.skip_extras)

    habitat_result: Optional[Any] = None
    if args.level == "full":
        demo_ok = run_full_demo_presence(runner, demo_root)
        if demo_ok and demo_root is not None:
            run_full_atomic_api(runner, demo_root)
            habitat_result = run_full_habitat(runner, demo_root, mode="two_step")
            run_full_habitat(runner, demo_root, mode="direct_pooling")
            run_full_feature_extraction(runner, demo_root, habitat_result, out_dir)
            run_full_ml(runner, demo_root)
            run_full_cli_pipeline(runner, demo_root, config_root, out_dir)

    total = time.perf_counter() - started
    report = HarnessReport(
        level=args.level,
        habit_version=version,
        total_seconds=total,
        checks=runner.records,
    )
    json_path = write_results_json(report, out_dir)
    print_summary_table(report)
    print(f"Wrote {json_path}")

    failed = report.summary.get("FAIL", 0)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception:
        traceback.print_exc()
        raise SystemExit(2) from None
