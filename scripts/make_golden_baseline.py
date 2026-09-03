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
Freeze the v1.0 behaviour of HABIT as a golden baseline.

Why this exists
---------------
The v1.0 release pins reproducible CLI outputs over ``demo_data/``. This script
runs the shipped CLI and records two things every release must reproduce:

1. THE ARTEFACT CONTRACT -- the exact set of files a run produces. v1 emits
   habitat label maps, supervoxel label maps, a habitats table, a
   ``habitat_model.habitatmodel`` archive, ``run_manifest.json``, and habitat
   clustering visualisations; a pipeline that computes the same numbers but
   stops writing the ``.nrrd`` maps or the cluster plots is a regression.

2. THE NUMBERS -- label maps voxel-by-voxel (sha256 over the raw buffer plus
   geometry), and tables column-by-column with their values, compared later
   under an explicit tolerance.

Stability rules applied while capturing
---------------------------------------
Only content that is reproducible in a fixed environment is compared:

* ``.log`` files carry timestamps                       -> excluded entirely.
* Plots (``.png``/``.html``/``.pdf``) are not byte-stable across matplotlib
  builds                                                -> existence + non-empty.
* Pickles (``.pkl``) embed class paths and are the very thing the refactor
  replaces                                              -> existence + non-empty.
* Checkpoint payloads under ``.habitat_checkpoint``     -> existence only.
* Label maps and tables                                 -> full content.

Every run is written to a FRESH output directory, so ``resume: true`` in the
shipped configs cannot silently skip recomputation and hand us a baseline that
was never actually computed.

Usage
-----
    python scripts/make_golden_baseline.py                 # all cases
    python scripts/make_golden_baseline.py --case habitat_two_step
    python scripts/make_golden_baseline.py --out-root D:/tmp/golden
    python scripts/make_golden_baseline.py --verify        # compare, do not write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from habit.utils.progress_utils import CustomTqdm  # noqa: E402

#: Directory holding the committed baseline fingerprints.
BASELINE_DIR: Path = REPO_ROOT / "tests" / "golden" / "baseline"

#: Default scratch directory for baseline runs (inside the untracked demo data).
DEFAULT_OUT_ROOT: Path = REPO_ROOT / "demo_data" / "results" / "_golden"

#: Volumetric label images: compared voxel-by-voxel.
ARRAY_SUFFIXES: Tuple[str, ...] = (".nrrd", ".nii", ".nii.gz", ".mha", ".mhd")

#: Tabular results: compared column-by-column under a tolerance.
TABLE_SUFFIXES: Tuple[str, ...] = (".csv", ".parquet")

#: Structured results: compared leaf-by-leaf. The machine-learning workflows
#: report fold metrics (AUC, sensitivity, calibration p-values) and per-sample
#: predicted probabilities here rather than in a table, so treating these as
#: opaque would leave the ML half of the baseline unpinned.
JSON_SUFFIXES: Tuple[str, ...] = (".json",)

#: Produced but not byte-stable: only presence and non-emptiness are checked.
OPAQUE_SUFFIXES: Tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
    ".html",
    ".pkl",
    ".joblib",
    ".habitatmodel",
)

#: Never compared: timestamps, interpreter caches, or in-flight checkpoint writes.
EXCLUDED_SUFFIXES: Tuple[str, ...] = (".log", ".pyc", ".tmp")

#: Directory names whose contents are recorded by path only.
PRESENCE_ONLY_DIRS: Tuple[str, ...] = (".habitat_checkpoint", "__pycache__")

#: Above this many numeric cells a table is summarised instead of stored inline,
#: keeping the committed JSON reviewable in a diff.
MAX_INLINE_VALUES: int = 20000

#: Relative tolerance recorded in the baseline and used by the comparison tests.
FLOAT_RTOL: float = 1e-6

#: Absolute tolerance guarding values legitimately near zero.
FLOAT_ATOL: float = 1e-9


@dataclass(frozen=True)
class GoldenCase:
    """
    One reproducible CLI invocation captured as a baseline.

    Attributes:
        name: Baseline identifier; also the JSON filename and output subdir.
        config: Config template path, relative to the repository root.
        command: HABIT CLI subcommand to invoke.
        out_dir_key: Top-level config key holding the output directory, which
            differs between the habitat workflows (``out_dir``) and the
            machine-learning workflows (``output``).
        description: Why this case is in the baseline.
        depends_on: Name of a case whose output this one consumes, run first
            into the same scratch root. Prediction and habitat-feature
            extraction are not standalone workflows -- they read a trained
            pipeline and a habitat map -- so pinning them requires producing
            their input in the same run rather than pointing at whatever
            happens to be lying in ``demo_data/results/``.
        overrides: Extra top-level config keys to inject. Values may contain
            ``{dependency_out_dir}``, substituted with the dependency's
            absolute output directory.
    """

    name: str
    config: str
    command: str
    out_dir_key: str
    description: str
    depends_on: Optional[str] = None
    overrides: Tuple[Tuple[str, str], ...] = ()


#: The three habitat clustering modes plus the ML workflow, per refactor plan
#: phase 0. Between them they exercise every artefact family the v1.0
#: implementation has to keep producing.
GOLDEN_CASES: Tuple[GoldenCase, ...] = (
    GoldenCase(
        name="habitat_two_step",
        config="config/habitat/config_habitat_two_step.yaml",
        command="get-habitat",
        out_dir_key="out_dir",
        description="two_step: supervoxels per subject, then population clustering",
    ),
    GoldenCase(
        name="habitat_one_step",
        config="config/habitat/config_habitat_one_step_raw_concat_train.yaml",
        command="get-habitat",
        out_dir_key="out_dir",
        description="one_step: voxels clustered per subject, then pooled",
    ),
    GoldenCase(
        name="habitat_direct_pooling",
        config="config/habitat/config_habitat_direct_pooling.yaml",
        command="get-habitat",
        out_dir_key="out_dir",
        description="direct_pooling: all voxels pooled and clustered once",
    ),
    GoldenCase(
        name="ml_kfold",
        config="config/machine_learning/config_machine_learning_kfold_demo.yaml",
        command="cv",
        out_dir_key="output",
        description="k-fold cross-validation over the demo feature table",
    ),
    GoldenCase(
        name="habitat_two_step_predict",
        config="config/habitat/config_habitat_two_step_predict.yaml",
        command="get-habitat",
        out_dir_key="out_dir",
        description=(
            "predict: a trained pipeline projected onto the same cohort; "
            "pins the train/predict label agreement v1 must reproduce"
        ),
        depends_on="habitat_two_step",
        overrides=(("pipeline_path", "{dependency_out_dir}/habitat_model.habitatmodel"),),
    ),
    GoldenCase(
        name="habitat_features",
        config="config/feature_extraction/config_extract_features_demo.yaml",
        command="extract",
        out_dir_key="out_dir",
        description=(
            "habitat feature families (volume / msi / ith / non_radiomics / "
            "graph) extracted from the two-step habitat maps"
        ),
        depends_on="habitat_two_step",
        overrides=(("habitats_map_folder", "{dependency_out_dir}"),),
    ),
)


def _relative_posix(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` with forward slashes."""
    return path.relative_to(root).as_posix()


def _has_suffix(path: Path, suffixes: Sequence[str]) -> bool:
    """Return whether the filename ends with any of ``suffixes``."""
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in suffixes)


def _in_presence_only_dir(path: Path, root: Path) -> bool:
    """Return whether the file lives under a presence-only directory."""
    parts = path.relative_to(root).parts
    return any(part in PRESENCE_ONLY_DIRS for part in parts)


def _fingerprint_array_file(path: Path) -> Dict[str, Any]:
    """
    Fingerprint a label image voxel-by-voxel, together with its geometry.

    A habitat map that is numerically identical but sitting on a different
    grid is still wrong, so spacing/origin/direction are part of the identity
    rather than metadata printed alongside it.

    Args:
        path: Image file to read.

    Returns:
        Digest, shape, dtype and geometry of the image.
    """
    import SimpleITK as sitk

    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    return {
        "kind": "array",
        "sha256": digest,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "spacing": [float(v) for v in image.GetSpacing()],
        "origin": [float(v) for v in image.GetOrigin()],
        "direction": [float(v) for v in image.GetDirection()],
        "label_values": sorted(int(v) for v in np.unique(array)),
    }


def _read_table(path: Path) -> pd.DataFrame:
    """Load a CSV or Parquet result table."""
    if path.name.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _fingerprint_table_file(path: Path) -> Dict[str, Any]:
    """
    Fingerprint a result table, keeping column order and cell values.

    Numeric columns are stored as values (not a hash) because they must be
    compared under a tolerance: a bit-exact requirement on floats would fail
    for reasons that have nothing to do with the refactor. Non-numeric columns
    -- subject ids above all -- are stored verbatim, since cohort order is part
    of the reproducibility contract for population-level clustering.

    Args:
        path: Table file to read.

    Returns:
        Shape, ordered column names, and either inline values or per-column
        summary statistics when the table is large.
    """
    frame = _read_table(path)
    numeric_columns = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    other_columns = [column for column in frame.columns if column not in numeric_columns]

    record: Dict[str, Any] = {
        "kind": "table",
        "shape": list(frame.shape),
        "columns": [str(column) for column in frame.columns],
        "numeric_columns": [str(column) for column in numeric_columns],
        "categorical": {
            str(column): [None if pd.isna(v) else str(v) for v in frame[column].tolist()]
            for column in other_columns
        },
    }

    n_numeric_cells = len(numeric_columns) * len(frame)
    if n_numeric_cells <= MAX_INLINE_VALUES:
        record["values"] = {
            str(column): [
                None if pd.isna(value) else float(value)
                for value in frame[column].tolist()
            ]
            for column in numeric_columns
        }
    else:
        # Large tables (e.g. voxel-level exports) would bloat the committed
        # JSON beyond reviewability; per-column moments still catch drift.
        record["summary"] = {
            str(column): {
                "count": int(frame[column].count()),
                "sum": float(frame[column].sum()),
                "mean": float(frame[column].mean()),
                "min": float(frame[column].min()),
                "max": float(frame[column].max()),
            }
            for column in numeric_columns
        }
    return record


#: JSON leaves whose list value is a set, not a sequence. Checkpoint manifests
#: append subject ids in worker-completion order, so their order records the
#: scheduler's behaviour rather than anything about the analysis; pinning it
#: would make the baseline fail whenever two subjects finish in a different
#: order, which is exactly what parallel execution is allowed to do.
ORDER_INSENSITIVE_JSON_LEAVES: Tuple[str, ...] = (
    "completed_subjects",
    "failed_subjects",
)

#: JSON leaves that record wall-clock time, ephemeral ids, or absolute paths.
#: They are reproducible in structure but not in value across runs or scratch
#: directories, so they are omitted from golden fingerprints and comparisons.
#: ``git_commit`` is provenance metadata: every commit after baseline
#: generation would otherwise read as spurious "drift" (the dedicated
#: top-level ``environment.git_commit`` still records the baseline's origin).
VOLATILE_JSON_LEAF_NAMES: Tuple[str, ...] = (
    "started_at",
    "finished_at",
    "created_at",
    "run_id",
    "config_hash",
    "git_commit",
)
VOLATILE_JSON_LEAF_PATHS: Tuple[str, ...] = (
    "resolved_config.out_dir",
)


# The distribution version describes the executable that produced a manifest,
# not the scientific definition or its dataflow.  It therefore changes during
# ordinary package releases, while every other dependency/version and all Spec
# fingerprints remain exact golden contracts.
_HABIT_SOFTWARE_LEAF = "software.habit"


def _is_volatile_json_leaf(leaf_path: str) -> bool:
    """Return whether a flattened JSON leaf should be ignored in golden diffs."""
    if leaf_path in VOLATILE_JSON_LEAF_PATHS:
        return True
    if leaf_path.endswith(".resolved_config.out_dir"):
        return True
    if leaf_path == _HABIT_SOFTWARE_LEAF or leaf_path.endswith(
        f".{_HABIT_SOFTWARE_LEAF}"
    ):
        return True
    last_segment = leaf_path.rsplit(".", 1)[-1]
    return last_segment in VOLATILE_JSON_LEAF_NAMES


def _json_leaf_agrees(leaf: str, expected: Any, actual: Any) -> bool:
    """
    Compare one non-numeric JSON leaf, honouring set-valued exceptions.

    Args:
        leaf: Leaf name inside the fingerprinted document.
        expected: Baseline value.
        actual: Freshly captured value.

    Returns:
        Whether the two values agree.
    """
    if (
        leaf in ORDER_INSENSITIVE_JSON_LEAVES
        and isinstance(expected, list)
        and isinstance(actual, list)
    ):
        return sorted(map(str, expected)) == sorted(map(str, actual))
    return bool(expected == actual)


def _is_number(value: Any) -> bool:
    """Return whether a JSON leaf should be compared numerically."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _flatten_json(node: Any, prefix: str = "") -> Dict[str, Any]:
    """
    Flatten a JSON document into dotted leaf paths.

    A list of scalars is kept whole rather than exploded into one path per
    element: the ML results store 114 predicted probabilities per fold and per
    model, and per-element paths would make the baseline unreadable without
    making it any stricter.

    Args:
        node: Current JSON node.
        prefix: Dotted path accumulated so far.

    Returns:
        Mapping from leaf path to leaf value (scalar or list of scalars).
    """
    leaves: Dict[str, Any] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(_flatten_json(value, child))
    elif isinstance(node, list):
        if all(not isinstance(item, (dict, list)) for item in node):
            leaves[prefix] = node
        else:
            for index, value in enumerate(node):
                leaves.update(_flatten_json(value, f"{prefix}[{index}]"))
    else:
        leaves[prefix] = node
    return leaves


def _fingerprint_json_file(path: Path) -> Dict[str, Any]:
    """
    Fingerprint a structured result document leaf-by-leaf.

    Numeric leaves are stored as values so they can be compared under the same
    tolerance as tables; everything else (selected feature names, method
    labels) is stored verbatim and compared exactly.

    Args:
        path: JSON file to read.

    Returns:
        Numeric and literal leaves, or a presence marker when the document
        cannot be parsed.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _fingerprint_opaque_file(path)

    numeric: Dict[str, List[Optional[float]]] = {}
    literal: Dict[str, Any] = {}
    for leaf_path, value in _flatten_json(document).items():
        if _is_volatile_json_leaf(leaf_path):
            continue
        if isinstance(value, list):
            if value and all(_is_number(item) for item in value):
                # NaN is not valid JSON; store it as null and let the
                # comparison treat null and NaN as the same missing value.
                numeric[leaf_path] = [
                    None if (isinstance(item, float) and np.isnan(item)) else float(item)
                    for item in value
                ]
            else:
                literal[leaf_path] = value
        elif _is_number(value):
            numeric[leaf_path] = [
                None if (isinstance(value, float) and np.isnan(value)) else float(value)
            ]
        else:
            literal[leaf_path] = value

    return {"kind": "json", "numeric": numeric, "literal": literal}


def _fingerprint_opaque_file(path: Path) -> Dict[str, Any]:
    """
    Record that a non-comparable artefact was produced and is not empty.

    Plots and pickles are exactly the artefacts a refactor tends to drop
    silently, so their presence is asserted even though their bytes are not.

    Args:
        path: File to record.

    Returns:
        Presence marker with a coarse size bucket.
    """
    return {"kind": "presence", "non_empty": path.stat().st_size > 0}


def fingerprint_output_dir(out_dir: Path) -> Dict[str, Any]:
    """
    Fingerprint every artefact produced by one run.

    Args:
        out_dir: Directory the CLI wrote into.

    Returns:
        Mapping from relative artefact path to its fingerprint, plus the
        ordered artefact list that forms the output contract.

    Raises:
        FileNotFoundError: If the run produced no output directory.
    """
    if not out_dir.is_dir():
        raise FileNotFoundError(f"No output directory produced: {out_dir}")

    artefacts: Dict[str, Any] = {}
    files = sorted(p for p in out_dir.rglob("*") if p.is_file())
    for file_path in files:
        if file_path.suffix.lower() == ".tmp":
            continue
        if _has_suffix(file_path, EXCLUDED_SUFFIXES):
            continue
        key = _relative_posix(file_path, out_dir)
        if _in_presence_only_dir(file_path, out_dir):
            artefacts[key] = {"kind": "presence", "non_empty": file_path.stat().st_size > 0}
        elif _has_suffix(file_path, ARRAY_SUFFIXES):
            artefacts[key] = _fingerprint_array_file(file_path)
        elif _has_suffix(file_path, TABLE_SUFFIXES):
            artefacts[key] = _fingerprint_table_file(file_path)
        elif _has_suffix(file_path, JSON_SUFFIXES):
            artefacts[key] = _fingerprint_json_file(file_path)
        elif _has_suffix(file_path, OPAQUE_SUFFIXES):
            artefacts[key] = _fingerprint_opaque_file(file_path)
        else:
            artefacts[key] = _fingerprint_opaque_file(file_path)

    return {"artefacts": sorted(artefacts), "fingerprints": artefacts}


def environment_fingerprint() -> Dict[str, Any]:
    """
    Record the software stack a baseline was produced on.

    Habitat clustering results depend on the numerical libraries underneath,
    so a mismatch here explains a diff that is otherwise inexplicable.

    Returns:
        Interpreter, dependency versions and the git commit.
    """
    versions: Dict[str, str] = {"python": sys.version.split()[0]}
    for module_name in (
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "SimpleITK",
        "skimage",
        "radiomics",
    ):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[module_name] = "not installed"

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"

    return {
        "versions": versions,
        "git_commit": commit,
        "float_rtol": FLOAT_RTOL,
        "float_atol": FLOAT_ATOL,
    }


def _case_by_name(name: str) -> GoldenCase:
    """Return the declared case with this name."""
    for case in GOLDEN_CASES:
        if case.name == name:
            return case
    raise KeyError(f"Unknown golden case: {name}")


def _materialise_config(
    case: GoldenCase, out_dir: Path, dependency_out_dir: Optional[Path] = None
) -> Path:
    """
    Write a temporary config pointing at a fresh output directory.

    The temporary file is written NEXT TO the original template on purpose:
    HABIT resolves relative paths (``data_dir``, and the input manifest used by
    the direct_pooling template) against the config file's own directory, so
    moving the config elsewhere would silently break data discovery.

    Args:
        case: Case being prepared.
        out_dir: Absolute output directory to inject.
        dependency_out_dir: Output directory of ``case.depends_on``, used to
            resolve the ``{dependency_out_dir}`` placeholder in overrides.

    Returns:
        Path of the temporary config file.

    Raises:
        ValueError: If an override needs a dependency directory that was not
            supplied.
    """
    template_path = REPO_ROOT / case.config
    with open(template_path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)

    document[case.out_dir_key] = str(out_dir)
    for key, template in case.overrides:
        if "{dependency_out_dir}" in template and dependency_out_dir is None:
            raise ValueError(
                f"Case '{case.name}' override '{key}' needs a dependency "
                "output directory, but none was produced."
            )
        document[key] = template.format(
            dependency_out_dir=(
                dependency_out_dir.as_posix() if dependency_out_dir else ""
            )
        )

    # The pid keeps concurrent runs (e.g. pytest-xdist, or a manual run beside
    # a test session) from deleting each other's temporary config.
    temp_path = template_path.with_name(f".golden_{case.name}_{os.getpid()}.yaml")
    with open(temp_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle, allow_unicode=True, sort_keys=False)
    return temp_path


def run_case(case: GoldenCase, out_root: Path) -> Dict[str, Any]:
    """
    Run one CLI case from scratch and fingerprint everything it produced.

    Args:
        case: Case to run.
        out_root: Parent directory for the case's fresh output directory.

    Returns:
        The baseline record for this case.

    Raises:
        RuntimeError: If the CLI exits non-zero.
    """
    # Resolved unconditionally: HABIT interprets a relative out_dir against the
    # config file's own directory, so a relative path here would scatter output
    # under config/ instead of the requested location.
    out_dir = (out_root / case.name).resolve()
    if out_dir.exists():
        # A stale directory would let ``resume: true`` skip the work and
        # produce a baseline that was never actually recomputed.
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dependency_out_dir: Optional[Path] = None
    if case.depends_on is not None:
        # Deliberately the default root rather than ``out_root``: v0.1 hashes
        # the whole config -- ``pipeline_path`` included -- into its checkpoint
        # manifest, so a dependency living in a per-run temporary directory
        # would make that hash, and therefore the baseline, irreproducible.
        # The dependency's own correctness is pinned by its own case.
        dependency_out_dir = (DEFAULT_OUT_ROOT / case.depends_on).resolve()
        if not dependency_out_dir.is_dir() or not any(dependency_out_dir.iterdir()):
            run_case(_case_by_name(case.depends_on), DEFAULT_OUT_ROOT)

    config_path = _materialise_config(case, out_dir, dependency_out_dir)
    try:
        # Merge stderr into stdout and forward each line immediately.  ML
        # explainability can legitimately take tens of minutes; withholding
        # its tqdm/log output makes external runners misclassify healthy work
        # as a hung child process.  Keeping a bounded tail preserves concise
        # failure diagnostics without buffering the entire child transcript.
        output_tail: List[str] = []
        process = subprocess.Popen(
            [sys.executable, "-m", "habit", case.command, "-c", str(config_path)],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            try:
                sys.stdout.write(line)
            except UnicodeEncodeError:
                # Windows consoles may still expose a legacy GBK code page,
                # while CLI progress output can contain Unicode symbols such
                # as a check mark.  Preserve progress visibility rather than
                # turning a successful child run into a parent-side failure.
                encoding = sys.stdout.encoding or "utf-8"
                sys.stdout.write(
                    line.encode(encoding, errors="backslashreplace").decode(
                        encoding
                    )
                )
            sys.stdout.flush()
            output_tail.append(line)
            if len(output_tail) > 200:
                output_tail.pop(0)
        returncode = process.wait()
        if returncode != 0:
            tail = "".join(output_tail)[-4000:]
            raise RuntimeError(
                f"Case '{case.name}' failed with exit code {returncode}:\n{tail}"
            )
    finally:
        config_path.unlink(missing_ok=True)

    record = fingerprint_output_dir(out_dir)
    record["case"] = case.name
    record["config"] = case.config
    record["command"] = case.command
    record["description"] = case.description
    return record


def _compare_values(
    expected: Optional[List[Optional[float]]],
    actual: Optional[List[Optional[float]]],
    label: str,
) -> List[str]:
    """Compare two numeric columns under the recorded tolerance."""
    if expected is None or actual is None:
        return []
    if len(expected) != len(actual):
        return [f"{label}: length {len(actual)} != baseline {len(expected)}"]
    expected_array = np.array([np.nan if v is None else v for v in expected], dtype=float)
    actual_array = np.array([np.nan if v is None else v for v in actual], dtype=float)
    if np.allclose(expected_array, actual_array, rtol=FLOAT_RTOL, atol=FLOAT_ATOL, equal_nan=True):
        return []
    diff = np.nanmax(np.abs(expected_array - actual_array))
    return [f"{label}: max abs diff {diff:.3e} exceeds rtol={FLOAT_RTOL}"]


def compare_records(baseline: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    """
    Compare a fresh run against a stored baseline.

    Args:
        baseline: Stored baseline record.
        current: Freshly captured record.

    Returns:
        Human-readable differences; empty when the run reproduces the baseline.
    """
    problems: List[str] = []

    expected_artefacts = set(baseline.get("artefacts", []))
    actual_artefacts = set(current.get("artefacts", []))
    for missing in sorted(expected_artefacts - actual_artefacts):
        problems.append(f"missing artefact: {missing}")
    for added in sorted(actual_artefacts - expected_artefacts):
        problems.append(f"unexpected artefact: {added}")

    baseline_prints: Dict[str, Any] = baseline.get("fingerprints", {})
    current_prints: Dict[str, Any] = current.get("fingerprints", {})
    for key in sorted(expected_artefacts & actual_artefacts):
        expected = baseline_prints[key]
        actual = current_prints[key]
        if expected.get("kind") != actual.get("kind"):
            problems.append(f"{key}: kind changed {expected.get('kind')} -> {actual.get('kind')}")
            continue

        kind = expected.get("kind")
        if kind == "array":
            if expected["sha256"] != actual["sha256"]:
                problems.append(f"{key}: label map differs voxel-wise")
            for field in ("shape", "spacing", "origin", "direction", "label_values"):
                if expected.get(field) != actual.get(field):
                    problems.append(f"{key}: {field} changed")
        elif kind == "table":
            if expected["columns"] != actual["columns"]:
                problems.append(f"{key}: columns changed")
                continue
            if expected["shape"] != actual["shape"]:
                problems.append(f"{key}: shape {actual['shape']} != baseline {expected['shape']}")
                continue
            if expected.get("categorical") != actual.get("categorical"):
                problems.append(f"{key}: non-numeric column values changed")
            for column, values in (expected.get("values") or {}).items():
                problems.extend(
                    _compare_values(values, (actual.get("values") or {}).get(column), f"{key}[{column}]")
                )
            for column, stats in (expected.get("summary") or {}).items():
                actual_stats = (actual.get("summary") or {}).get(column, {})
                for stat_name, stat_value in stats.items():
                    problems.extend(
                        _compare_values(
                            [stat_value], [actual_stats.get(stat_name)], f"{key}[{column}.{stat_name}]"
                        )
                    )
        elif kind == "json":
            expected_numeric: Dict[str, Any] = expected.get("numeric", {})
            actual_numeric: Dict[str, Any] = actual.get("numeric", {})
            for leaf in sorted(set(expected_numeric) - set(actual_numeric)):
                problems.append(f"{key}: missing numeric leaf {leaf}")
            for leaf in sorted(set(actual_numeric) - set(expected_numeric)):
                problems.append(f"{key}: unexpected numeric leaf {leaf}")
            for leaf in sorted(set(expected_numeric) & set(actual_numeric)):
                problems.extend(
                    _compare_values(expected_numeric[leaf], actual_numeric[leaf], f"{key}.{leaf}")
                )
            expected_literal: Dict[str, Any] = expected.get("literal", {})
            actual_literal: Dict[str, Any] = actual.get("literal", {})
            # Reported leaf by leaf: "content changed" on a run manifest is
            # unactionable, and the difference is usually one field.
            for leaf in sorted(set(expected_literal) | set(actual_literal)):
                if _is_volatile_json_leaf(leaf):
                    continue
                if not _json_leaf_agrees(
                    leaf, expected_literal.get(leaf), actual_literal.get(leaf)
                ):
                    problems.append(
                        f"{key}.{leaf}: {actual_literal.get(leaf)!r} "
                        f"!= baseline {expected_literal.get(leaf)!r}"
                    )
        elif kind == "presence":
            if expected.get("non_empty") and not actual.get("non_empty"):
                problems.append(f"{key}: produced but empty")

    return problems


def baseline_path(case_name: str) -> Path:
    """Return the committed baseline file for a case."""
    return BASELINE_DIR / f"{case_name}.json"


def write_baseline(record: Dict[str, Any]) -> Path:
    """Persist one case baseline as formatted JSON."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = baseline_path(record["case"])
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Generate or verify the golden baseline.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 when verification finds drift.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in GOLDEN_CASES],
        help="Restrict to one case; repeatable. Defaults to all cases.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Scratch directory for baseline runs.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Re-run and compare against the stored baseline without writing it.",
    )
    args = parser.parse_args(argv)

    selected = [case for case in GOLDEN_CASES if not args.case or case.name in args.case]
    out_root: Path = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    environment = environment_fingerprint()
    print(f"Environment: {json.dumps(environment['versions'], indent=None)}")
    print(f"Git commit : {environment['git_commit']}")
    print(f"Output root: {out_root}\n")

    failures: List[str] = []
    for case in CustomTqdm(selected, total=len(selected), desc="Golden cases"):
        print(f"\n--- {case.name}: {case.description}")
        try:
            record = run_case(case, out_root)
        except Exception as exc:  # noqa: BLE001 - report and continue to next case
            failures.append(f"{case.name}: {exc}")
            print(f"    FAILED: {exc}")
            continue

        print(f"    artefacts: {len(record['artefacts'])}")
        if args.verify:
            stored_path = baseline_path(case.name)
            if not stored_path.is_file():
                failures.append(f"{case.name}: no stored baseline at {stored_path}")
                print("    FAILED: no stored baseline")
                continue
            with open(stored_path, "r", encoding="utf-8") as handle:
                stored = json.load(handle)
            problems = compare_records(stored, record)
            if problems:
                failures.append(f"{case.name}: {len(problems)} difference(s)")
                for problem in problems[:20]:
                    print(f"    DIFF: {problem}")
            else:
                print("    verified: identical to baseline")
        else:
            record["environment"] = environment
            written = write_baseline(record)
            print(f"    wrote {_relative_posix(written, REPO_ROOT)}")

    if not args.verify:
        with open(BASELINE_DIR / "environment.json", "w", encoding="utf-8") as handle:
            json.dump(environment, handle, indent=2, sort_keys=True)
            handle.write("\n")

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nAll cases completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
