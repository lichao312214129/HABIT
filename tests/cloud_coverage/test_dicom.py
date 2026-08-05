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
Coverage matrix: DICOM utilities on a pydicom-synthesized CT series.

- ``habit dicom-info`` runs for real on the synthesized series;
- ``habit sort-dicom`` runs when a dcm2niix binary is available, and is
  otherwise asserted to fail with the documented clean error (the runner
  raises ``RuntimeError: dcm2niix not found``); dcm2niix is on the
  do-not-install list for this environment.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from habit.cli import cli
from tests.cloud_coverage.conftest import RenderedConfig, run_cli
from tests.fixtures.synthetic_data import SyntheticTree, make_dicom_series

#: Reason string shared by every dcm2niix skip in this module.
DCM2NIIX_SKIP_REASON = (
    "dcm2niix binary unavailable on this Linux image (only tools/bin/dcm2niix.exe "
    "ships; dcm2niix is on the do-not-install list)"
)


@pytest.fixture(scope="module")
def dicom_series(results_root: Path) -> Path:
    """
    Synthesize the tiny CT series once per module.

    Args:
        results_root: Session results directory.

    Returns:
        Directory holding five ``.dcm`` slices.
    """
    return make_dicom_series(results_root / "dicom_series", n_slices=5, seed=42)


@pytest.mark.unit
def test_synthesized_dicom_series_is_valid(dicom_series: Path) -> None:
    """The synthesized series has five readable slices with core tags."""
    pydicom = pytest.importorskip("pydicom", reason="pydicom not installed")
    files = sorted(dicom_series.glob("*.dcm"))
    assert len(files) == 5
    dataset = pydicom.dcmread(str(files[0]))
    assert dataset.Modality == "CT"
    assert dataset.Rows == 16 and dataset.Columns == 16
    assert dataset.PatientID == "subj_dicom_001"


@pytest.mark.integration
def test_dicom_info_cli(dicom_series: Path, results_root: Path) -> None:
    """dicom-info extracts one row per slice with the requested tags."""
    if pytest.importorskip("pydicom", reason="pydicom not installed") is None:
        return
    output_csv = results_root / "dicom_info" / "info.csv"
    run_cli(
        CliRunner(),
        [
            "dicom-info",
            "-i",
            str(dicom_series),
            "-t",
            "Modality,PatientID,InstanceNumber",
            "-o",
            str(output_csv),
            "-f",
            "csv",
        ],
    )
    assert output_csv.is_file()
    frame = pd.read_csv(output_csv)
    # dicom-info reports one row per SERIES, counting the slices within it.
    assert len(frame) == 1
    assert set(frame["Modality"]) == {"CT"}
    assert int(frame["Files_In_Series"].iloc[0]) == 5


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("dcm2niix") is not None, reason="dcm2niix present; error-path not applicable")
def test_sort_dicom_missing_binary_fails_cleanly(
    dicom_series: Path, synthetic_tree: SyntheticTree, render_config
) -> None:
    """Without dcm2niix, sort-dicom exits non-zero with the documented error.

    The runner raises ``RuntimeError("dcm2niix not found: ...")`` instead of
    an opaque ``FileNotFoundError``; the CLI surfaces that message and a
    non-zero exit code (the full traceback additionally lands in the run
    log by codebase convention, which is why only the message and exit code
    are asserted here).
    """
    rendered: RenderedConfig = render_config(
        "sort_dicom.yaml",
        "sort_dicom",
        synthetic_tree,
        {"@DICOM_DIR@": dicom_series.as_posix()},
    )
    result = CliRunner().invoke(cli, ["sort-dicom", "-c", str(rendered.path)])
    assert result.exit_code != 0
    output = result.output + (result.stderr or "")
    assert "dcm2niix not found" in output
    # The graceful RuntimeError must be the failure reason, not an OSError
    # from subprocess failing to locate the executable.
    assert "No such file or directory" not in output


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("dcm2niix") is None, reason=DCM2NIIX_SKIP_REASON)
def test_sort_dicom_cli(
    dicom_series: Path, synthetic_tree: SyntheticTree, render_config
) -> None:
    """sort-dicom rewrites the series through dcm2niix (binary present only)."""
    rendered: RenderedConfig = render_config(
        "sort_dicom.yaml",
        "sort_dicom_run",
        synthetic_tree,
        {"@DICOM_DIR@": dicom_series.as_posix()},
    )
    run_cli(CliRunner(), ["sort-dicom", "-c", str(rendered.path)])
    outputs = [p for p in rendered.out_dir.glob("**/*") if p.is_file()]
    assert outputs, "sort-dicom produced no output files"
