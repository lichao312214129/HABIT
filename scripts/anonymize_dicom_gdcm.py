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
Anonymize DICOM files with GDCM (gdcmconv + gdcmanon).

Workflow per file:
  1. Decompress JPEG/JPEG-LS/RLE transfer syntax via ``gdcmconv -w`` when needed.
  2. Clear sensitive tags via ``gdcmanon --dumb --empty <group,elem>``.

Tag values passed to gdcmanon use **hexadecimal** element numbers, e.g.
``8,20`` means DICOM tag (0008,0020) Study Date — not decimal 32.

Example (Windows, default paths for demo_data):
    python scripts/anonymize_dicom_gdcm.py

Example (custom paths):
    python scripts/anonymize_dicom_gdcm.py \\
        -i F:/work/habit_project/demo_data/dicom \\
        -o F:/work/habit_project/demo_data/dicom_anon \\
        --gdcm-bin E:/software/dcmnor/GDCM-3.2.7-Windows-x86/bin
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Allow running as ``python scripts/anonymize_dicom_gdcm.py`` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from habit.utils.progress_utils import CustomTqdm


# Default GDCM install on the user's Windows machine (override with --gdcm-bin).
DEFAULT_GDCM_BIN = Path(r"E:\software\dcmnor\GDCM-3.2.7-Windows-x86\bin")

# Default demo dataset paths in this repository.
DEFAULT_INPUT_DIR = _REPO_ROOT / "demo_data" / "dicom"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "demo_data" / "dicom_anon"

# DICOM tags to empty in gdcmanon hex format: "group,element".
# See https://dicom.innolitics.com/ciods for tag definitions.
DEFAULT_TAGS_TO_EMPTY: Tuple[str, ...] = (
    # "8,20",   # (0008,0020) Study Date
    # "8,21",   # (0008,0021) Series Date
    # "8,22",   # (0008,0022) Acquisition Date
    # "8,23",   # (0008,0023) Content Date
    # "8,30",   # (0008,0030) Study Time
    # "8,31",   # (0008,0031) Series Time
    # "8,32",   # (0008,0032) Acquisition Time
    # "8,33",   # (0008,0033) Content Time
    # "8,50",   # (0008,0050) Accession Number
    # "8,80",   # (0008,0080) Institution Name
    # "8,90",   # (0008,0090) Referring Physician's Name
    "10,10",  # (0010,0010) Patient's Name
    "10,20",  # (0010,0020) Patient ID
    "10,30",  # (0010,0030) Patient's Birth Date
    "10,40",  # (0010,0040) Patient's Sex
)


@dataclass
class AnonymizeStats:
    """Summary counters for a batch anonymization run."""

    total_files: int = 0
    success_files: int = 0
    failed_files: int = 0
    failed_paths: List[Path] = field(default_factory=list)


def resolve_gdcm_executable(gdcm_bin: Path, name: str) -> Path:
    """
    Resolve a GDCM CLI executable path.

    Args:
        gdcm_bin: Directory containing GDCM binaries, or empty to search PATH.
        name: Executable base name, e.g. ``gdcmanon`` or ``gdcmconv``.

    Returns:
        Absolute path to the executable.

    Raises:
        FileNotFoundError: If the executable cannot be located.
    """
    if gdcm_bin:
        candidate = gdcm_bin / f"{name}.exe" if sys.platform == "win32" else gdcm_bin / name
        if candidate.is_file():
            return candidate

    found = shutil.which(name)
    if found:
        return Path(found)

    raise FileNotFoundError(
        f"GDCM executable '{name}' not found. "
        f"Set --gdcm-bin to the GDCM bin directory or add GDCM to PATH."
    )


def collect_dicom_files(input_dir: Path, recursive: bool = True) -> List[Path]:
    """
    Collect DICOM files under ``input_dir``.

    Args:
        input_dir: Root directory containing DICOM files.
        recursive: When True, search all subdirectories.

    Returns:
        Sorted list of DICOM file paths.
    """
    pattern = "**/*" if recursive else "*"
    files: List[Path] = []
    for path in input_dir.glob(pattern):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".dcm", ".dicom"}:
            files.append(path)
    return sorted(files)


def mirror_directory_tree(input_dir: Path, output_dir: Path) -> None:
    """
    Create ``output_dir`` and replicate subdirectory layout from ``input_dir``.

    gdcmanon on Windows may fail to create nested output folders; pre-creating
    the tree avoids ``Could not create directory`` errors.

    Args:
        input_dir: Source DICOM root directory.
        output_dir: Destination root directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in input_dir.rglob("*"):
        if subdir.is_dir():
            rel = subdir.relative_to(input_dir)
            (output_dir / rel).mkdir(parents=True, exist_ok=True)


def build_gdcmanon_cmd(
    gdcmanon: Path,
    tags_to_empty: Sequence[str],
    input_file: Path,
    output_file: Path,
) -> List[str]:
    """
    Build the gdcmanon command line for dumb-mode tag clearing.

    Args:
        gdcmanon: Path to gdcmanon executable.
        tags_to_empty: Tag specifiers in gdcmanon hex format, e.g. ``8,20``.
        input_file: Decompressed DICOM input path.
        output_file: Anonymized DICOM output path.

    Returns:
        Argument list suitable for ``subprocess.run``.
    """
    cmd: List[str] = [str(gdcmanon), "--dumb"]
    for tag in tags_to_empty:
        cmd.extend(["--empty", tag])
    cmd.extend(["-i", str(input_file), "-o", str(output_file)])
    return cmd


def run_subprocess(cmd: List[str]) -> Tuple[int, str]:
    """
    Run a subprocess and capture combined stdout/stderr.

    Args:
        cmd: Command and arguments.

    Returns:
        Tuple of (return_code, combined_output_text).
    """
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode, output.strip()


def anonymize_single_file(
    src_file: Path,
    dst_file: Path,
    gdcmconv: Path,
    gdcmanon: Path,
    tags_to_empty: Sequence[str],
    tmp_dir: Path,
) -> Tuple[bool, str]:
    """
    Decompress (if needed) and anonymize one DICOM file.

    Args:
        src_file: Source DICOM path.
        dst_file: Destination anonymized DICOM path.
        gdcmconv: Path to gdcmconv executable.
        gdcmanon: Path to gdcmanon executable.
        tags_to_empty: Tags to clear in gdcmanon hex format.
        tmp_dir: Directory for intermediate decompressed file.

    Returns:
        Tuple of (success, error_message). error_message is empty on success.
    """
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / f"decomp_{src_file.name}"

    decompress_cmd = [str(gdcmconv), "-w", "-i", str(src_file), "-o", str(tmp_file)]
    rc, msg = run_subprocess(decompress_cmd)
    if rc != 0 or not tmp_file.is_file():
        return False, f"gdcmconv failed: {msg}"

    anonymize_cmd = build_gdcmanon_cmd(gdcmanon, tags_to_empty, tmp_file, dst_file)
    rc, msg = run_subprocess(anonymize_cmd)

    try:
        tmp_file.unlink(missing_ok=True)
    except OSError:
        pass

    # gdcmanon may return non-zero on Windows even when the output file exists.
    if dst_file.is_file() and dst_file.stat().st_size > 0:
        return True, ""

    return False, f"gdcmanon failed: {msg}"


def anonymize_dicom_directory(
    input_dir: Path,
    output_dir: Path,
    gdcm_bin: Path,
    tags_to_empty: Sequence[str],
    recursive: bool = True,
    overwrite: bool = False,
) -> AnonymizeStats:
    """
    Batch-anonymize all DICOM files under ``input_dir``.

    Args:
        input_dir: Source DICOM root directory.
        output_dir: Destination root directory (mirrors relative layout).
        gdcm_bin: GDCM ``bin`` directory, or empty Path to search PATH.
        tags_to_empty: Tag specifiers in gdcmanon hex format.
        recursive: Search subdirectories when True.
        overwrite: Re-process files that already exist in ``output_dir``.

    Returns:
        AnonymizeStats with per-file success/failure counts.
    """
    gdcmconv = resolve_gdcm_executable(gdcm_bin, "gdcmconv")
    gdcmanon = resolve_gdcm_executable(gdcm_bin, "gdcmanon")

    dicom_files = collect_dicom_files(input_dir, recursive=recursive)
    stats = AnonymizeStats(total_files=len(dicom_files))

    if not dicom_files:
        return stats

    mirror_directory_tree(input_dir, output_dir)

    with tempfile.TemporaryDirectory(prefix="gdcm_anon_") as tmp_root:
        tmp_dir = Path(tmp_root)
        progress = CustomTqdm(dicom_files, desc="Anonymizing DICOM", unit="file")

        for src_file in progress:
            rel_path = src_file.relative_to(input_dir)
            dst_file = output_dir / rel_path

            if dst_file.exists() and not overwrite:
                stats.success_files += 1
                continue

            ok, err = anonymize_single_file(
                src_file=src_file,
                dst_file=dst_file,
                gdcmconv=gdcmconv,
                gdcmanon=gdcmanon,
                tags_to_empty=tags_to_empty,
                tmp_dir=tmp_dir,
            )
            if ok:
                stats.success_files += 1
            else:
                stats.failed_files += 1
                stats.failed_paths.append(src_file)
                progress.write(f"FAILED {rel_path}: {err}")

    return stats


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Anonymize DICOM files using GDCM (gdcmconv + gdcmanon).",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input DICOM directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--gdcm-bin",
        type=Path,
        default=DEFAULT_GDCM_BIN,
        help=(
            "GDCM bin directory containing gdcmconv and gdcmanon. "
            "Use an empty string to search PATH (Linux)."
        ),
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not search subdirectories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-anonymize files that already exist in the output directory.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    CLI entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on full success, 1 if any file failed.
    """
    args = parse_args(argv)

    input_dir: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()
    gdcm_bin: Path = args.gdcm_bin

    if not input_dir.is_dir():
        print(f"Error: input directory does not exist: {input_dir}", file=sys.stderr)
        return 1

    if input_dir == output_dir:
        print("Error: input and output directories must differ.", file=sys.stderr)
        return 1

    print("DICOM anonymization (GDCM)")
    print(f"  Input : {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  GDCM  : {gdcm_bin if gdcm_bin else 'PATH'}")
    print()

    stats = anonymize_dicom_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        gdcm_bin=gdcm_bin,
        tags_to_empty=DEFAULT_TAGS_TO_EMPTY,
        recursive=not args.no_recursive,
        overwrite=args.overwrite,
    )

    print()
    print(f"Total  : {stats.total_files}")
    print(f"Success: {stats.success_files}")
    print(f"Failed : {stats.failed_files}")

    if stats.failed_paths:
        print("\nFailed files:")
        for path in stats.failed_paths[:20]:
            print(f"  {path}")
        if len(stats.failed_paths) > 20:
            print(f"  ... and {len(stats.failed_paths) - 20} more")

    return 0 if stats.failed_files == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
