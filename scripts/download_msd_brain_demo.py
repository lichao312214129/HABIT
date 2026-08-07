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
Download a small MSD Task01 BrainTumour (BraTS-like) demo for HABIT.

Doctor-friendly helper: plain HTTPS + stdlib + SimpleITK only. No MONAI,
torch, or gdown. After download, splits each 4D NIfTI into HABIT's
conventional layout under ``--out``.

MSD Task01 channel order (4D last axis / SimpleITK component index)
------------------------------------------------------------------
Medical Segmentation Decathlon Task01 stores multiparametric MRI as one
4D volume per subject. The official modality order is:

    0: flair   (T2-FLAIR)
    1: t1      (native T1-weighted)
    2: t1ce    (post-Gd T1 / T1gd)
    3: t2      (native T2-weighted)

This matches the MSD website description (FLAIR, T1w, T1gd, T2w) and the
nnU-Net Task001 conversion suffixes ``_0000``..``_0003``. It is *not*
the BraTS filename suffix order (t1 / t1ce / t2 / flair).

Output layout (HABIT convention)
--------------------------------
::

    <out>/
      images/<subj>/{flair,t1,t1ce,t2}/<subj>_<mod>.nii.gz
      masks/<subj>/tumor/<subj>_tumor.nii.gz
      masks/<subj>/{flair,t1,t1ce,t2}/<subj>_tumor.nii.gz   # CLI compat copies

The primary ROI folder is ``tumor`` (whole-tumor = label > 0). Identical
mask files are also written under each image modality name so the current
``habit get-habitat`` / ``habit ha`` path (which resolves ROI as the first
modality key in the feature expression) can load the directory without a
separate manifest.

Acknowledgement
---------------
Data are from the Medical Segmentation Decathlon (MSD) Task01 BrainTumour,
derived from BraTS 2016/2017. Please cite the MSD / BraTS publications when
using these cases in research. License: CC-BY-SA 4.0 (see medicaldecathlon.com).

Example
-------
::

    python scripts/download_msd_brain_demo.py --n 5
    habit check-config -c config/habitat/config_habitat_msd_demo.yaml
    habit get-habitat -c config/habitat/config_habitat_msd_demo.yaml
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# Optional progress bar: prefer habit.utils.progress_utils when available
# (editable / installed package). Fall back to plain print so clinicians can
# still run this script with only SimpleITK + the Python standard library.
# -----------------------------------------------------------------------------
try:
    from habit.utils.progress_utils import CustomTqdm as _ProgressBar
except Exception:  # noqa: BLE001 — any import failure must not abort download
    _ProgressBar = None  # type: ignore[misc, assignment]


# Primary host: AWS Open Data mirror used by MONAI tutorials (HTTPS, no login).
# Historical NVIDIA Clara URL returns 404 as of 2026-08; kept as a documented
# fallback name only. Hugging Face LFS resolve URL is a secondary HTTPS mirror.
DEFAULT_URLS: Tuple[str, ...] = (
    "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
    "https://huggingface.co/datasets/Novel-BioMedAI/Medical_Segmentation_Decathlon/"
    "resolve/main/Task01_BrainTumour.tar",
)

# Official MSD Task01 4D channel order → HABIT modality folder names.
MSD_CHANNEL_TO_MODALITY: Tuple[str, ...] = ("flair", "t1", "t1ce", "t2")

TAR_MEMBER_PREFIX = "Task01_BrainTumour"
IMAGES_TR = f"{TAR_MEMBER_PREFIX}/imagesTr"
LABELS_TR = f"{TAR_MEMBER_PREFIX}/labelsTr"


def _repo_root() -> Path:
    """
    Return the repository root (parent of ``scripts/``).

    Returns:
        Absolute path to the habit_project root when this file lives under
        ``scripts/``; otherwise the current working directory.
    """
    here = Path(__file__).resolve()
    if here.parent.name == "scripts":
        return here.parent.parent
    return Path.cwd()


def _print(msg: str) -> None:
    """Write a user-facing status line and flush immediately."""
    print(msg, flush=True)


def _iter_progress(
    iterable: Iterable,
    *,
    total: Optional[int],
    desc: str,
) -> Iterable:
    """
    Wrap ``iterable`` with habit ``CustomTqdm`` when available, else yield as-is.

    Args:
        iterable: Items to iterate.
        total: Expected length for the progress bar (may be None).
        desc: Short English label for the bar.

    Returns:
        An iterable that yields the same items (optionally with a progress UI).
    """
    if _ProgressBar is None:
        return iterable
    return _ProgressBar(iterable, total=total, desc=desc)


def probe_url(url: str, timeout: float = 30.0) -> Tuple[bool, Optional[int]]:
    """
    Probe a download URL with HEAD, falling back to a short GET.

    Args:
        url: Absolute HTTPS URL to the Task01 tarball.
        timeout: Socket timeout in seconds.

    Returns:
        ``(ok, content_length_or_None)``. ``ok`` is True when the server
        responds with HTTP 2xx.
    """
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "HABIT-msd-demo/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length_header = response.headers.get("Content-Length")
            length = int(length_header) if length_header else None
            return 200 <= int(response.status) < 300, length
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError, TimeoutError):
        pass

    # Some hosts reject HEAD; try GET and close immediately after headers.
    get_request = urllib.request.Request(
        url,
        method="GET",
        headers={"User-Agent": "HABIT-msd-demo/1.0"},
    )
    try:
        with urllib.request.urlopen(get_request, timeout=timeout) as response:
            length_header = response.headers.get("Content-Length")
            length = int(length_header) if length_header else None
            ok = 200 <= int(response.status) < 300
            return ok, length
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError, TimeoutError):
        return False, None


def resolve_download_url(urls: Sequence[str]) -> Tuple[str, Optional[int]]:
    """
    Return the first working URL from ``urls``.

    Args:
        urls: Candidate HTTPS URLs, preferred first.

    Returns:
        ``(url, content_length)`` for the first reachable mirror.

    Raises:
        RuntimeError: If every candidate fails the probe.
    """
    errors: List[str] = []
    for url in urls:
        ok, length = probe_url(url)
        if ok:
            size_msg = f"{length / (1024 ** 3):.1f} GiB" if length else "unknown size"
            _print(f"Using download URL ({size_msg}):\n  {url}")
            return url, length
        errors.append(url)
    raise RuntimeError(
        "No working HTTPS mirror found for MSD Task01_BrainTumour.tar.\n"
        "Tried:\n  - " + "\n  - ".join(errors) + "\n"
        "Check your network / proxy, or download the tar manually into --cache."
    )


def download_file(
    url: str,
    destination: Path,
    *,
    expected_size: Optional[int] = None,
    force: bool = False,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """
    Download ``url`` to ``destination`` with optional resume / skip.

    Args:
        url: HTTPS URL of the archive.
        destination: Local file path for the tar.
        expected_size: Content-Length from the probe (bytes), if known.
        force: When True, delete any partial/complete cache and re-download.
        chunk_size: Read buffer size in bytes.

    Returns:
        Path to the downloaded (or cached) file.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if force and destination.exists():
        destination.unlink()

    if (
        destination.exists()
        and expected_size is not None
        and destination.stat().st_size == expected_size
    ):
        _print(f"Cache hit (complete): {destination}")
        return destination

    # Resume from a partial file when the server supports Range.
    existing = destination.stat().st_size if destination.exists() else 0
    headers = {"User-Agent": "HABIT-msd-demo/1.0"}
    mode = "wb"
    if existing > 0 and not force:
        if expected_size is not None and existing > expected_size:
            destination.unlink()
            existing = 0
        elif expected_size is None or existing < expected_size:
            headers["Range"] = f"bytes={existing}-"
            mode = "ab"
            _print(f"Resuming download from byte {existing} ...")

    request = urllib.request.Request(url, headers=headers)
    _print(f"Downloading to {destination} ...")
    with urllib.request.urlopen(request, timeout=120) as response:
        # If the server ignored Range and sent 200, restart from scratch.
        if mode == "ab" and int(response.status) == 200:
            mode = "wb"
            existing = 0
        total = expected_size
        if total is None:
            remaining = response.headers.get("Content-Length")
            if remaining:
                total = existing + int(remaining)

        downloaded = existing
        bar = None
        if _ProgressBar is not None and total:
            bar = _ProgressBar(
                total=total,
                desc="Download",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            )
            bar.update(existing)

        with destination.open(mode) as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if bar is not None:
                    bar.update(len(chunk))
                elif total:
                    pct = 100.0 * downloaded / total
                    _print(f"  ... {downloaded / (1024 ** 2):.1f} MiB ({pct:.1f}%)")
                else:
                    _print(f"  ... {downloaded / (1024 ** 2):.1f} MiB")

        if bar is not None:
            bar.close()

    _print(f"Download complete: {destination} ({destination.stat().st_size} bytes)")
    return destination


def list_training_subjects(tar_path: Path) -> List[str]:
    """
    List subject IDs under ``imagesTr`` inside the Task01 tarball.

    Args:
        tar_path: Path to ``Task01_BrainTumour.tar``.

    Returns:
        Sorted unique subject IDs (e.g. ``BRATS_001``).
    """
    subjects: List[str] = []
    with tarfile.open(tar_path, mode="r:") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            name = member.name.replace("\\", "/")
            if not name.startswith(IMAGES_TR + "/"):
                continue
            if not (name.endswith(".nii.gz") or name.endswith(".nii")):
                continue
            stem = Path(name).name
            # MSD names look like BRATS_001.nii.gz (single 4D file per subject).
            subject_id = stem.replace(".nii.gz", "").replace(".nii", "")
            subjects.append(subject_id)
    return sorted(set(subjects))


def _member_name(folder: str, subject_id: str) -> str:
    """
    Build the tar member path for one subject file.

    Args:
        folder: ``imagesTr`` or ``labelsTr`` prefix inside the archive.
        subject_id: Subject stem without extension.

    Returns:
        Posix-style member path ending in ``.nii.gz``.
    """
    return f"{folder}/{subject_id}.nii.gz"


def extract_subject_pair(
    tar_path: Path,
    subject_id: str,
    extract_dir: Path,
) -> Tuple[Path, Path]:
    """
    Extract one subject's 4D image and label from the tarball.

    Args:
        tar_path: Path to the Task01 archive.
        subject_id: Subject ID (e.g. ``BRATS_001``).
        extract_dir: Temporary directory receiving the two NIfTI files.

    Returns:
        ``(image_path, label_path)``.

    Raises:
        FileNotFoundError: If either member is missing from the archive.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    image_member = _member_name(IMAGES_TR, subject_id)
    label_member = _member_name(LABELS_TR, subject_id)
    with tarfile.open(tar_path, mode="r:") as archive:
        try:
            image_info = archive.getmember(image_member)
        except KeyError as exc:
            raise FileNotFoundError(
                f"Missing image member {image_member!r} in {tar_path}"
            ) from exc
        try:
            label_info = archive.getmember(label_member)
        except KeyError as exc:
            raise FileNotFoundError(
                f"Missing label member {label_member!r} in {tar_path}"
            ) from exc
        archive.extract(image_info, path=extract_dir)
        archive.extract(label_info, path=extract_dir)

    image_path = extract_dir / image_member
    label_path = extract_dir / label_member
    if not image_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(
            f"Extraction failed for {subject_id}: "
            f"image={image_path.is_file()} label={label_path.is_file()}"
        )
    return image_path, label_path


def split_msd_image_to_modalities(
    image_path: Path,
    subject_id: str,
    images_root: Path,
    *,
    channel_names: Sequence[str] = MSD_CHANNEL_TO_MODALITY,
) -> Dict[str, Path]:
    """
    Split a 4D MSD NIfTI into one 3D ``.nii.gz`` per modality.

    Args:
        image_path: Path to the 4D (or multi-component) MSD training image.
        subject_id: Subject folder name under ``images/``.
        images_root: ``<out>/images`` directory.
        channel_names: Modality names in MSD channel order.

    Returns:
        Mapping from modality name to written file path.

    Raises:
        RuntimeError: If SimpleITK is missing or the channel count mismatches.
    """
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError(
            "SimpleITK is required to split MSD 4D images. "
            "Install habitat-analysis (or `pip install SimpleITK`) and retry."
        ) from exc

    image = sitk.ReadImage(str(image_path))
    n_components = int(image.GetNumberOfComponentsPerPixel())
    size = list(image.GetSize())

    # MSD Task01 is usually a vector image (components = modalities). Some
    # writers instead store modalities as a 4th spatial dimension.
    if n_components == 1 and len(size) == 4:
        n_channels = int(size[3])
        use_vector = False
    elif n_components >= 2:
        n_channels = n_components
        use_vector = True
    else:
        raise RuntimeError(
            f"{image_path} is not a multi-modality MSD volume "
            f"(components={n_components}, size={size})."
        )

    if n_channels != len(channel_names):
        raise RuntimeError(
            f"Expected {len(channel_names)} channels {tuple(channel_names)}, "
            f"found {n_channels} in {image_path}."
        )

    written: Dict[str, Path] = {}
    for index, modality in enumerate(channel_names):
        if use_vector:
            channel = sitk.VectorIndexSelectionCast(image, index)
        else:
            channel = image[:, :, :, index]
        out_dir = images_root / subject_id / modality
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{subject_id}_{modality}.nii.gz"
        sitk.WriteImage(channel, str(out_path))
        written[modality] = out_path
    return written


def write_whole_tumor_mask(
    label_path: Path,
    subject_id: str,
    masks_root: Path,
    *,
    modality_copies: Sequence[str] = MSD_CHANNEL_TO_MODALITY,
) -> Path:
    """
    Binarize MSD labels (``label > 0``) and write the whole-tumor ROI.

    Primary path: ``masks/<subj>/tumor/<subj>_tumor.nii.gz``.
    Compatibility copies are also written under each image modality folder so
    ``habit get-habitat`` (ROI = first modality key) can load the tree.

    Args:
        label_path: Path to the MSD ``labelsTr`` NIfTI (multi-class).
        subject_id: Subject folder name under ``masks/``.
        masks_root: ``<out>/masks`` directory.
        modality_copies: Extra folder names that receive an identical mask.

    Returns:
        Path to the primary ``tumor`` mask file.
    """
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError(
            "SimpleITK is required to write tumor masks. "
            "Install habitat-analysis (or `pip install SimpleITK`) and retry."
        ) from exc

    label = sitk.ReadImage(str(label_path))
    # Whole-tumor ROI for habitat analysis: any non-background label.
    binary = sitk.Cast(label > 0, sitk.sitkUInt8)

    primary_dir = masks_root / subject_id / "tumor"
    primary_dir.mkdir(parents=True, exist_ok=True)
    primary_path = primary_dir / f"{subject_id}_tumor.nii.gz"
    sitk.WriteImage(binary, str(primary_path))

    for modality in modality_copies:
        copy_dir = masks_root / subject_id / modality
        copy_dir.mkdir(parents=True, exist_ok=True)
        copy_path = copy_dir / f"{subject_id}_tumor.nii.gz"
        if copy_path.resolve() != primary_path.resolve():
            shutil.copy2(primary_path, copy_path)

    return primary_path


def convert_subjects(
    tar_path: Path,
    subject_ids: Sequence[str],
    out_dir: Path,
    *,
    force: bool = False,
) -> List[str]:
    """
    Extract and convert selected subjects into the HABIT layout.

    Args:
        tar_path: Path to ``Task01_BrainTumour.tar``.
        subject_ids: Subject IDs to materialise (already capped by ``--n``).
        out_dir: HABIT data root (contains ``images/`` and ``masks/``).
        force: When True, rebuild subjects even if outputs already exist.

    Returns:
        List of subject IDs successfully written.
    """
    images_root = out_dir / "images"
    masks_root = out_dir / "masks"
    images_root.mkdir(parents=True, exist_ok=True)
    masks_root.mkdir(parents=True, exist_ok=True)

    done: List[str] = []
    for subject_id in _iter_progress(
        subject_ids, total=len(subject_ids), desc="Convert subjects"
    ):
        marker = images_root / subject_id / "t1ce" / f"{subject_id}_t1ce.nii.gz"
        tumor_mask = masks_root / subject_id / "tumor" / f"{subject_id}_tumor.nii.gz"
        if marker.is_file() and tumor_mask.is_file() and not force:
            _print(f"  skip {subject_id} (already present; pass --force to rebuild)")
            done.append(subject_id)
            continue

        with tempfile.TemporaryDirectory(prefix=f"msd_{subject_id}_") as tmp:
            tmp_path = Path(tmp)
            _print(f"  extracting {subject_id} ...")
            image_path, label_path = extract_subject_pair(
                tar_path, subject_id, tmp_path
            )
            _print(f"  splitting modalities for {subject_id} ...")
            split_msd_image_to_modalities(image_path, subject_id, images_root)
            write_whole_tumor_mask(label_path, subject_id, masks_root)
        done.append(subject_id)
        _print(f"  wrote {subject_id}")
    return done


def print_next_steps(out_dir: Path, subjects: Sequence[str]) -> None:
    """
    Print doctor-facing instructions after a successful conversion.

    Args:
        out_dir: HABIT data root that was populated.
        subjects: Subject IDs that are ready to analyse.
    """
    _print("")
    _print("=" * 72)
    _print("MSD BrainTumour demo data is ready.")
    _print(f"  data_dir : {out_dir.resolve()}")
    _print(f"  subjects : {', '.join(subjects)}")
    _print("  modalities: flair, t1, t1ce, t2")
    _print("  roi       : tumor  (also copied under each modality folder)")
    _print("")
    _print("Next steps (from the repository root):")
    _print("  habit check-config -c config/habitat/config_habitat_msd_demo.yaml")
    _print("  habit get-habitat  -c config/habitat/config_habitat_msd_demo.yaml")
    _print("")
    _print("Or point any habitat YAML at this folder and set:")
    _print(
        "  feature_construction.voxel_level.method: "
        "concat(raw(t1ce), raw(t1), raw(t2), raw(flair))"
    )
    _print("  (modality folder names must match the expression exactly)")
    _print("=" * 72)


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.

    Returns:
        Configured ``ArgumentParser`` instance.
    """
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Download MSD Task01 BrainTumour cases and convert them to the "
            "HABIT images/masks layout (no MONAI / torch required)."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "demo_data" / "preprocessed" / "processed_images",
        help="HABIT data root (default: demo_data/preprocessed/processed_images)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of training subjects to convert (default: 5, max: 10)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=root / "demo_data" / "_msd_cache",
        help="Directory for the downloaded tar (default: demo_data/_msd_cache)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download / rebuild even when cache or outputs already exist",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=None,
        help=(
            "Override download URL (may be repeated for fallbacks). "
            "Default: AWS Open Data, then Hugging Face."
        ),
    )
    parser.add_argument(
        "--tar",
        type=Path,
        default=None,
        help="Use an already-downloaded Task01_BrainTumour.tar (skip download)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    CLI entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    n_subjects = int(args.n)
    if n_subjects < 1 or n_subjects > 10:
        parser.error("--n must be between 1 and 10 (doctor-friendly demo size)")

    out_dir: Path = args.out
    cache_dir: Path = args.cache
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.tar is not None:
        tar_path = Path(args.tar)
        if not tar_path.is_file():
            _print(f"ERROR: --tar not found: {tar_path}")
            return 1
        _print(f"Using local tar: {tar_path}")
    else:
        urls = tuple(args.url) if args.url else DEFAULT_URLS
        url, expected_size = resolve_download_url(urls)
        tar_path = cache_dir / "Task01_BrainTumour.tar"
        try:
            download_file(
                url,
                tar_path,
                expected_size=expected_size,
                force=bool(args.force),
            )
        except Exception as exc:  # noqa: BLE001
            _print(f"ERROR: download failed: {exc}")
            return 1

    try:
        all_subjects = list_training_subjects(tar_path)
    except Exception as exc:  # noqa: BLE001
        _print(f"ERROR: cannot read tarball: {exc}")
        return 1

    if not all_subjects:
        _print(f"ERROR: no imagesTr subjects found in {tar_path}")
        return 1

    selected = all_subjects[:n_subjects]
    _print(
        f"Converting first {len(selected)} of {len(all_subjects)} "
        f"training subjects → {out_dir}"
    )

    try:
        written = convert_subjects(
            tar_path, selected, out_dir, force=bool(args.force)
        )
    except Exception as exc:  # noqa: BLE001
        _print(f"ERROR: conversion failed: {exc}")
        return 1

    print_next_steps(out_dir, written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
