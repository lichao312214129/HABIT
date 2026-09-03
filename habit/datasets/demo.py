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
"""Official HABIT demo imaging pack: download once, reuse from a local cache."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
from urllib.error import URLError
from urllib.request import urlopen

from habit.exceptions import DataFormatError, HabitError

__all__ = [
    "DEMO_RELEASE_TAG",
    "DEMO_SHA256",
    "DEMO_URL",
    "PreprocessedInventory",
    "fetch_demo",
    "get_data_home",
    "inspect_preprocessed_root",
]

#: GitHub Release tag that hosts the official preprocessed imaging zip.
DEMO_RELEASE_TAG = "demo-data-v1"

#: Official asset name inside that release.
DEMO_ASSET_NAME = "preprocessed.zip"

#: SHA-256 of the official ``preprocessed.zip`` (lowercase hex).
DEMO_SHA256 = "0c58b1dc976312bef8bf765c997ac58ddbb29d0b4b6f5d4cb3bedcb18ff4032c"

#: Public HTTPS URL. Override in tests with the ``url=`` argument.
DEMO_URL = (
    "https://github.com/lichao312214129/HABIT/releases/download/"
    f"{DEMO_RELEASE_TAG}/{DEMO_ASSET_NAME}"
)

_CHUNK_BYTES = 1024 * 1024
_LAYOUT_HELP = """\
Your own data must use the same folder tree (change IDs / series names):

  DATA/
    images/<subject_id>/<modality>/<one image file>
    masks/<subject_id>/<roi>/<one mask file>

Then load it with the same call the demos use:

  cohort = cohort_from_directory(DATA, modalities=("LAP",), roi="LAP")

Swap DATA / modalities / roi to match your tree. Mask key is often the
same as one image series (here LAP).
"""


@dataclass(frozen=True)
class PreprocessedInventory:
    """Filesystem inventory of a HABIT ``images/`` + ``masks/`` root.

    This is a directory listing, not a loaded :class:`~habit.contracts.Cohort`.
    Use it to show users what is on disk and what their own tree should look
    like before they call :func:`~habit.contracts.cohort_from_directory`.

    Attributes:
        root: Absolute preprocessed root (contains ``images/`` and ``masks/``).
        subjects: Sorted subject folder names under ``images/``.
        image_keys: Sorted union of modality folder names.
        mask_keys: Sorted union of ROI folder names under ``masks/``.
        example_image: One relative image path, if any file was found.
        example_mask: One relative mask path, if any file was found.
    """

    root: Path
    subjects: Tuple[str, ...]
    image_keys: Tuple[str, ...]
    mask_keys: Tuple[str, ...]
    example_image: Optional[str]
    example_mask: Optional[str]

    def __str__(self) -> str:
        """Return the English layout report printed by :func:`fetch_demo`."""
        return format_preprocessed_inventory(self)


def get_data_home(data_home: Optional[Union[str, Path]] = None) -> Path:
    """
    Return the cache directory for downloaded HABIT demo packs.

    Resolution order: the ``data_home`` argument, then the ``HABIT_DATA``
    environment variable, then ``~/.habit_data``. The directory is created
    if it does not exist.

    Args:
        data_home: Optional override for the cache root.

    Returns:
        Absolute cache root.
    """
    if data_home is not None:
        root = Path(data_home)
    else:
        env_home = os.environ.get("HABIT_DATA")
        root = Path(env_home) if env_home else Path.home() / ".habit_data"
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def inspect_preprocessed_root(root: Union[str, Path]) -> PreprocessedInventory:
    """
    List subjects and series folders under a HABIT preprocessed root.

    Does not open NRRD/NIfTI files. Safe to call on the user's own tree to
    check that it matches the convention.

    Args:
        root: Folder that should contain ``images/`` and ``masks/``.

    Returns:
        A :class:`PreprocessedInventory` (also used as the printed report).

    Raises:
        DataFormatError: If ``images/`` is missing or has no subject folders.
    """
    resolved = Path(root).expanduser().resolve()
    images = resolved / "images"
    masks = resolved / "masks"
    if not images.is_dir():
        raise DataFormatError(
            f"Not a HABIT preprocessed root (missing images/): {resolved}"
        )
    subjects = _sorted_subdirs(images)
    if not subjects:
        raise DataFormatError(
            f"No subject folders under {images}. Expected "
            "images/<subject_id>/<modality>/<file>."
        )
    image_keys = _union_child_dirs(images, subjects)
    mask_subjects = _sorted_subdirs(masks) if masks.is_dir() else ()
    mask_keys = _union_child_dirs(masks, mask_subjects) if mask_subjects else ()
    return PreprocessedInventory(
        root=resolved,
        subjects=subjects,
        image_keys=image_keys,
        mask_keys=mask_keys,
        example_image=_first_file_rel(images, resolved),
        example_mask=_first_file_rel(masks, resolved) if masks.is_dir() else None,
    )


def format_preprocessed_inventory(info: PreprocessedInventory) -> str:
    """
    Format an inventory as the user-facing layout report.

    Args:
        info: Result of :func:`inspect_preprocessed_root`.

    Returns:
        Multi-line English text (no Chinese), suitable for ``print``.
    """
    subjects = ", ".join(info.subjects) if info.subjects else "(none)"
    image_keys = ", ".join(info.image_keys) if info.image_keys else "(none)"
    mask_keys = ", ".join(info.mask_keys) if info.mask_keys else "(none)"
    example_image = info.example_image or "(none)"
    example_mask = info.example_mask or "(none)"
    lines = [
        f"DATA (preprocessed root): {info.root}",
        "",
        "On-disk inventory of this folder:",
        f"  subjects ({len(info.subjects)}): {subjects}",
        f"  image series: {image_keys}",
        f"  mask keys:    {mask_keys}",
        f"  example image: {example_image}",
        f"  example mask:  {example_mask}",
        "",
        _LAYOUT_HELP.rstrip(),
    ]
    return "\n".join(lines)


def fetch_demo(
    *,
    data_home: Optional[Union[str, Path]] = None,
    force: bool = False,
    verbose: bool = True,
    url: Optional[str] = None,
    sha256: Optional[str] = None,
) -> Path:
    """
    Download the official preprocessed demo pack once and return its root.

    Later calls reuse the cache (or a local ``demo_data/preprocessed`` tree
    in the current working directory) and do not hit the network. When
    ``verbose`` is true the function prints the absolute path and a layout
    report so users can see where the files landed and how to arrange their
    own data.

    Args:
        data_home: Cache root override (default: ``HABIT_DATA`` or
            ``~/.habit_data``).
        force: If true, download again even when a valid cache exists.
        verbose: If true, print the path and the layout report to stdout.
        url: Optional download URL (tests / mirrors). Default is the
            official GitHub Release asset.
        sha256: Optional lowercase hex digest of the zip. Default is the
            official pack checksum.

    Returns:
        Absolute path of the preprocessed root (contains ``images/`` and
        ``masks/``). Pass this to :func:`~habit.contracts.cohort_from_directory`.

    Raises:
        HabitError: Network or extract failure.
        DataFormatError: Checksum mismatch or extracted tree is invalid.
    """
    home = get_data_home(data_home)
    cache_root = home / DEMO_RELEASE_TAG / "preprocessed"
    expected_hash = (sha256 or DEMO_SHA256).lower()
    download_url = url or DEMO_URL
    source = "cached"

    if force or not _is_valid_preprocessed(cache_root):
        local_demo = Path("demo_data") / "preprocessed"
        if (
            not force
            and url is None
            and _is_valid_preprocessed(local_demo)
        ):
            cache_root = local_demo.resolve()
            source = "local demo_data/preprocessed"
        else:
            _download_and_extract(
                dest_parent=home / DEMO_RELEASE_TAG,
                url=download_url,
                expected_hash=expected_hash,
                verbose=verbose,
            )
            cache_root = (home / DEMO_RELEASE_TAG / "preprocessed").resolve()
            source = "downloaded"
            if not _is_valid_preprocessed(cache_root):
                raise DataFormatError(
                    "Downloaded zip did not contain preprocessed/images/. "
                    f"Looked in {cache_root}."
                )
    else:
        cache_root = cache_root.resolve()

    inventory = inspect_preprocessed_root(cache_root)
    if verbose:
        _print_fetch_report(inventory, source=source)
    return inventory.root


def _print_fetch_report(info: PreprocessedInventory, *, source: str) -> None:
    """Write the location + layout report to stdout."""
    print(f"HABIT demo data ({source})")
    print(format_preprocessed_inventory(info))
    sys.stdout.flush()


def _is_valid_preprocessed(root: Path) -> bool:
    """Return True when ``root/images/<subject>/`` exists."""
    images = Path(root) / "images"
    if not images.is_dir():
        return False
    return any(path.is_dir() for path in images.iterdir())


def _sorted_subdirs(folder: Path) -> Tuple[str, ...]:
    """Return sorted non-hidden subdirectory names."""
    if not folder.is_dir():
        return ()
    names = [
        entry.name
        for entry in folder.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    ]
    return tuple(sorted(names))


def _union_child_dirs(parent: Path, subjects: Sequence[str]) -> Tuple[str, ...]:
    """Return the sorted union of child folder names under each subject."""
    keys = set()
    for subject in subjects:
        keys.update(_sorted_subdirs(parent / subject))
    return tuple(sorted(keys))


def _first_file_rel(folder: Path, root: Path) -> Optional[str]:
    """Return one file path relative to ``root``, walking depth-first."""
    if not folder.is_dir():
        return None
    for path in sorted(folder.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            return path.relative_to(root).as_posix()
    return None


def _sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 hex digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_and_extract(
    *,
    dest_parent: Path,
    url: str,
    expected_hash: str,
    verbose: bool,
) -> None:
    """Download ``url`` into ``dest_parent``, verify, and extract."""
    dest_parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest_parent / DEMO_ASSET_NAME
    extract_root = dest_parent / "preprocessed"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    _download_file(url, zip_path, verbose=verbose)
    digest = _sha256_file(zip_path)
    if digest != expected_hash:
        zip_path.unlink(missing_ok=True)
        raise DataFormatError(
            "Demo zip SHA-256 mismatch. "
            f"expected={expected_hash} got={digest} url={url}"
        )
    if verbose:
        print(f"Extracting {zip_path.name} ...")
        sys.stdout.flush()
    try:
        with zipfile.ZipFile(zip_path) as archive:
            _safe_extract(archive, dest_parent)
    except zipfile.BadZipFile as exc:
        zip_path.unlink(missing_ok=True)
        raise DataFormatError(f"Demo zip is not a valid zip: {zip_path}") from exc
    zip_path.unlink(missing_ok=True)
    sidecar = dest_parent / "SHA256"
    sidecar.write_text(expected_hash + "\n", encoding="utf-8")


def _download_file(url: str, dest: Path, *, verbose: bool) -> None:
    """Stream ``url`` to ``dest`` with a progress bar when possible."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    try:
        with urlopen(url, timeout=60) as response:
            total = _content_length(response)
            bar = None
            if verbose:
                from habit.utils.progress_utils import CustomTqdm

                bar = CustomTqdm(
                    total=total or None,
                    desc="Download demo data",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    file=sys.stdout,
                )
            try:
                with tmp.open("wb") as handle:
                    while True:
                        chunk = response.read(_CHUNK_BYTES)
                        if not chunk:
                            break
                        handle.write(chunk)
                        if bar is not None:
                            bar.update(len(chunk))
            finally:
                if bar is not None:
                    bar.close()
    except URLError as exc:
        tmp.unlink(missing_ok=True)
        raise HabitError(
            f"Could not download HABIT demo data from {url}. {exc}"
        ) from exc
    tmp.replace(dest)


def _content_length(response: object) -> Optional[int]:
    """Read Content-Length from an urllib response, if present."""
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("Content-Length")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _safe_extract(archive: zipfile.ZipFile, dest_parent: Path) -> None:
    """Extract ``archive`` into ``dest_parent``, rejecting path traversal."""
    dest_parent = dest_parent.resolve()
    for info in archive.infolist():
        target = (dest_parent / info.filename).resolve()
        if dest_parent not in target.parents and target != dest_parent:
            raise DataFormatError(
                f"Refusing to extract unsafe zip path: {info.filename}"
            )
    archive.extractall(dest_parent)
