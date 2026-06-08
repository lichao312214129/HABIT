# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
#
"""
Subprocess helpers with robust text decoding for external CLI tools.

On Windows, executables such as dcm2niix and elastix often print to stderr/stdout
using the system locale (e.g. GBK/CP936). ``subprocess.run(..., text=True)`` defaults
to UTF-8 and can raise ``UnicodeDecodeError`` in the reader thread. These helpers
capture raw bytes first, then decode with a multi-encoding fallback chain.
"""

from __future__ import annotations

import locale
import subprocess
from typing import Any, List, Mapping, Optional, Sequence, Union

# Encodings commonly seen for CLI output on Windows and cross-platform setups.
_FALLBACK_ENCODINGS: tuple[str, ...] = (
    "utf-8",
    "utf-8-sig",
    "gbk",
    "cp936",
    "cp1252",
)


def _candidate_output_encodings() -> List[str]:
    """
    Build an ordered list of encodings for decoding subprocess byte streams.

    Returns:
        Encoding names: locale preference first, then common UTF-8 / Windows fallbacks.
    """
    candidates: List[str] = []
    preferred: str = locale.getpreferredencoding(False) or ""
    if preferred:
        candidates.append(preferred)
    for encoding in _FALLBACK_ENCODINGS:
        if encoding not in candidates:
            candidates.append(encoding)
    return candidates


def decode_subprocess_bytes(raw: Optional[bytes]) -> str:
    """
    Decode subprocess stdout/stderr bytes using a multi-encoding fallback chain.

    Args:
        raw: Raw byte stream from ``subprocess.run(capture_output=True, text=False)``,
            or None when the stream is empty.

    Returns:
        Decoded text. Uses UTF-8 with ``errors='replace'`` only when every candidate
        encoding fails.
    """
    if not raw:
        return ""

    for encoding in _candidate_output_encodings():
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw.decode("utf-8", errors="replace")


def run_capture_text(
    cmd: Union[str, Sequence[str]],
    *,
    shell: bool = False,
    check: bool = False,
    cwd: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout: Optional[float] = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    """
    Run a subprocess, capture stdout/stderr, and decode them as text safely.

    Unlike ``subprocess.run(..., text=True)``, this avoids ``UnicodeDecodeError``
    when external tools emit locale-encoded output (typical on Chinese Windows).

    Args:
        cmd: Command and arguments, or a shell command string when ``shell=True``.
        shell: Whether to run through the system shell.
        check: If True, raise ``CalledProcessError`` on non-zero exit code.
        cwd: Optional working directory for the child process.
        env: Optional environment mapping for the child process.
        timeout: Optional timeout in seconds.
        **kwargs: Forwarded to ``subprocess.run`` except ``capture_output`` and
            ``text``, which are managed by this helper.

    Returns:
        ``subprocess.CompletedProcess`` with ``stdout`` and ``stderr`` as ``str``.

    Raises:
        subprocess.CalledProcessError: When ``check=True`` and the process fails.
        subprocess.TimeoutExpired: When ``timeout`` is exceeded.
    """
    unsupported = {"capture_output", "text", "encoding", "errors", "universal_newlines"}
    if unsupported.intersection(kwargs):
        raise ValueError(
            "run_capture_text manages capture_output/text/encoding; "
            f"do not pass: {sorted(unsupported.intersection(kwargs))}"
        )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=False,
        shell=shell,
        check=check,
        cwd=cwd,
        env=env,
        timeout=timeout,
        **kwargs,
    )
    return subprocess.CompletedProcess(
        args=result.args,
        returncode=result.returncode,
        stdout=decode_subprocess_bytes(result.stdout),
        stderr=decode_subprocess_bytes(result.stderr),
    )
