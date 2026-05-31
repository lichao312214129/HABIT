"""Tests for habit.utils.subprocess_utils (Windows CLI encoding safety)."""

from __future__ import annotations

import shutil
import sys

import pytest

from habit.core.preprocessing.dcm2niix_runner import Dcm2niixRunner
from habit.utils.subprocess_utils import decode_subprocess_bytes, run_capture_text


class TestDecodeSubprocessBytes:
    """Unit tests for multi-encoding byte decoding."""

    def test_empty_input(self) -> None:
        assert decode_subprocess_bytes(None) == ""
        assert decode_subprocess_bytes(b"") == ""

    def test_utf8(self) -> None:
        assert decode_subprocess_bytes(b"hello") == "hello"

    def test_gbk_chinese_windows_cli(self) -> None:
        text: str = "转换完成"
        assert decode_subprocess_bytes(text.encode("gbk")) == text


class TestRunCaptureText:
    """Integration tests for subprocess capture without UnicodeDecodeError."""

    def test_python_version(self) -> None:
        result = run_capture_text([sys.executable, "--version"], check=False)
        assert result.returncode == 0
        combined: str = (result.stdout or "") + (result.stderr or "")
        assert "Python" in combined

    def test_stderr_capture(self) -> None:
        result = run_capture_text(
            [sys.executable, "-c", "import sys; sys.stderr.write('stderr-ok')"],
            check=False,
        )
        assert result.returncode == 0
        assert "stderr-ok" in (result.stderr or "")


@pytest.mark.skipif(
    shutil.which("dcm2niix") is None and shutil.which("dcm2niix.exe") is None,
    reason="dcm2niix executable not available on PATH",
)
class TestDcm2niixRunnerEncoding:
    """Verify dcm2niix CLI output is captured without reader-thread decode errors."""

    def test_verify_and_version(self) -> None:
        runner = Dcm2niixRunner()
        version = run_capture_text([runner.executable, "--version"], check=False)
        combined: str = (version.stdout or "") + (version.stderr or "")
        assert "dcm2nii" in combined.lower()
        # dcm2niix may use a non-zero exit code for --version; verify() only checks availability.
        runner.verify()
