"""
共享的 dcm2niix 执行工具，用于 DICOM 整理和转换。

这个 Module 把 dcm2niix 路径解析、可用性检查、命令格式化和进程执行集中在一处。
``dcm2nii`` conversion uses this runner; standalone DICOM rename/sort uses ``habit.core.dicom_sort`` / ``habit sort-dicom``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

from habit.utils.log_utils import get_module_logger
from habit.utils.subprocess_utils import run_capture_text


class Dcm2niixRunner:
    """运行 dcm2niix 命令，支持 DICOM 整理和 DICOM-to-NIfTI 转换。"""

    def __init__(self, dcm2niix_path: Optional[str] = None) -> None:
        self.logger = get_module_logger(__name__)
        self.executable = self._resolve_executable(dcm2niix_path)
        self.verify()

    def _resolve_executable(self, dcm2niix_path: Optional[str]) -> str:
        if not dcm2niix_path:
            return "dcm2niix"

        path = Path(dcm2niix_path)
        if path.is_file():
            executable_name = path.name
            executable_dir = path.parent
        elif path.is_dir():
            executable_name = "dcm2niix.exe" if os.name == "nt" else "dcm2niix"
            executable_dir = path
        else:
            self.logger.warning("Specified dcm2niix path does not exist: %s", dcm2niix_path)
            return "dcm2niix"

        executable_dir_str = str(executable_dir)
        current_path = os.environ.get("PATH", "")
        if executable_dir_str not in current_path:
            os.environ["PATH"] = f"{executable_dir_str}{os.pathsep}{current_path}"
            self.logger.info("Added dcm2niix path to environment: %s", executable_dir_str)

        return executable_name

    def verify(self) -> None:
        if shutil.which(self.executable) is None and not Path(self.executable).exists():
            raise RuntimeError(
                "dcm2niix executable not found: "
                f"{self.executable}. Set dcm2niix_path in the preprocessing step."
            )

        result = run_capture_text(
            [self.executable, "--version"],
            check=False,
        )
        if result.returncode != 0:
            self.logger.debug("dcm2niix --version stderr: %s", result.stderr)

    def run(
        self,
        args: Sequence[str],
        subject_id: str = "unknown_subject",
        context: str = "dcm2niix",
        shell: bool = False,
    ) -> subprocess.CompletedProcess:
        """执行 dcm2niix，args 不包含可执行文件名。"""
        cmd = [self.executable, *[str(arg) for arg in args]]
        self.logger.debug("[%s] %s command: %s", subject_id, context, self.format_command(cmd))

        if shell:
            command = self.format_command(cmd)
            result = run_capture_text(command, shell=True, check=False)
        else:
            result = run_capture_text(cmd, check=False)

        if result.stdout:
            self.logger.debug("[%s] %s stdout: %s", subject_id, context, result.stdout)
        if result.stderr:
            self.logger.debug("[%s] %s stderr: %s", subject_id, context, result.stderr)
        if result.returncode != 0:
            raise RuntimeError(
                f"{context} failed for {subject_id} with exit code "
                f"{result.returncode}: {result.stderr}"
            )

        return result

    def run_sort(self, args: Sequence[str], subject_id: str = "unknown_subject") -> subprocess.CompletedProcess:
        return self.run(args, subject_id=subject_id, context="dcm2niix DICOM sorting")

    def run_convert(self, args: Sequence[str], subject_id: str = "unknown_subject") -> subprocess.CompletedProcess:
        return self.run(args, subject_id=subject_id, context="dcm2niix conversion", shell=True)

    @staticmethod
    def format_command(cmd: Sequence[str]) -> str:
        formatted: List[str] = []
        for part in cmd:
            text = str(part)
            if " " in text and not (text.startswith('"') and text.endswith('"')):
                formatted.append(f'"{text}"')
            else:
                formatted.append(text)
        return " ".join(formatted)
