"""
Run elastix and transformix CLIs (external executables).

This mirrors :class:`Dcm2niixRunner`: resolve executable paths, build subprocess
arguments, and surface stderr on failure. Command-line flags follow elastix 5.x
documentation (``-f``, ``-m``, ``-out``, ``-p``, ``-fMask``, ``-mMask`` for elastix;
``-in``, ``-out``, ``-tp`` for transformix).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

from habit.utils.log_utils import get_module_logger


def format_elastix_parameter_value(value: Any) -> str:
    """Format a Python value as elastix parameter file token(s) after the key name.

    Elastix lines look like ``(NumberOfSpatialSamples 2048)`` or
    ``(Metric AdvancedMattesMutualInformation)``. Lists/tuples become
    space-separated tokens; scalars become a single token.

    Args:
        value: Scalar, string, or sequence of values from YAML / Python overrides.

    Returns:
        str: Text placed after the parameter name inside the parentheses.
    """
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def merge_elastix_parameter_file_text(
    content: str,
    spatial_dim: int,
    overrides: Optional[dict] = None,
) -> str:
    """Apply dimension forcing and user overrides to one elastix parameter file body.

    Parses lines of the form ``(ParameterName token token ...)``, replaces or appends
    keys present in ``overrides``, and always sets ``FixedImageDimension`` /
    ``MovingImageDimension`` to ``spatial_dim`` (elastix manual: dimension must match images).

    Args:
        content: Full text of an elastix ``.txt`` parameter file.
        spatial_dim: Number of spatial dimensions (2 or 3 for typical medical volumes).
        overrides: Optional map parameter name -> scalar, string, or list of values.

    Returns:
        str: New file body ending with a newline.
    """
    overrides = dict(overrides or {})
    overrides["FixedImageDimension"] = spatial_dim
    overrides["MovingImageDimension"] = spatial_dim

    pattern = re.compile(r"^\s*\(\s*([A-Za-z0-9_]+)\s+(.*)\)\s*\r?$")
    out_lines: List[str] = []
    consumed: set[str] = set()

    for line in content.splitlines():
        match = pattern.match(line)
        if not match:
            out_lines.append(line)
            continue
        key = match.group(1)
        if key in overrides:
            payload = format_elastix_parameter_value(overrides[key])
            out_lines.append(f"({key} {payload})")
            consumed.add(key)
        else:
            out_lines.append(line)

    for key, raw_val in overrides.items():
        if key not in consumed:
            payload = format_elastix_parameter_value(raw_val)
            out_lines.append(f"({key} {payload})")

    text = "\n".join(out_lines)
    if text and not text.endswith("\n"):
        text += "\n"
    elif not text:
        text = "\n"
    return text


class ElastixCliRunner:
    """Locate elastix / transformix executables and run them via subprocess."""

    def __init__(
        self,
        elastix_path: Optional[str] = None,
        transformix_path: Optional[str] = None,
    ) -> None:
        self.logger = get_module_logger(__name__)
        self.elastix = self._resolve_executable(elastix_path, "elastix")
        self.transformix = self._resolve_executable(transformix_path, "transformix")

    def _candidate_paths_in_dir(self, directory: Path, basename: str) -> List[Path]:
        """Build ordered paths to probe inside a user-provided install directory.

        On Windows the binary on disk is usually ``tool.exe``; ``subprocess`` can still
        launch ``tool`` when resolving via PATH (``PATHEXT``). For a concrete directory we
        probe ``.exe`` first, then extensionless ``tool``, and return the first path that
        exists.

        Args:
            directory: Folder that should contain the elastix / transformix binary.
            basename: ``elastix`` or ``transformix``.

        Returns:
            List[Path]: Paths to test with ``Path.is_file()``.
        """
        paths: List[Path] = []
        if os.name == "nt":
            paths.append(directory / f"{basename}.exe")
        paths.append(directory / basename)
        return paths

    def _resolve_executable(self, user_path: Optional[str], basename: str) -> str:
        """Resolve ``basename`` to a command name or path, similar to dcm2niix.

        When no explicit path is given, returns ``basename`` only (no ``.exe``). On Windows,
        ``CreateProcess`` / ``subprocess`` still resolve ``elastix`` → ``elastix.exe`` via
        ``PATHEXT`` when the binary is on ``PATH``.

        If ``user_path`` is a file, use that path. If it is a directory, search for the
        binary inside it, prepend that directory to ``PATH`` (for DLLs on Windows), and
        return the resolved file path if found; otherwise fall back to ``basename`` on
        PATH.

        Args:
            user_path: Optional file or directory from YAML.
            basename: Tool name without extension (``elastix`` or ``transformix``).

        Returns:
            str: Executable name or path for subprocess.
        """
        if not user_path:
            return basename

        path = Path(user_path)
        if path.is_file():
            executable_dir_str = str(path.parent)
            current_path = os.environ.get("PATH", "")
            if executable_dir_str not in current_path:
                os.environ["PATH"] = f"{executable_dir_str}{os.pathsep}{current_path}"
                self.logger.info("Prepended to PATH for %s: %s", basename, executable_dir_str)
            return str(path)

        if path.is_dir():
            executable_dir_str = str(path.resolve())
            current_path = os.environ.get("PATH", "")
            if executable_dir_str not in current_path:
                os.environ["PATH"] = f"{executable_dir_str}{os.pathsep}{current_path}"
                self.logger.info("Prepended to PATH for %s: %s", basename, executable_dir_str)
            for candidate in self._candidate_paths_in_dir(path, basename):
                if candidate.is_file():
                    return str(candidate.resolve())
            return basename

        self.logger.warning("%s path does not exist: %s; trying PATH.", basename, user_path)
        return basename

    def _run(
        self,
        executable: str,
        args: Sequence[str],
        context: str,
    ) -> subprocess.CompletedProcess:
        cmd: List[str] = [executable, *[str(a) for a in args]]
        self.logger.debug("%s command: %s", context, self._format_command(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.stdout:
            self.logger.debug("%s stdout: %s", context, result.stdout[:4000])
        if result.stderr:
            self.logger.debug("%s stderr: %s", context, result.stderr[:4000])
        if result.returncode != 0:
            raise RuntimeError(
                f"{context} failed (exit {result.returncode}): "
                f"{(result.stderr or result.stdout or '').strip()}"
            )
        return result

    @staticmethod
    def _format_command(cmd: Sequence[str]) -> str:
        formatted: List[str] = []
        for part in cmd:
            text = str(part)
            if " " in text and not (text.startswith('"') and text.endswith('"')):
                formatted.append(f'"{text}"')
            else:
                formatted.append(text)
        return " ".join(formatted)

    def run_elastix(
        self,
        fixed_image: Union[str, Path],
        moving_image: Union[str, Path],
        out_dir: Union[str, Path],
        parameter_files: Sequence[Union[str, Path]],
        fixed_mask: Optional[Union[str, Path]] = None,
        moving_mask: Optional[Union[str, Path]] = None,
        threads: Optional[int] = None,
    ) -> None:
        """Run ``elastix`` with mandatory ``-f``, ``-m``, ``-out``, and one or more ``-p``.

        Args:
            fixed_image: Path to fixed (reference) image.
            moving_image: Path to moving image.
            out_dir: Output directory (created by caller if needed).
            parameter_files: One or more elastix parameter ``.txt`` paths (sequential stages).
            fixed_mask: Optional fixed mask path (``-fMask``).
            moving_mask: Optional moving mask path (``-mMask``).
            threads: Optional ``-threads`` count.
        """
        args: List[str] = [
            "-f",
            str(fixed_image),
            "-m",
            str(moving_image),
            "-out",
            str(out_dir),
        ]
        if fixed_mask is not None:
            args.extend(["-fMask", str(fixed_mask)])
        if moving_mask is not None:
            args.extend(["-mMask", str(moving_mask)])
        for pf in parameter_files:
            args.extend(["-p", str(pf)])
        if threads is not None and threads > 0:
            args.extend(["-threads", str(int(threads))])

        if shutil.which(self.elastix) is None and not Path(self.elastix).is_file():
            raise RuntimeError(
                f"elastix executable not found: {self.elastix}. "
                "Set elastix_path in registration config (directory or full path to elastix)."
            )
        self._run(self.elastix, args, context="elastix")

    def run_transformix(
        self,
        input_image: Union[str, Path],
        out_dir: Union[str, Path],
        transform_parameters: Union[str, Path],
        threads: Optional[int] = None,
    ) -> None:
        """Run ``transformix`` with ``-in``, ``-out``, and ``-tp``.

        Args:
            input_image: Image to resample (e.g. moving mask).
            out_dir: Directory for transformix outputs (e.g. ``result.nii``).
            transform_parameters: Path to ``TransformParameters.N.txt`` (final stage).
            threads: Optional ``-threads`` count.
        """
        args: List[str] = [
            "-in",
            str(input_image),
            "-out",
            str(out_dir),
            "-tp",
            str(transform_parameters),
        ]
        if threads is not None and threads > 0:
            args.extend(["-threads", str(int(threads))])

        if shutil.which(self.transformix) is None and not Path(self.transformix).is_file():
            raise RuntimeError(
                f"transformix executable not found: {self.transformix}. "
                "Set transformix_path in registration config."
            )
        self._run(self.transformix, args, context="transformix")
