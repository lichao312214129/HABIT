"""
DICOM sorting preprocessor powered by dcm2niix rename mode.

This step uses ``dcm2niix -r y`` to reorganize raw DICOM files without
converting them to NIfTI. It is useful before image conversion when the source
DICOM directory is flat or vendor-organized in an inconvenient way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base_preprocessor import BasePreprocessor
from .dcm2niix_runner import Dcm2niixRunner
from .preprocessor_factory import PreprocessorFactory
from habit.utils.file_system_utils import safe_mkdir
from habit.utils.log_utils import get_module_logger


@PreprocessorFactory.register("sort_dicom")
class DicomSorter(BasePreprocessor):
    """
    Sort/rename DICOM files with dcm2niix rename mode.

    The interface intentionally exposes the common dcm2niix options needed for
    DICOM organization, while ``extra_args`` keeps an escape hatch for less
    common flags.
    """

    DEFAULT_FILENAME_FORMAT = "%n_%g_%x/%s_%d/%r_%o.dcm"

    def __init__(
        self,
        keys: Union[str, List[str]],
        dcm2niix_path: Optional[str] = None,
        filename_format: str = DEFAULT_FILENAME_FORMAT,
        output_dir: Optional[str] = None,
        directory_depth: Optional[int] = None,
        adjacent_dicoms: Optional[bool] = None,
        ignore_derived: Optional[bool] = None,
        verbose: Union[bool, int, str] = False,
        write_behavior: Optional[int] = None,
        progress: Optional[bool] = None,
        extra_args: Optional[List[str]] = None,
        update_data_paths: bool = False,
        allow_missing_keys: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the DICOM sorter.

        Args:
            keys: Data keys containing input DICOM directories.
            dcm2niix_path: Full path to dcm2niix executable or its directory.
            filename_format: dcm2niix ``-f`` pattern. Default matches the
                project convention: ``%n_%g_%x/%s_%d/%r_%o.dcm``.
            output_dir: Optional root output directory. If omitted, the
                pipeline's ``output_dirs`` entry for the key is used.
            directory_depth: Optional ``-d`` value, 0..9.
            adjacent_dicoms: Optional ``-a`` flag.
            ignore_derived: Optional ``-i`` flag.
            verbose: ``-v`` value. ``False`` -> ``n``, ``True`` -> ``y``.
            write_behavior: Optional ``-w`` value, 0/1/2.
            progress: Optional ``--progress`` flag.
            extra_args: Extra raw dcm2niix arguments, e.g. ``["--terse"]``.
            update_data_paths: If True, replace ``data[key]`` with the sorted
                output directory. Default False because this step normally only
                organizes files on disk.
            allow_missing_keys: Allow absent keys in a subject entry.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.logger = get_module_logger(__name__)
        self.runner = Dcm2niixRunner(dcm2niix_path)
        self.filename_format = filename_format
        self.output_dir = output_dir
        self.directory_depth = directory_depth
        self.adjacent_dicoms = adjacent_dicoms
        self.ignore_derived = ignore_derived
        self.verbose = verbose
        self.write_behavior = write_behavior
        self.progress = progress
        self.extra_args = list(extra_args or [])
        self.update_data_paths = update_data_paths

    @staticmethod
    def _yn(value: bool) -> str:
        return "y" if value else "n"

    def _verbose_value(self) -> str:
        if isinstance(self.verbose, bool):
            return self._yn(self.verbose)
        return str(self.verbose)

    def _build_command(self, input_dir: str, output_dir: str) -> List[str]:
        cmd = [
            "-r",
            "y",
            "-f",
            self.filename_format,
        ]

        if self.directory_depth is not None:
            cmd.extend(["-d", str(self.directory_depth)])
        if self.adjacent_dicoms is not None:
            cmd.extend(["-a", self._yn(self.adjacent_dicoms)])
        if self.ignore_derived is not None:
            cmd.extend(["-i", self._yn(self.ignore_derived)])
        if self.verbose is not False:
            cmd.extend(["-v", self._verbose_value()])
        if self.write_behavior is not None:
            cmd.extend(["-w", str(self.write_behavior)])
        if self.progress is not None:
            cmd.extend(["--progress", self._yn(self.progress)])

        cmd.extend(self.extra_args)
        cmd.extend(["-o", output_dir, input_dir])
        return cmd


    def _resolve_output_dir(self, data: Dict[str, Any], key: str, subject_id: str) -> str:
        if self.output_dir:
            output_root = Path(self.output_dir)
            output_path = output_root / subject_id / key
        elif "output_dirs" in data and key in data["output_dirs"]:
            output_path = Path(data["output_dirs"][key])
        else:
            output_path = Path.cwd() / "sorted_dicom" / subject_id / key

        safe_mkdir(str(output_path))
        return str(output_path)

    def _sort_one(self, input_dir: str, output_dir: str, subject_id: str, key: str) -> None:
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input DICOM directory does not exist: {input_dir}")
        if not input_path.is_dir():
            raise ValueError(f"Input DICOM path is not a directory: {input_dir}")

        cmd = self._build_command(str(input_path), output_dir)
        self.logger.info("[%s] Sorting DICOM files for %s into %s", subject_id, key, output_dir)
        self.runner.run_sort(cmd, subject_id=f"{subject_id}/{key}")

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self._check_keys(data)
        subject_id = data.get("subj", data.get("subject_id", "unknown_subject"))

        for key in self.keys:
            if key not in data:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"Key {key} not found in data dictionary")

            value = data[key]
            if not isinstance(value, str):
                self.logger.info("[%s] Key %s is not a path string, skipping sort_dicom", subject_id, key)
                continue

            output_dir = self._resolve_output_dir(data, key, subject_id)
            self._sort_one(value, output_dir, subject_id, key)

            meta_key = f"{key}_meta_dict"
            if meta_key not in data:
                data[meta_key] = {}
            data[meta_key].update(
                {
                    "dicom_sorted": True,
                    "original_dicom_dir": value,
                    "sorted_dicom_dir": output_dir,
                    "filename_format": self.filename_format,
                }
            )
            if self.update_data_paths:
                data[key] = output_dir

        return data
