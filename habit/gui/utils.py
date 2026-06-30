# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""
GUI-specific utilities.
Contains file dialog helpers, validation error translators.
This module is self-contained under habit/gui directory.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Any, List, Dict, Union

import gradio as gr
import yaml
from pydantic import ValidationError


def gui_draft_dir() -> Path:
    """
    Return the directory used to store per-tab GUI draft paths.

    The directory is created on first access so callers never need to mkdir
    themselves.  Using the user home keeps drafts isolated per OS user and
    independent of the project root or conda environment.

    Returns:
        Path: ``~/.habit/gui_drafts/``, guaranteed to exist.
    """
    draft_dir = Path.home() / ".habit" / "gui_drafts"
    draft_dir.mkdir(parents=True, exist_ok=True)
    return draft_dir


def save_gui_draft(tab_name: str, config_path: str) -> None:
    """
    Persist the path of the last saved config YAML for a GUI tab.

    Called immediately after the config is written to disk so that the next
    page load can auto-restore all form fields from that YAML.

    Args:
        tab_name: Short identifier for the tab, e.g. ``"extract"``.
        config_path: Absolute path to the config YAML that was just saved.
    """
    try:
        draft_file = gui_draft_dir() / f"{tab_name}.txt"
        draft_file.write_text(str(config_path), encoding="utf-8")
    except OSError:
        pass  # Non-critical; ignore write failures silently.


def load_gui_draft(tab_name: str) -> Optional[str]:
    """
    Return the path stored by the last ``save_gui_draft`` call for a tab.

    Returns ``None`` when no draft exists or the stored path no longer points
    to an existing file (e.g. the user deleted the output folder).

    Args:
        tab_name: Short identifier matching the one used in ``save_gui_draft``.

    Returns:
        Optional[str]: Absolute path string if the file exists, else ``None``.
    """
    try:
        draft_file = gui_draft_dir() / f"{tab_name}.txt"
        if not draft_file.exists():
            return None
        path = draft_file.read_text(encoding="utf-8").strip()
        return path if (path and os.path.isfile(path)) else None
    except OSError:
        return None


def coerce_str_list(items: Any) -> List[str]:
    """
    Convert a YAML-loaded sequence (possibly containing bytes) to str list.

    Args:
        items: Raw value from YAML (list, tuple, or scalar).

    Returns:
        List[str]: Normalized string tokens.
    """
    if items is None:
        return []
    if isinstance(items, (list, tuple)):
        return [str(x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x) for x in items]
    if isinstance(items, bytes):
        return [items.decode("utf-8", errors="replace")]
    return [str(items)]


def _list_modality_folder_names(images_root: str) -> List[str]:
    """
    Collect unique modality folder names under ``images_root/<subject>/<modality>/``.

    Args:
        images_root: Directory whose immediate children are subject folders.

    Returns:
        List[str]: Sorted unique modality folder names.
    """
    modality_names: set[str] = set()
    if not os.path.isdir(images_root):
        return []
    for subject_name in os.listdir(images_root):
        if subject_name.startswith("."):
            continue
        subject_path = os.path.join(images_root, subject_name)
        if not os.path.isdir(subject_path):
            continue
        for modality_name in os.listdir(subject_path):
            if modality_name.startswith("."):
                continue
            modality_path = os.path.join(subject_path, modality_name)
            if os.path.isdir(modality_path):
                modality_names.add(modality_name)
    return sorted(modality_names)


def _looks_like_images_root(path: str) -> bool:
    """
    Return True when ``path/<subject>/<modality>/`` folders exist (images root layout).

    Args:
        path: Candidate images root directory.

    Returns:
        bool: True when at least one subject/modality folder pair is found.
    """
    if not os.path.isdir(path):
        return False
    for subject_name in os.listdir(path):
        if subject_name.startswith("."):
            continue
        subject_path = os.path.join(path, subject_name)
        if not os.path.isdir(subject_path):
            continue
        for modality_name in os.listdir(subject_path):
            if modality_name.startswith("."):
                continue
            if os.path.isdir(os.path.join(subject_path, modality_name)):
                return True
    return False


def discover_modalities_from_data_dir(
    data_dir: str,
    auto_select_first_file: bool = True,
) -> tuple[List[str], str]:
    """
    Scan ``data_dir`` for modality folder names.

    Supported layouts:
    - ``data_dir/images/<subject>/<modality>/`` (habit_default)
    - ``data_dir/<subject>/<modality>/`` when ``data_dir`` itself is the images root
    - YAML manifest passed as ``data_dir`` (``images: {subj: {modality: path}}``)

    When the directory follows a flat subject-root DICOM layout (``data_dir/<subject>/`` with no
    modality subfolders), modality folder names cannot be inferred; callers should use ``dicom``
    for the dcm2nii step in that case.

    Args:
        data_dir: Dataset root directory or YAML manifest path entered in the GUI.
        auto_select_first_file: Passed through to ``get_image_and_mask_paths``.

    Returns:
        tuple[List[str], str]: Sorted unique modality keys and a short status message for the GUI.
    """
    if not data_dir or not str(data_dir).strip():
        return [], "Set input data root directory to detect modalities."

    data_path = abs_path(str(data_dir).strip())

    if os.path.isfile(data_path) and data_path.lower().endswith((".yaml", ".yml")):
        try:
            from habit.utils.io_utils import detect_image_names, get_image_and_mask_paths

            images_paths, _ = get_image_and_mask_paths(
                data_path,
                auto_select_first_file=auto_select_first_file,
            )
            if images_paths:
                modalities = detect_image_names(images_paths)
                if modalities:
                    return modalities, (
                        f"Detected {len(modalities)} modality key(s) from YAML manifest: "
                        f"{', '.join(modalities)}"
                    )
        except OSError as exc:
            return [], f"Failed to read YAML manifest: {exc}"
        except Exception as exc:  # noqa: BLE001 — surface scan errors in GUI
            return [], f"Failed to read YAML manifest: {exc}"
        return [], f"No modalities found in YAML manifest: {data_path}"

    if not os.path.isdir(data_path):
        return [], f"Directory not found: {data_path}"

    images_root = os.path.join(data_path, "images")
    scan_roots: List[str] = []
    if os.path.isdir(images_root):
        scan_roots.append(images_root)
    if _looks_like_images_root(data_path):
        scan_roots.append(data_path)

    for root in scan_roots:
        try:
            if root == images_root:
                from habit.utils.io_utils import detect_image_names, get_image_and_mask_paths

                images_paths, _ = get_image_and_mask_paths(
                    data_path,
                    auto_select_first_file=auto_select_first_file,
                )
                if images_paths:
                    modalities = detect_image_names(images_paths)
                    if modalities:
                        return modalities, (
                            f"Detected {len(modalities)} modality folder(s) from images/: "
                            f"{', '.join(modalities)}"
                        )
        except OSError:
            pass
        except Exception:  # noqa: BLE001 — fall back to folder-name scan below
            pass

        folder_modalities = _list_modality_folder_names(root)
        if folder_modalities:
            label = "images/" if root == images_root else "data_dir"
            return folder_modalities, (
                f"Detected {len(folder_modalities)} modality folder(s) under {label}: "
                f"{', '.join(folder_modalities)}"
            )

    subject_dirs = [
        name
        for name in os.listdir(data_path)
        if not name.startswith(".") and os.path.isdir(os.path.join(data_path, name))
    ]
    if subject_dirs:
        return [], (
            "Subject-root DICOM layout detected (no images/ tree). "
            'Use modality key "dicom" for the dcm2nii step.'
        )

    return [], f"No modalities found under {data_path}/images/<subject>/ or {data_path}/<subject>/."


def parse_comma_list(value: str) -> List[str]:
    """
    Parse a comma-separated string into a trimmed list of non-empty tokens.

    Args:
        value: Raw comma-separated user input.

    Returns:
        List[str]: Parsed tokens; empty list when input is blank.
    """
    if not value or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def open_directory(path: str) -> None:
    """
    Open a local directory in the OS file explorer.

    Args:
        path: Absolute or relative directory path.
    """
    if not path or not os.path.exists(path):
        return
    try:
        if os.name == "nt":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as exc:  # noqa: BLE001 — best-effort folder open
        print(f"Unable to open folder: {exc}")


def yaml_block_to_dict(yaml_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse a YAML snippet (typically a nested config block) into a dictionary.

    Args:
        yaml_text: Multiline YAML string from a Gradio textbox.

    Returns:
        Optional[Dict[str, Any]]: Parsed dict, or None when empty/invalid.
    """
    if not yaml_text or not str(yaml_text).strip():
        return None
    try:
        parsed = yaml.safe_load(yaml_text)
        if parsed is None:
            return None
        if not isinstance(parsed, dict):
            raise ValueError("YAML block must be a mapping/dict at the top level.")
        return parsed
    except Exception as exc:
        raise ValueError(f"Invalid YAML block: {exc}") from exc


def dict_to_yaml_block(data: Optional[Dict[str, Any]]) -> str:
    """
    Serialize a dictionary to a compact YAML string for Gradio text areas.

    Args:
        data: Mapping to serialize; None yields empty string.

    Returns:
        str: YAML text suitable for display/editing.
    """
    if not data:
        return ""
    return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)

def select_local_path(select_type: str = "folder", title: str = "Select Path") -> Optional[str]:
    """
    Launch a native Windows / OS-level file or folder chooser dialog using tkinter.
    This avoids web browser sandbox limitations and allows users to select local paths.

    Args:
        select_type (str): Either "folder" to select directory or "file" to select a file.
        title (str): Title of the dialog window.

    Returns:
        Optional[str]: Absolute path selected by the user, or None if cancelled.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        # Create a hidden root window
        root = tk.Tk()
        root.withdraw()
        # Bring dialog to the front
        root.attributes("-topmost", True)

        selected_path: str = ""
        if select_type == "folder":
            selected_path = filedialog.askdirectory(title=title)
        elif select_type == "file":
            selected_path = filedialog.askopenfilename(
                title=title,
                filetypes=[("YAML Configuration", "*.yaml *.yml"), ("All Files", "*.*")]
            )

        root.destroy()
        if selected_path:
            return os.path.abspath(selected_path)
    except Exception as e:
        # Fallback if tkinter is not installed or GUI context is unavailable
        print(f"Unable to launch file dialog: {e}. Please enter the path manually.")
    return None


_BUNDLED_RADIOMICS_DIR: Path = Path(__file__).resolve().parent / "resources" / "radiomics"


def default_radiomics_param(filename: str) -> str:
    """
    Resolve a bundled default PyRadiomics parameter file shipped inside the package.

    The GUI must work out-of-the-box in the portable conda-pack distribution, where
    the project-level ``config/radiomics/`` folder is shipped separately and its
    location is unknown. Bundling the default parameter files inside the package and
    resolving them by absolute path here avoids the previous broken relative default
    (``../radiomics/...``) that only worked from one specific working directory.

    Args:
        filename: Parameter file name, e.g. ``"params_voxel_radiomics.yaml"``.

    Returns:
        str: Absolute path to the bundled file. The path is returned even when the
            file is missing so the caller can surface a clear "file not found" error.
    """
    return str((_BUNDLED_RADIOMICS_DIR / filename).resolve())


def abs_path(val: str) -> str:
    """
    Convert a path string to an absolute path using the current working directory.
    If the path is already absolute or empty, it is returned as-is.

    This is used to normalize user-entered paths in the GUI before saving them
    to the config YAML, so that the pipeline's path-resolution logic (which
    resolves relative paths relative to the YAML file directory) does not
    accidentally map user paths to the wrong location.

    Args:
        val: Path string that may be relative or absolute.

    Returns:
        str: Absolute path string, or the original value if it is empty/blank.
    """
    if not val or not str(val).strip():
        return val
    p = str(val).strip()
    if os.path.isabs(p):
        return p
    return os.path.abspath(p)


def user_visible_path(val: str) -> str:
    """
    Format an internal/runtime path for display in GUI text boxes (Docker: F:\\...).

    Args:
        val: Path string from YAML or filesystem.

    Returns:
        str: User-visible path.
    """
    from habit.utils.docker_path_utils import to_user_visible_path

    if not val or not str(val).strip():
        return val
    return to_user_visible_path(str(val).strip())


def extract_validation_msgs(exc: Exception) -> Optional[List[str]]:
    """
    Extract user-friendly validation messages from a Pydantic ``ValidationError``
    or from any exception whose ``__cause__`` is a ``ValidationError``
    (e.g. ``ConfigValidationError`` raised by ``BaseConfig.__init__``).

    ``BaseConfig.__init__`` wraps Pydantic's ``ValidationError`` in a custom
    ``ConfigValidationError``.  GUI ``except ValidationError`` handlers therefore
    miss it; calling this function lets the ``except Exception`` fallback still
    produce friendly translated messages rather than a raw exception string.

    Args:
        exc: Any exception caught in a GUI tab run function.

    Returns:
        List[str] of translated messages when ``exc`` is (or wraps) a
        Pydantic ``ValidationError``, or ``None`` when it is not.
    """
    if isinstance(exc, ValidationError):
        return translate_pydantic_error(exc)
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, ValidationError):
        return translate_pydantic_error(cause)
    return None


def translate_pydantic_error(err: ValidationError) -> List[str]:
    """
    Translate raw Pydantic ValidationErrors into clear, user-friendly Chinese messages
    specifically tailored for clinicians and medical researchers.

    Args:
        err (ValidationError): The validation error raised by Pydantic.

    Returns:
        List[str]: Translated Chinese error descriptions.
    """
    translated: List[str] = []
    
    # Map common configuration keys to friendly Chinese terms
    fields_map: Dict[str, str] = {
        "data_dir": "数据输入目录/文件",
        "out_dir": "输出结果保存目录",
        "output_dir": "输出保存目录",
        "output": "模型输出保存目录",
        "pipeline_path": "已存模型流水线路径 (*_final_pipeline.pkl)",
        "run_mode": "运行模式 (train / predict)",
        "test_size": "测试集占比 (0 < test_size < 1)",
        "split_method": "数据集拆分方法",
        "processes": "CPU并行进程数",
        "n_splits": "K-Fold交叉验证折数",
        "random_state": "随机种子",
        "f": "DICOM重命名命名格式 (-f)",
        "filename_format": "DICOM命名格式",
        "input": "输入文件配置列表",
        "models": "候选待训练机器学习模型",
        "FeatureConstruction": "特质构建配置 (FeatureConstruction)",
        "HabitatSegmentation": "栖息地分割配置 (HabitatSegmentation)",
        "clustering_mode": "影像生境聚类分割模式 (one_step / two_step / direct_pooling)",
    }

    for error in err.errors():
        loc_path: List[Union[str, int]] = error["loc"]
        err_type: str = error["type"]
        raw_msg: str = error["msg"]

        # Extract the field name
        field_name: str = "未知参数"
        if loc_path:
            # Try to map the last part or whole path if simple
            last_part: str = str(loc_path[-1])
            field_name = fields_map.get(last_part, last_part)

        # Map error types to human-friendly Chinese error templates
        if err_type == "missing" or "value_error.missing" in err_type:
            translated.append(f"【缺失必填项】: 请填写/设置「{field_name}」。")
        elif "greater_than" in err_type or "less_than" in err_type or "range" in err_type:
            translated.append(f"【数值不合规】: 「{field_name}」的值不符合范围要求。具体要求: {raw_msg}。")
        elif "enum" in err_type or "literal" in err_type:
            translated.append(f"【无效的选项】: 「{field_name}」的值不符合预设选项。具体要求: {raw_msg}。")
        elif "value_error" in err_type:
            translated.append(f"【参数冲突或越界】: 「{field_name}」验证未通过。提示: {raw_msg}。")
        else:
            translated.append(f"【输入格式错误】: 「{field_name}」配置有误。详情: {raw_msg}。")

    return translated


def render_console_log(
    label: str = "Console log",
    lines: int = 18,
    *,
    elem_id: Optional[str] = None,
) -> gr.Textbox:
    """
    Create a fixed-height console log textbox that does not hijack page scroll.

    Gradio Textbox defaults to ``autoscroll=True``, which re-scrolls the textarea
    (and often the whole page) on every streamed update. Progress bars (tqdm)
    update several times per second, so users cannot read other controls while a
    job runs. ``autoscroll=False`` plus a fixed ``max_lines`` keeps the log area
    stable so manual scrolling is preserved.

    Args:
        label: Widget label shown above the log area.
        lines: Visible row count for the textarea.
        elem_id: Optional unique DOM id (one per tab; avoid duplicate ids).

    Returns:
        gr.Textbox: Non-interactive log widget configured for streaming jobs.
    """
    kwargs: Dict[str, Any] = {
        "label": label,
        "lines": lines,
        "max_lines": lines,
        "interactive": False,
        "autoscroll": False,
        "elem_classes": ["habit-console-log"],
    }
    if elem_id:
        kwargs["elem_id"] = elem_id
    return gr.Textbox(**kwargs)


def read_pipeline_log(log_path: Union[str, Path]) -> str:
    """
    Read a pipeline log file for GUI display.

    Args:
        log_path: Path to ``processing.log``, ``habitat_analysis.log``, etc.

    Returns:
        str: Log text, or empty string when the file is missing/unreadable.
    """
    try:
        path = Path(log_path)
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def save_config_yaml(data: Dict[str, Any], path: str) -> None:
    """
    Saves a configuration dictionary to a local YAML file.

    Args:
        data (Dict[str, Any]): Configuration content.
        path (str): File path to save configuration.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def load_config_yaml(path: str) -> Optional[Dict[str, Any]]:
    """
    Loads configuration dictionary from a local YAML file.

    Args:
        path (str): File path to load.

    Returns:
        Optional[Dict[str, Any]]: Loaded configuration dictionary or None if error.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"无法解析配置文件 {path}: {e}")
        return None
