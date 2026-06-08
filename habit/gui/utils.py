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
from typing import Optional, Any, List, Dict, Union
import yaml
from pydantic import ValidationError


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
