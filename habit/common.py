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
"""Shared helpers for HABIT CLI command implementations."""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

import click
import yaml

from habit.utils.log_utils import setup_logger

TConfig = TypeVar("TConfig")

CONFIG_REQUIRED_MSG: str = (
    "Error: Configuration file is required. Use --config / -c."
)

# Short bilingual tips shown when YAML fails to parse (clinician-oriented).
_YAML_EDIT_TIPS_ZH_EN: str = (
    "【YAML 修改提示 / YAML tips】\n"
    "  - 用空格缩进，不要用 Tab；冒号后面必须有一个空格：key: value\n"
    "  - true / false / null 请用小写\n"
    "  - 路径一般不用加引号；仅当路径含空格或特殊字符（如 # : *）时才加引号\n"
    "      例：D:/data/images 可不加；\"D:/my data/images\" 有空格必须加\n"
    "  - Windows 路径推荐正斜杠 /（如 D:/work/...）\n"
    "  - 用记事本或 VS Code 编辑并保存为 .yaml，不要用 Word\n"
    "  - 改完可先运行: habit check-config -c <本文件>"
)


def format_config_load_error(exc: BaseException, config_path: str) -> str:
    """
    Format a config load/validation failure for CLI users (bilingual tips).

    Args:
        exc: Exception raised while loading or validating the config.
        config_path: Path to the YAML file being loaded.

    Returns:
        Multi-line error message suitable for ``exit_with_error``.
    """
    lines = [
        f"Error: 配置加载失败 / Failed to load configuration: {config_path}",
        f"原因 / Cause: {exc}",
    ]

    if isinstance(exc, yaml.YAMLError):
        lines.append("")
        lines.append("这通常是 YAML 格式写错了（缩进、冒号、引号等）。")
        # PyYAML marks often expose line/column for ScannerError/ParserError.
        mark = getattr(exc, "problem_mark", None)
        if mark is not None:
            lines.append(
                f"出错位置约在 / near: 第 {mark.line + 1} 行, "
                f"第 {mark.column + 1} 列"
            )
        lines.append("")
        lines.append(_YAML_EDIT_TIPS_ZH_EN)
    else:
        # Schema / path / other errors — still remind about common edit pitfalls.
        msg = str(exc).lower()
        if any(
            token in msg
            for token in (
                "validation",
                "field required",
                "extra inputs",
                "value error",
                "missing",
            )
        ):
            lines.append("")
            lines.append(
                "提示: 请检查「★ 必改」字段是否填写、拼写是否与注释一致；"
                "不要删掉必填键，也不要新增未知键名。"
            )
            lines.append("可用: habit check-config -c <本文件> 做检查（不跑完整流程）。")

    return "\n".join(lines)


def require_config_path(config_path: Optional[str]) -> str:
    """
    Validate that a config path was provided on the CLI.

    Args:
        config_path: Path from Click (may be None when optional at decorator level).

    Returns:
        Non-empty config path string.

    Raises:
        SystemExit: When ``config_path`` is missing or blank.
    """
    if not config_path or not str(config_path).strip():
        exit_with_error(CONFIG_REQUIRED_MSG)
    return str(config_path)


def load_config_or_exit(config_cls: Type[TConfig], config_path: str) -> TConfig:
    """
    Load a typed config via ``config_cls.from_file`` with uniform CLI errors.

    Args:
        config_cls: Pydantic config class exposing ``from_file``.
        config_path: Path to the YAML configuration file.

    Returns:
        Validated config instance.

    Raises:
        SystemExit: On load/validation failure.
    """
    path = require_config_path(config_path)
    if not Path(path).is_file():
        exit_with_error(
            f"Error: 找不到配置文件 / Configuration file not found: {path}"
        )
    try:
        return config_cls.from_file(path)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        exit_with_error(format_config_load_error(exc, path))


def echo_error(message: str) -> None:
    """Print a CLI error line to stderr."""
    click.echo(message, err=True)


def echo_success(message: str) -> None:
    """Print a standardized success line."""
    # ASCII prefix keeps Windows GBK consoles from crashing on Unicode checkmarks.
    try:
        click.secho(f"✓ {message}", fg="green")
    except UnicodeEncodeError:
        click.secho(f"[OK] {message}", fg="green")


def exit_with_error(message: str, *, exit_code: int = 1) -> None:
    """
    Echo an error and terminate the process.

    Args:
        message: User-facing error text.
        exit_code: Process exit code (default 1).
    """
    echo_error(message)
    sys.exit(exit_code)


def run_cli_job(
    *,
    logger_name: str,
    output_dir: Path,
    log_filename: str,
    level: int = logging.INFO,
    start_message: str,
    job: Any,
    success_message: str,
) -> None:
    """
    Run a core job with consistent logging and CLI feedback.

    Args:
        logger_name: Logger name passed to ``setup_logger``.
        output_dir: Directory for log files (created if missing).
        log_filename: Log file basename under ``output_dir``.
        level: Logging level.
        start_message: Echoed when the job starts.
        job: Zero-argument callable executed inside the try block.
        success_message: Echoed on success (prefixed with ✓).

    Raises:
        SystemExit: When ``job`` raises.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        name=logger_name,
        output_dir=output_dir,
        log_filename=log_filename,
        level=level,
    )
    logger.info(start_message)
    click.echo(start_message)
    try:
        job()
    except Exception as exc:  # noqa: BLE001
        logger.error("Job failed: %s", exc, exc_info=True)
        exit_with_error(f"Error: {exc}")
    logger.info(success_message)
    echo_success(success_message)


def log_platform_info(logger: logging.Logger, config_path: str) -> None:
    """
    Log Python/platform metadata at CLI startup.

    Args:
        logger: Active CLI logger.
        config_path: Configuration file path in use.
    """
    import platform

    logger.info("Python version: %s", sys.version)
    logger.info("Platform: %s", platform.platform())
    logger.info("Using configuration file: %s", config_path)


def echo_fatal(exc: BaseException) -> None:
    """Echo a fatal error with traceback (config/bootstrap failures)."""
    echo_error(f"Fatal error: {exc}")
    echo_error(traceback.format_exc())
