# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text.

"""YAML text normalization and compact serialization helpers."""

from __future__ import annotations

import re
from typing import Any

import yaml


def normalize_yaml_text(text: str, *, max_blank_lines: int = 0) -> str:
    """
    Repair corrupted line endings and collapse excessive blank lines.

    Some legacy config templates contain repeated ``\\r`` bytes before ``\\n``,
    which editors render as many empty lines between each content line.

    Args:
        text: Raw YAML file contents.
        max_blank_lines: Maximum consecutive blank lines to keep (default 1).

    Returns:
        str: Normalized YAML text ending with a single newline.
    """
    if not text:
        return "\n"

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")

    output: list[str] = []
    blank_run = 0
    for line in lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= max_blank_lines:
                output.append("")
            continue
        blank_run = 0
        output.append(line.rstrip())

    body = "\n".join(output).strip("\n")
    return f"{body}\n" if body else "\n"


def dump_yaml(config: dict[str, Any]) -> str:
    """
    Serialize a config dict to compact, human-readable YAML.

    Args:
        config: Mapping to persist (typically nested workflow configuration).

    Returns:
        str: YAML document without trailing blank lines.
    """
    dumped = yaml.safe_dump(
        config,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    return normalize_yaml_text(dumped, max_blank_lines=0)


def write_yaml_file(path: str, config: dict[str, Any]) -> None:
    """
    Write a config dict to disk using :func:`dump_yaml`.

    Args:
        path: Destination file path.
        config: Mapping to serialize.
    """
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dump_yaml(config), encoding="utf-8")
