#!/usr/bin/env python3
"""
Repair invalid UTF-8 in Sphinx RST under docs/ caused by ASCII '?' (0x3F)
replacing the last byte of a multi-byte UTF-8 sequence.

Strategy
--------
1. Scan all *.rst under the given roots and count valid UTF-8 trigrams
   (three-byte lead + cont + cont) and two-byte (lead + cont), excluding any
   trigram where the third byte is 0x3F.
2. For each corrupt position ending in 0x3F, pick the most frequent
   continuation byte observed for that prefix in the corpus.
3. Apply a small set of manual text replacements where corpus frequency
   disagrees with domain meaning.

Usage::

    python scripts/repair_docs_utf8.py [--dry-run] [path ...]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter
from typing import Dict, List, Tuple

import re

_TWO_BYTE_FLAG: int = 0xFFFE
_TWO_BYTE_BAD_MARK: int = 0xFFFF


def _build_guess_from_corpus(roots: List[pathlib.Path]) -> Dict[Tuple[int, int], int]:
    """
    Build (b0, b1) -> replacement continuation byte.

    For two-byte UTF-8, bad keys are (b0, _TWO_BYTE_BAD_MARK); good keys in
    the Counter use (b0, b1_actual, _TWO_BYTE_FLAG).
    """
    good: Counter = Counter()
    bad: Counter = Counter()

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.rst"):
            data = path.read_bytes()
            for i in range(len(data) - 2):
                b0, b1, b2 = data[i], data[i + 1], data[i + 2]
                if 0xE0 <= b0 <= 0xEF and 0x80 <= b1 <= 0xBF:
                    if b2 == 0x3F:
                        bad[(b0, b1)] += 1
                    elif 0x80 <= b2 <= 0xBF:
                        good[(b0, b1, b2)] += 1

            for i in range(len(data) - 1):
                b0, b1 = data[i], data[i + 1]
                if 0xC2 <= b0 <= 0xDF:
                    if b1 == 0x3F:
                        bad[(b0, _TWO_BYTE_BAD_MARK)] += 1
                    elif 0x80 <= b1 <= 0xBF:
                        good[(b0, b1, _TWO_BYTE_FLAG)] += 1

    guess: Dict[Tuple[int, int], int] = {}
    for pref, _ in bad.items():
        if pref[1] == _TWO_BYTE_BAD_MARK:
            subs = [(k[1], v) for k, v in good.items() if len(k) == 3 and k[0] == pref[0] and k[2] == _TWO_BYTE_FLAG]
        else:
            subs = [(k[2], v) for k, v in good.items() if len(k) == 3 and k[:2] == pref]
        subs.sort(key=lambda item: -item[1])
        if subs:
            guess[pref] = subs[0][0]
    return guess


def _repair_bytes(data: bytes, guess: Dict[Tuple[int, int], int]) -> Tuple[bytes, int]:
    """Replace invalid 0x3F continuation bytes; return (new_bytes, fix_count)."""
    d = bytearray(data)
    n = 0
    i = 0
    while i + 2 < len(d):
        b0, b1, b2 = d[i], d[i + 1], d[i + 2]
        if 0xE0 <= b0 <= 0xEF and 0x80 <= b1 <= 0xBF and b2 == 0x3F:
            repl = guess.get((b0, b1))
            if repl is not None:
                d[i + 2] = repl
                n += 1
                i += 3
                continue
        i += 1

    i = 0
    while i + 1 < len(d):
        b0, b1 = d[i], d[i + 1]
        if 0xC2 <= b0 <= 0xDF and b1 == 0x3F:
            repl = guess.get((b0, _TWO_BYTE_BAD_MARK))
            if repl is not None:
                d[i + 1] = repl
                n += 1
                i += 2
                continue
        i += 1

    return bytes(d), n


_TEXT_FIXES: Tuple[Tuple[str, str], ...] = (
    ("配置参者", "配置参数"),
    ("(int, 默认: ``30``, ≤)", "(int, 默认: ``30``，须 ≥1)"),
    ("(int, 默认 ``3``, ≤):", "(int, 默认 ``3``，须 ≥1):"),
    (
        "DEBUG/INFO/—、``console_output``",
        "DEBUG/INFO/WARNING/ERROR/CRITICAL、``console_output``",
    ),
    # Broken Python string literals (apply before global 分果 -> 分析 substitution).
    ('logger.info("开始生境分果)', 'logger.info("开始生境分析")'),
    ('logger.info("生境分析完成，)', 'logger.info("生境分析完成。")'),
    # Corpus picked wrong continuation for (0xE6, 0x9E): 果 vs 析 in 分析.
    ("分果", "分析"),
    # Sentence missing 详 in 详见 ;doc role.
    ("API 解:doc:", "API 详见 :doc:"),
    # Bold / fullwidth colon markup damaged to “，*” or “*:" patterns.
    ("**配置文件类型，*", "**配置文件类型：**"),
    ("**配置文件特点，*", "**配置文件特点：**"),
    ("**预处理配置*:", "**预处理配置**:"),
    ("**可重复性*:", "**可重复性**:"),
    ("控制生境分割和特征提可", "控制生境分割和特征提取"),
    (
        "下列「通用配置参数」中的默认值适用于多处文档叙述；**图像预处理*（``PreprocessingConfig``）"
        "的专用默认值与额外字段以本节**「预处理配置参数。* 为准（例如``processes`` 默认个1、顶层含 "
        "``auto_select_first_file`` 等）。",
        "下列「通用配置参数」中的默认值适用于多处文档叙述；**图像预处理**（``PreprocessingConfig``）"
        "的专用默认值与额外字段以本节**「预处理配置参数」**为准（例如 ``processes`` 默认为 ``1``、顶层含 "
        "``auto_select_first_file`` 等）。",
    ),
    # AutoGluon aside: comma + asterisk was a damaged closing bold before an em dash lead-in.
    (
        "**可选能力（如AutoGluon，* ——先查询，",
        "**可选能力（如 AutoGluon）**——先查询，",
    ),
    ("须包合``fixed_image``", "须包含 ``fixed_image``"),
    (
        "详见 :doc:`../reference/upstream_libraries_zh`，*scikit-learn**）；",
        "详见 :doc:`../reference/upstream_libraries_zh`（**scikit-learn**）；",
    ),
    ("自定义扩展指单", "自定义扩展指南"),
    ("无限扩展\"，通过", "无限扩展，通过"),
    # Stray ``可`` before a doc role (intended ``详见``).
    ("可:doc:", "；详见 :doc:"),
    (
        "- **说明**: 用于并行处理的进程数；图像预处理中实除worker 数为 ``min(配置值 CPU核心数2)``，且至少个1",
        "- **说明**: 用于并行处理的进程数；图像预处理中实际 worker 数为 ``min(配置值, CPU 核心数 - 2)``；若结果 ≤ 0 则使用 ``1``。",
    ),
    ("默认个**1**", "默认为 ``1``"),
    ("默认个**42**", "默认为 ``42``"),
    ("预测结果文件默认个`", "预测结果文件默认为 `"),
    ("- **必需**: 否（``predict`` 模式必需)", "- **必需**: 否（``predict`` 模式必需）"),
    (
        "真正的问题（比如编``SimpleITK``、``pyradiomics``）",
        "真正的问题（例如缺少 ``SimpleITK`` 或 ``pyradiomics`` 等依赖）",
    ),
    (
        "``type_of_transform``：**全部可选值，详见 :doc:",
        "``type_of_transform``：*全部可选值*，详见 :doc:",
    ),
    ("其使ANTs", "其它 ANTs"),
)


def _apply_text_fixes(text: str) -> str:
    for old, new in _TEXT_FIXES:
        text = text.replace(old, new)
    # ``*解:doc:`` (italic marker + stray 解) and bare ``解:doc:`` after Chinese prose.
    text = re.sub(r"\*解:doc:", "，详见 :doc:", text)
    text = re.sub(r"(?<!详)解:doc:", "详见 :doc:", text)
    # Headings / labels written as **foo，* instead of **foo：** (RST bold + fullwidth colon).
    # Avoid ``AutoGluon，* ——``-style continuations where the comma is part of the phrase.
    text = re.sub(r"\*\*([^*]+?)，\*(?! ——)", r"**\1：**", text)
    # ``**标签*:`` -> ``**标签**:`` (missing second ``*`` before colon).
    text = re.sub(r"\*\*([^*]+)\*:", r"**\1**:", text)
    # 否 was corrupted to visually similar 合 (last UTF-8 byte differs).
    text = text.replace("**必需**: 合(", "**必需**: 否（").replace("**必需**: 合", "**必需**: 否")
    text = text.replace("会指合", "会指向")
    text = text.replace("**用选**:", "**用途**:")
    return text


def repair_rst_file(
    path: pathlib.Path, guess: Dict[Tuple[int, int], int], dry_run: bool
) -> Tuple[int, bool]:
    """
    Repair one RST file.

    Returns
    -------
    byte_fixes:
        Number of 0x3F byte replacements.
    changed:
        True if file content (bytes) differs from disk after full pipeline.
    """
    raw = path.read_bytes()
    repaired, byte_fixes = _repair_bytes(raw, guess)
    try:
        text = repaired.decode("utf-8")
    except UnicodeDecodeError as exc:
        sys.stderr.write(f"{path}: still invalid UTF-8 after byte repair: {exc}\n")
        return byte_fixes, False

    text2 = _apply_text_fixes(text)
    out = text2.encode("utf-8")
    changed = out != raw
    if changed and not dry_run:
        path.write_bytes(out)
    return byte_fixes, changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair UTF-8 garbling in docs RST.")
    parser.add_argument(
        "roots",
        nargs="*",
        type=pathlib.Path,
        default=[pathlib.Path("docs")],
        help="Directories to scan (default: docs/)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    roots = [r.resolve() for r in args.roots]
    guess = _build_guess_from_corpus(roots)

    changed_files = 0
    total_byte_fixes = 0
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.rst")):
            bf, chg = repair_rst_file(path, guess, args.dry_run)
            if bf:
                total_byte_fixes += bf
            if chg:
                changed_files += 1
                extra = f" ({bf} byte-level)" if bf else ""
                print(f"{path}{extra}")

    mode = "dry-run: would update" if args.dry_run else "updated"
    print(f"{mode} {changed_files} file(s); {total_byte_fixes} byte-level fixes applied across corpus")


if __name__ == "__main__":
    main()
