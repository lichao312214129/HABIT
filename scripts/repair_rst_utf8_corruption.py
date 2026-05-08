#!/usr/bin/env python3
"""
Repair RST files where the last byte of a 3-byte UTF-8 sequence was replaced
with 0x3F ('?'), which breaks decoding and shows as garbled text in Sphinx.

Given correct first two UTF-8 continuation bytes, the code point is determined
up to the low 6 bits; we brute-force those 64 values and keep the one whose
UTF-8 encoding matches the first two bytes.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List, Tuple


def _candidates_for_prefix(b0: int, b1: int) -> List[int]:
    """Return code points whose UTF-8 starts with (b0, b1) (3-byte BMP only)."""
    if not (0xE0 <= b0 <= 0xEF and 0x80 <= b1 <= 0xBF):
        return []
    high4 = b0 & 0x0F
    mid6 = b1 & 0x3F
    base = (high4 << 12) | (mid6 << 6)
    out: List[int] = []
    for low6 in range(64):
        cp = base | low6
        if cp < 0x800 or cp > 0xFFFF:
            continue
        if 0xD800 <= cp <= 0xDFFF:
            continue
        enc = chr(cp).encode("utf-8")
        if len(enc) == 3 and enc[0] == b0 and enc[1] == b1:
            out.append(cp)
    return out


def _fix_three_byte_corruption(data: bytearray) -> Tuple[int, int]:
    """Replace ... XY 3F with correct UTF-8 third byte when XY start a 3-byte char."""
    i = 0
    replacements = 0
    while i + 2 < len(data):
        b0, b1, b2 = data[i], data[i + 1], data[i + 2]
        if 0xE0 <= b0 <= 0xEF and 0x80 <= b1 <= 0xBF and b2 == 0x3F:
            cands = _candidates_for_prefix(b0, b1)
            if len(cands) == 1:
                enc = chr(cands[0]).encode("utf-8")
                data[i + 2] = enc[2]
                replacements += 1
            elif len(cands) == 0:
                pass
            else:
                # Ambiguous: keep as-is (should not happen for doc corpus)
                sys.stderr.write(
                    f"ambiguous prefix {b0:02x} {b1:02x}: {len(cands)} codepoints\n"
                )
            i += 3
            continue
        i += 1
    return replacements, len(data)


def _fix_two_byte_corruption(data: bytearray) -> int:
    """Replace C0-DF 3F with correct second byte of a 2-byte UTF-8 sequence."""
    i = 0
    replacements = 0
    while i + 1 < len(data):
        b0, b1 = data[i], data[i + 1]
        if 0xC2 <= b0 <= 0xDF and b1 == 0x3F:
            high5 = b0 & 0x1F
            base = high5 << 6
            found = False
            for low6 in range(64):
                cp = base | low6
                if cp < 0x80:
                    continue
                enc = chr(cp).encode("utf-8")
                if len(enc) == 2 and enc[0] == b0 and enc[1] != 0x3F:
                    data[i + 1] = enc[1]
                    replacements += 1
                    found = True
                    break
            if not found:
                sys.stderr.write(f"no fix for 2-byte prefix {b0:02x} 3f\n")
            i += 2
            continue
        i += 1
    return replacements


def repair_file(path: pathlib.Path, dry_run: bool) -> Tuple[int, int]:
    raw = path.read_bytes()
    data = bytearray(raw)
    r3, _ = _fix_three_byte_corruption(data)
    r2 = _fix_two_byte_corruption(data)
    total = r3 + r2
    if total and not dry_run:
        path.write_bytes(data)
    return total, r2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots",
        nargs="*",
        type=pathlib.Path,
        default=[pathlib.Path("docs/source")],
        help="Directories to scan for *.rst",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    total_files = 0
    total_repls = 0
    for root in args.roots:
        root = root.resolve()
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.rst")):
            n, _ = repair_file(path, args.dry_run)
            if n:
                total_files += 1
                total_repls += n
                print(f"{path}: {n} replacements")
    print(f"done: {total_files} files, {total_repls} replacements")


if __name__ == "__main__":
    main()
