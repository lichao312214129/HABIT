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
"""Guarantee every label drawn on a figure is journal-safe ASCII.

HABIT's hard-coded figure text (titles, axis labels, legends) is already
English, but a large class of labels is DATA-DRIVEN: a feature name, a cohort
group, a KM stratum. When those come from a clinical table they may carry
non-ASCII characters (CJK names, accented units), which most journals and
some font stacks cannot render. Centralising the sanitisation here means the
rule "no non-ASCII on a figure" is enforced in CODE at the single point every
plot funnels through, rather than by convention across a dozen call sites.

The mapping is lossy on purpose: it is better for a figure to show a clearly
transliterated placeholder than to embed a glyph that renders as a box -- or
worse, that silently drops out -- in the publisher's pipeline.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

__all__ = ["sanitize_label"]

# Characters that survive transliteration but break matplotlib's mathtext or
# TeX-like parsing (``$``) or are simply noise on an axis. Curly quotes and
# dashes are folded onto their ASCII forms first.
_TRANSLATE = {
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "–": "-", "—": "-", "−": "-", "…": "...",
    "°": " deg ", "±": "+/-", "×": "x", "÷": "/", "μ": "u",
}
_NON_ASCII = re.compile(r"[^\x00-\x7F]+")


def sanitize_label(value: Any) -> str:
    """
    Return ``value`` as a printable ASCII string safe for a figure.

    The transformation is: fold common typographic characters onto ASCII,
    strip accents via NFKD normalisation, then replace any remaining run of
    non-ASCII characters (e.g. CJK, which has no accent to strip) with a
    single ``"?"`` so its presence is VISIBLE rather than silently lost.

    Args:
        value: The label to sanitise; typically a feature or group name.
            Non-strings are converted with ``str`` first.

    Returns:
        An ASCII string. ASCII input passes through unchanged apart from the
        typographic folding, so English labels are never altered.

    Examples:
        >>> sanitize_label("MSI_score")
        'MSI_score'
        >>> sanitize_label("T1加权")
        'T1?'
        >>> sanitize_label("CAFé")
        'CAFe'
    """
    text = str(value)
    for src, dst in _TRANSLATE.items():
        text = text.replace(src, dst)
    # NFKD splits accented characters into base + combining mark; dropping
    # the combining marks yields the ASCII base ("é" -> "e").
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    # Any non-ASCII that survived (CJK, symbols) becomes a visible "?".
    text = _NON_ASCII.sub("?", text)
    return text
