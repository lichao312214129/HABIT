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
"""
Progress bar utilities backed by tqdm.

All interactive bars should be created via :class:`CustomTqdm` so habit shares
one implementation (terminal behavior, defaults, future tweaks).

Multiprocessing note:
    Only the parent process should own and ``update`` a bar. Workers should
    write structured logs (files) instead of printing competing ``\\r`` lines.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

from tqdm.auto import tqdm as _TqdmAuto


class CustomTqdm(_TqdmAuto):
    """
    Thin tqdm wrapper with habit-wide defaults.

    Call sites keep the legacy keyword-style constructor::

        CustomTqdm(total=n, desc="Label")

    Iterator wrapping (e.g. ``as_completed``) is also supported::

        for item in CustomTqdm(items, total=len(items), desc="Work"):
            ...

    Inherits tqdm's ``update``, ``set_description``, ``close``, etc.
    """

    def __init__(
        self,
        iterable: Optional[Iterable[Any]] = None,
        *,
        total: Optional[float] = None,
        desc: str = "Progress",
        mininterval: float = 0.1,
        dynamic_ncols: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            iterable: Optional iterable to wrap (same semantics as tqdm).
            total: Expected number of iterations when ``iterable`` length is unknown
                or when using manual ``update`` mode with ``iterable=None``.
            desc: Short label shown left of the bar (English recommended for CLI).
            mininterval: Minimum seconds between refreshes (reduces flicker under load).
            dynamic_ncols: If True, tqdm adjusts width to the terminal.
            **kwargs: Forwarded to tqdm (e.g. ``disable``, ``leave``, ``file``).
        """
        # GUI jobs stream logs into a Gradio Textbox; fast tqdm refresh causes
        # constant re-renders and scroll position jumps in the browser.
        if os.environ.get("HABIT_GUI_JOB") == "1":
            mininterval = max(mininterval, 1.5)
        super().__init__(
            iterable,
            total=total,
            desc=desc,
            mininterval=mininterval,
            dynamic_ncols=dynamic_ncols,
            **kwargs,
        )
