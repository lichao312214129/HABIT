# Copyright (c) 2024-2026 Li Chao, Dong Mengshi and HABIT Contributors.
#
# This file is part of HABIT (Habitat Analysis: Biomedical Imaging Toolkit).
# Use is governed by the HABIT Software License — see the LICENSE file in the
# project root for the full text. Summary:
#
#   - Non-commercial use (academic, research, education, personal) is permitted
#     provided that copyright notices are retained and HABIT usage is
#     acknowledged in publications, reports, or documentation.
#   - Commercial use requires prior written consent from the copyright holder
#     (lichao19870617@163.com) and public acknowledgment of HABIT usage in
#     product documentation or user-facing materials.
#   - Unauthorized commercial use or removal of attribution is prohibited.
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
        super().__init__(
            iterable,
            total=total,
            desc=desc,
            mininterval=mininterval,
            dynamic_ncols=dynamic_ncols,
            **kwargs,
        )
