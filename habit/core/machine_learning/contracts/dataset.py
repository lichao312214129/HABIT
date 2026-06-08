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
Dataset snapshot data contract for machine-learning workflows.

The :class:`DatasetSnapshot` decouples the *training/test feature matrices*
from the *result* objects.  Earlier iterations of :class:`RunResult` embedded
``x_train``/``x_test``/``y_train``/``y_test`` directly, which mixed the
"output of a run" with the "data used by the run".  Pulling the dataset out
into its own module gives:

1. A clear seam between **dataset state** and **run result**, so the result
   object stays small and serialisable.
2. A reusable container that the inference path (single dataframe, no split)
   and the K-Fold path (no fixed train/test split) can share.
3. A natural place to attach subject identifiers without coupling them to a
   particular workflow shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class DatasetSnapshot:
    """
    Frozen view of the data fed into a single workflow run.

    Attributes
    ----------
    x_train:
        Training feature matrix.  ``None`` when the workflow does not produce a
        canonical train split (e.g. inference-only runs).
    x_test:
        Test/holdout feature matrix.  ``None`` for inference-only or full
        cross-validation runs without a fixed split.
    y_train:
        Training labels.  Aligned by index with ``x_train``.  ``None`` when
        ``x_train`` is ``None``.
    y_test:
        Test/holdout labels.  Aligned by index with ``x_test``.  ``None`` when
        ``x_test`` is ``None``.
    label_col:
        Name of the label column from the originating ``DataManager``.
    subject_id_col:
        Optional name of the subject-identifier column.  When provided the
        report writers can persist a ``subject_id`` field alongside
        predictions.
    """

    label_col: str
    x_train: Optional[pd.DataFrame] = None
    x_test: Optional[pd.DataFrame] = None
    y_train: Optional[pd.Series] = None
    y_test: Optional[pd.Series] = None
    subject_id_col: Optional[str] = None

    @property
    def train_subject_ids(self) -> Tuple[object, ...]:
        """Return training subject identifiers (DataFrame index when set)."""
        if self.x_train is None:
            return ()
        return tuple(self.x_train.index.tolist())

    @property
    def test_subject_ids(self) -> Tuple[object, ...]:
        """Return test subject identifiers (DataFrame index when set)."""
        if self.x_test is None:
            return ()
        return tuple(self.x_test.index.tolist())
