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
"""Lowest-level HABIT API demo: AutoGluon on an in-memory FeatureTable.

No YAML, no CLI, no recipes, no output directory: a pandas DataFrame goes in,
predictions come out. This is the layer a third-party notebook would embed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from habit.contracts.outcome import BinaryOutcome
from habit.contracts.table import FeatureTable
from habit.domain.classification import AutogluonTabularClassifier


def make_demo_frame(n_subjects: int = 120, seed: int = 7) -> pd.DataFrame:
    """Build a synthetic radiomics-like frame with a learnable signal.

    Args:
        n_subjects: Number of rows (one row per subject).
        seed: RNG seed for reproducibility.

    Returns:
        pd.DataFrame: Columns ``subject``, ``label`` and five features; the
        label depends on ``habitat_1_volume`` and ``entropy_mean`` so a model
        can beat chance.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    x1: np.ndarray = rng.normal(0.0, 1.0, n_subjects)
    x2: np.ndarray = rng.normal(0.0, 1.0, n_subjects)
    logit: np.ndarray = 1.5 * x1 - 1.0 * x2
    label: np.ndarray = (logit + rng.normal(0.0, 0.5, n_subjects) > 0).astype(int)
    return pd.DataFrame(
        {
            "subject": [f"subj{i:03d}" for i in range(n_subjects)],
            "label": label,
            "habitat_1_volume": x1,
            "entropy_mean": x2,
            "glcm_contrast": rng.normal(10.0, 2.0, n_subjects),
            "shape_sphericity": rng.uniform(0.5, 1.0, n_subjects),
            "msi_score": rng.normal(0.0, 1.0, n_subjects),
        }
    )


def to_table(frame: pd.DataFrame, with_outcome: bool) -> FeatureTable:
    """Wrap a raw DataFrame as a FeatureTable with explicit column roles.

    Args:
        frame: Raw table carrying id, feature and (optionally) label columns.
        with_outcome: Declare the binary endpoint; pass False for unlabeled
            prediction data.

    Returns:
        FeatureTable: Validated table; id/label columns can never leak into
        the model matrix.
    """
    feature_columns = tuple(
        c for c in frame.columns if c not in {"subject", "label"}
    )
    return FeatureTable(
        frame=frame,
        id_columns=("subject",),
        feature_columns=feature_columns,
        outcome=BinaryOutcome(column="label", positive_label=1)
        if with_outcome
        else None,
    )


def main() -> None:
    """Train AutoGluon through the domain API and score a held-out table."""
    df: pd.DataFrame = make_demo_frame()
    train_table: FeatureTable = to_table(df.iloc[:90].reset_index(drop=True), True)
    test_table: FeatureTable = to_table(df.iloc[90:].reset_index(drop=True), True)

    # AutoGluon's two-part API maps to two plain dicts: the constructor
    # (task definition) and fit (training control). Every AutoGluon argument
    # is reachable this way; HABIT only adds feature_importance/label on top.
    clf = AutogluonTabularClassifier(
        predictor={"eval_metric": "roc_auc"},
        fit={"presets": "medium_quality", "time_limit": 120, "verbosity": 1},
    )
    # AutoGluon fit() accepts no random_state (v1.3+); seed the global RNGs.
    clf.set_random_state(42)

    clf.fit(train_table)

    labels: pd.Series = clf.predict(test_table)
    proba: pd.DataFrame = clf.predict_proba(test_table)

    y_true: pd.Series = test_table.frame["label"].reset_index(drop=True)
    acc: float = float((labels.to_numpy() == y_true.to_numpy()).mean())
    print(f"holdout accuracy: {acc:.3f}")
    print(proba.head())


if __name__ == "__main__":
    main()
