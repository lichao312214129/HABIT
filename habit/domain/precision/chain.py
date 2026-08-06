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
"""Perturbation chains: composing one simulated retest from atomic steps.

The paper's perturbed image is not one perturbation but three applied in
sequence (noise, then translation, then rotation). The chain is an
assembler, not a registered component: its steps are the registered
components, and its specification records each step's specification in
order, so the provenance fingerprint captures the full composition.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from habit.contracts.subject import Subject
from habit.domain.protocols import ImagePerturbation
from habit.exceptions import HABITAPIError
from habit.spec.specs import Spec

__all__ = ["PerturbationChain"]


class PerturbationChain:
    """
    Apply several image perturbations in sequence to one subject.

    Args:
        steps: Perturbations applied in the given order; each receives the
            output of the previous one. At least one step is required.
    """

    def __init__(self, steps: Sequence[ImagePerturbation]) -> None:
        if not steps:
            raise HABITAPIError("PerturbationChain needs at least one step.")
        self.steps = tuple(steps)

    @property
    def spec(self) -> Spec:
        """Return the composite specification used for provenance."""
        return Spec(
            name="perturbation_chain",
            params={"steps": [step.spec.to_dict() for step in self.steps]},
        )

    def __call__(self, subject: Subject, *, rng: np.random.Generator) -> Subject:
        """
        Return a copy of ``subject`` with every step applied in order.

        Args:
            subject: Subject to perturb.
            rng: Random generator shared by all stochastic steps, so one
                seed defines the whole simulated retest.

        Returns:
            The perturbed subject copy.
        """
        for step in self.steps:
            subject = step(subject, rng=rng)
        return subject
