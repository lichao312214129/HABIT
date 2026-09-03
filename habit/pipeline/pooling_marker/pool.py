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
"""Built-in ``pool`` marker: fan-in watershed between subject and cohort."""

from __future__ import annotations

from typing import Dict

from habit.pipeline.pooling_marker.registry import PoolingRegistry
from habit.spec.specs import POOL_COMPONENT_NAME, Spec

__all__ = ["PoolMarker"]


@PoolingRegistry.register(POOL_COMPONENT_NAME)
class PoolMarker:
    """
    Marker component that declares the subject→cohort fan-in watershed.

    The executor recognises this marker and calls
    :func:`~habit.pipeline.pooling.fan_in`; the class itself performs no
    numeric work so third-party pool markers can reserve the same slot.
    """

    def __init__(self) -> None:
        """Construct the built-in pool marker (no configurable parameters)."""
        self._spec = Spec(name=POOL_COMPONENT_NAME, params={})

    @property
    def spec(self) -> Spec:
        """Return the component specification."""
        return self._spec

    def __call__(self) -> Dict[str, str]:
        """
        Return a tiny descriptor confirming the watershed.

        Returns:
            Mapping with the marker name (useful for plugin smoke tests).
        """
        return {"marker": POOL_COMPONENT_NAME}
