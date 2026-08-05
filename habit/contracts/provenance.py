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
"""Provenance: part of the data structure, not a separate reporting feature.

``Provenance`` travels with every derived object instead of being assembled at
the end of a workflow. This is what allows a third party who used only one
HABIT component inside their own pipeline to still emit a complete methods
description for a manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from typing import Any, Mapping, Optional, Tuple, Union

from habit._version import __version__ as _habit_version

__all__ = ["Provenance", "software_fingerprint"]

#: Sentinel for :meth:`Provenance.derive`: omit ``random_seed`` to inherit the
#: parent's seed. Distinct from passing ``random_seed=None``, which clears it.
_SEED_UNSET: object = object()

#: Dependencies whose versions are scientifically relevant to habitat
#: analysis and therefore recorded in every provenance record. Looked up via
#: ``importlib.metadata`` so checking them never imports the packages.
_TRACKED_DISTRIBUTIONS: Tuple[Tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scikit-learn", "scikit-learn"),
    ("SimpleITK", "SimpleITK"),
    ("pyradiomics", "pyradiomics"),
    ("scipy", "scipy"),
)


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def software_fingerprint() -> Mapping[str, str]:
    """
    Capture HABIT and scientifically relevant dependency versions.

    Versions are resolved through ``importlib.metadata`` rather than imports
    so that building a provenance record stays cheap and side-effect free;
    distributions that are not installed are simply omitted.

    Returns:
        Mapping of distribution name to installed version string, always
        including the ``habit`` entry.
    """
    versions = {"habit": _habit_version}
    for key, distribution in _TRACKED_DISTRIBUTIONS:
        try:
            versions[key] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


@dataclass(frozen=True)
class Provenance:
    """
    Immutable record answering "how was this object produced?".

    Attributes:
        produced_by: Registered component name that created this object, e.g.
            ``"supervoxelizer.slic"``.
        spec_fingerprint: Stable hash of the algorithm specification used, so
            two runs can be compared for scientific equivalence.
        inputs: Provenance of every object consumed to produce this one. This
            forms a directed acyclic graph back to the raw images.
        software: Version fingerprint of HABIT and the scientifically relevant
            dependencies (e.g. PyRadiomics, SimpleITK, scikit-learn).
        random_seed: Seed in effect for this scientific result, or ``None``
            when no seeded step contributed. Deterministic derivations inherit
            the parent's seed via :meth:`derive` unless cleared explicitly.
        created_at: ISO-8601 UTC timestamp.
        notes: Free-form annotations that must never be required for
            reproduction; they exist for human readers only.
    """

    produced_by: str
    spec_fingerprint: str
    inputs: Tuple["Provenance", ...] = ()
    software: Mapping[str, str] = field(default_factory=dict)
    random_seed: Optional[int] = None
    created_at: Optional[str] = None
    notes: Mapping[str, Any] = field(default_factory=dict)

    def derive(
        self,
        *,
        produced_by: str,
        spec_fingerprint: str,
        random_seed: Union[int, None, object] = _SEED_UNSET,
    ) -> "Provenance":
        """
        Create the provenance of an object derived from this one.

        Operator authors never write provenance by hand; base classes call
        this so that the propagation rule stays uniform across the codebase.
        The software fingerprint is inherited from ``self`` (the environment
        does not change mid-pipeline) and the timestamp is stamped here.

        The random seed follows the same inheritance rule unless overridden:
        a deterministic step that does not mention seeding must not erase the
        seed that defined the scientific result upstream (for example,
        attaching cohort-level feature preprocessing to a fitted
        ``HabitatModel``). Pass ``random_seed=`` explicitly to record a new
        seed, or ``random_seed=None`` to clear it.

        Args:
            produced_by: Registered name of the component doing the
                derivation.
            spec_fingerprint: Fingerprint of that component's specification.
            random_seed: Seed used by the derivation. Omitted (default) keeps
                ``self.random_seed``; ``None`` clears; an ``int`` replaces.

        Returns:
            A new ``Provenance`` whose ``inputs`` contains ``self``.
        """
        resolved_seed: Optional[int]
        if random_seed is _SEED_UNSET:
            resolved_seed = self.random_seed
        else:
            resolved_seed = random_seed  # type: ignore[assignment]
        return Provenance(
            produced_by=produced_by,
            spec_fingerprint=spec_fingerprint,
            inputs=(self,),
            software=dict(self.software) if self.software else software_fingerprint(),
            random_seed=resolved_seed,
            created_at=_utc_now_iso(),
        )

    @classmethod
    def source(cls, produced_by: str) -> "Provenance":
        """
        Create the root provenance of an object that has no HABIT inputs.

        Used for objects entering the pipeline from the outside world (raw
        images, user-constructed arrays) so that every derived record still
        terminates at a well-defined root.

        Args:
            produced_by: Description of the external origin, e.g.
                ``"directory_source"`` or ``"user_array"``.

        Returns:
            A provenance record with no inputs and the current software
            fingerprint.
        """
        return cls(
            produced_by=produced_by,
            spec_fingerprint="",
            inputs=(),
            software=software_fingerprint(),
            created_at=_utc_now_iso(),
        )
