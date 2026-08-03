"""
HABIT v1.0 architecture prototype.

Design-stage interface sketches accompanying ``06_v1_api_first_architecture.md``.
These modules define signatures, contracts, and invariants so that the proposed
API can be reviewed as code rather than prose. Implementation bodies raise
``NotImplementedError``.

This package is intentionally outside the shipped ``habit`` package: it is not
installed, not imported at runtime, and cannot affect v0.1.x behaviour. Once the
design is approved, these definitions move into ``habit/`` at the layer
positions described in the design document.
"""

from __future__ import annotations

__all__ = ["contracts", "protocols", "spec", "usage_examples"]
