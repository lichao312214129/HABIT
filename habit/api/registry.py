"""Public capability namespace registry.

The registry is declarative and does not import the capability packages.
"""

from habit._public_api import PUBLIC_API_SYMBOLS, PUBLIC_NAMESPACES

__all__ = ["PUBLIC_API_SYMBOLS", "PUBLIC_NAMESPACES"]
