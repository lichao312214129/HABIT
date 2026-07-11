"""Run-level provenance contracts shared by public HABIT workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Mapping, Optional, cast
from uuid import uuid4

import numpy as np
import pandas as pd

from habit._version import __version__
from habit.api.exceptions import HABITAPIError

__all__ = [
    "RunManifest",
    "create_run_manifest",
    "write_run_manifest",
]

_MANIFEST_SCHEMA_VERSION = 1


def _json_value(value: Any) -> Any:
    """Convert supported configuration values into deterministic JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(cast(Any, value)))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_value(model_dump(mode="json"))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_value(to_dict())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _config_payload(config: Any) -> Mapping[str, Any]:
    """Create a serializable, resolved configuration snapshot."""
    normalized = _json_value(config)
    if not isinstance(normalized, Mapping):
        raise HABITAPIError(
            "Workflow provenance requires a configuration mapping or model; "
            f"received {type(config).__name__}."
        )
    return normalized


def _config_hash(payload: Mapping[str, Any]) -> str:
    """Hash resolved configuration data using canonical JSON serialization."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _installed_version(package_name: str) -> Optional[str]:
    """Read an installed distribution version without making it a hard dependency."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(package_name)
    except PackageNotFoundError:
        return None


def _git_commit() -> Optional[str]:
    """Return the checked-out commit when HABIT is running from a Git checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


@dataclass(frozen=True)
class RunManifest:
    """Immutable, serializable provenance for one public HABIT workflow run."""

    workflow: str
    run_id: str
    created_at: str
    habit_version: str
    schema_version: int
    config_hash: str
    resolved_config: Mapping[str, Any]
    dependency_versions: Mapping[str, Optional[str]]
    runtime: Mapping[str, str]
    git_commit: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze nested public mappings so callers cannot mutate run history."""
        object.__setattr__(
            self, "resolved_config", MappingProxyType(dict(self.resolved_config))
        )
        object.__setattr__(
            self,
            "dependency_versions",
            MappingProxyType(dict(self.dependency_versions)),
        )
        object.__setattr__(self, "runtime", MappingProxyType(dict(self.runtime)))
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(self.metadata or {}))
        )

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON-serializable manifest payload."""
        return {
            "workflow": self.workflow,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "habit_version": self.habit_version,
            "schema_version": self.schema_version,
            "config_hash": self.config_hash,
            "resolved_config": dict(self.resolved_config),
            "dependency_versions": dict(self.dependency_versions),
            "runtime": dict(self.runtime),
            "git_commit": self.git_commit,
            "metadata": dict(self.metadata),
        }


def create_run_manifest(
    workflow: str,
    config: Any,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> RunManifest:
    """Build run provenance before or after a public workflow is executed."""
    if not workflow.strip():
        raise HABITAPIError("workflow must be a non-empty string.")
    resolved_config = _config_payload(config)
    dependencies = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "SimpleITK": _installed_version("SimpleITK"),
        "PyRadiomics": _installed_version("pyradiomics"),
        "scikit-learn": _installed_version("scikit-learn"),
        "torch": _installed_version("torch"),
    }
    return RunManifest(
        workflow=workflow,
        run_id=str(uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        habit_version=__version__,
        schema_version=_MANIFEST_SCHEMA_VERSION,
        config_hash=_config_hash(resolved_config),
        resolved_config=resolved_config,
        dependency_versions=dependencies,
        runtime={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        git_commit=_git_commit(),
        metadata=_json_value(metadata or {}),
    )


def write_run_manifest(
    manifest: RunManifest,
    output_dir: Path | str,
    *,
    filename: str = "habit_run_manifest.json",
) -> Path:
    """Atomically persist a run manifest in a workflow output directory."""
    if not filename.endswith(".json"):
        raise HABITAPIError("filename must end with '.json'.")
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / filename
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
