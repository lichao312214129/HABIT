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
"""Habitat pipeline vocabulary: voxel -> supervoxel -> habitat -> model.

These types are the nouns of habitat imaging research. ``HabitatModel`` is
HABIT's primary scientific artefact: a population-level habitat definition
that can circulate the way a pretrained segmentation model does today.

Note on equality: dataclasses holding NumPy arrays / pandas frames are
declared with ``eq=False`` because element-wise array comparison is
ambiguous; identity semantics are the safe default for value objects that
travel through pipelines.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from habit._version import __version__ as _habit_version
from habit.exceptions import CompatibilityError, HABITAPIError
from habit.contracts.geometry import Geometry
from habit.contracts.provenance import Provenance
from habit.contracts.subject import CohortFingerprint

__all__ = [
    "VoxelFeatureField",
    "Supervoxelization",
    "HabitatMap",
    "HabitatModel",
]

#: On-disk format identifier and the version this HABIT build can read/write.
#: Bump ``_FORMAT_VERSION`` (and extend the loader) whenever the layout
#: changes; older files must either load or fail with a clear message.
_FORMAT_NAME = "habit.habitatmodel"
_FORMAT_VERSION = 1


@dataclass(frozen=True, eq=False)
class VoxelFeatureField:
    """
    Per-voxel feature vectors inside one subject's ROI.

    This is where every habitat analysis begins. In v0.1 it existed only as
    an anonymous ``DataFrame`` passed between pipeline steps, which made it
    impossible for an external tool to supply its own voxel features (for
    example embeddings from a self-supervised model).

    Attributes:
        subject_id: Owning subject.
        feature_names: Column names in ``values`` order.
        values: Array of shape ``(n_voxels, n_features)``.
        voxel_index: Array of shape ``(n_voxels, 3)`` giving the ``(z, y, x)``
            grid position of each row, so the field can be rendered back into
            image space.
        geometry: Grid the indices refer to.
        provenance: How this field was produced.
    """

    subject_id: str
    feature_names: Tuple[str, ...]
    values: np.ndarray
    voxel_index: np.ndarray
    geometry: Geometry
    provenance: Provenance

    def __post_init__(self) -> None:
        """Enforce the row/column invariants that make the field renderable."""
        values = np.asarray(self.values)
        index = np.asarray(self.voxel_index)
        if values.ndim != 2:
            raise HABITAPIError(
                f"VoxelFeatureField.values must be 2D; received {values.ndim}D."
            )
        if index.ndim != 2 or index.shape[1] != 3:
            raise HABITAPIError(
                "VoxelFeatureField.voxel_index must have shape (n_voxels, 3); "
                f"received {index.shape}."
            )
        if values.shape[0] != index.shape[0]:
            raise HABITAPIError(
                "VoxelFeatureField row mismatch: values has "
                f"{values.shape[0]} rows but voxel_index has {index.shape[0]}."
            )
        if values.shape[1] != len(self.feature_names):
            raise HABITAPIError(
                "VoxelFeatureField column mismatch: values has "
                f"{values.shape[1]} columns but {len(self.feature_names)} "
                "feature names were given."
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "voxel_index", index)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))

    def to_frame(self) -> pd.DataFrame:
        """Return the field as a DataFrame for inspection and interoperability."""
        frame = pd.DataFrame(self.values, columns=list(self.feature_names))
        frame.insert(0, "x", self.voxel_index[:, 2])
        frame.insert(0, "y", self.voxel_index[:, 1])
        frame.insert(0, "z", self.voxel_index[:, 0])
        return frame

    def feature_frame(self) -> pd.DataFrame:
        """
        Return the bare unit-by-feature matrix.

        The uniform algorithm view shared with
        :meth:`Supervoxelization.feature_frame`. Any operation defined on "a
        matrix whose rows are clustering units" can therefore be written once
        and applied at either granularity, even though the two contracts
        store their matrices differently -- an array plus column names here,
        because a subject holds hundreds of thousands of voxels whose row
        identity is a 3D coordinate; an indexed frame there, because
        supervoxels are few and identified by a single id.

        Unlike :meth:`to_frame`, no coordinate columns are added: the result
        contains features and nothing else, so column-wise computations need
        no exclusion list.

        Returns:
            Feature matrix with a positional index, in ``feature_names``
            order.
        """
        return pd.DataFrame(self.values, columns=list(self.feature_names))

    def with_feature_frame(
        self,
        frame: pd.DataFrame,
        *,
        produced_by: str,
        spec_fingerprint: str,
    ) -> "VoxelFeatureField":
        """
        Return a copy carrying a recomputed feature matrix.

        Args:
            frame: Replacement matrix, row-aligned with this field. Columns
                may be fewer than the current ones (a filtering step) but the
                row count must match, since ``voxel_index`` continues to
                describe those rows.
            produced_by: Provenance label of the step that produced ``frame``.
            spec_fingerprint: Fingerprint of that step's specification.

        Returns:
            A new field sharing this field's geometry and voxel index.

        Raises:
            HABITAPIError: If ``frame`` has a different number of rows.
        """
        if len(frame) != self.values.shape[0]:
            raise HABITAPIError(
                f"{produced_by} returned {len(frame)} rows for a "
                f"{self.values.shape[0]}-voxel field. Replacing a feature "
                "matrix must preserve voxels: dropping rows would "
                "desynchronise the matrix from voxel_index."
            )
        # Preserve the frame's floating dtype. Forcing float64 here undid
        # float32 radiomics tables (v0.1 default) after subject-level
        # preprocessing and shifted cohort z-score means/stds enough to
        # break rtol=1e-6 parity on the matrix entering k-means.
        values = frame.to_numpy(copy=True)
        if not np.issubdtype(values.dtype, np.floating):
            values = np.asarray(values, dtype=np.float64)
        return VoxelFeatureField(
            subject_id=self.subject_id,
            feature_names=tuple(str(column) for column in frame.columns),
            values=values,
            voxel_index=self.voxel_index,
            geometry=self.geometry,
            provenance=self.provenance.derive(
                produced_by=produced_by,
                spec_fingerprint=spec_fingerprint,
            ),
        )


@dataclass(frozen=True, eq=False)
class Supervoxelization:
    """
    Within-subject partition of the ROI into supervoxels, plus their features.

    Scientific role: supervoxels denoise voxel-level features and reduce the
    clustering unit from a single voxel to a coherent local region, which is
    the first step of the ``two_step`` strategy.

    Attributes:
        subject_id: Owning subject.
        label_array: Supervoxel id per voxel, shape equal to the ROI grid;
            ``0`` denotes voxels outside the ROI.
        features: Index is supervoxel id, columns are aggregated features.
            This is the payload that a federated deployment would transmit
            instead of the images themselves.
        geometry: Grid ``label_array`` refers to.
        provenance: How this partition was produced.
    """

    subject_id: str
    label_array: np.ndarray
    features: pd.DataFrame
    geometry: Geometry
    provenance: Provenance

    def __post_init__(self) -> None:
        """Coerce the label array and record its dtype for downstream reuse."""
        object.__setattr__(self, "label_array", np.asarray(self.label_array))

    def feature_frame(self) -> pd.DataFrame:
        """
        Return the bare unit-by-feature matrix.

        The counterpart of :meth:`VoxelFeatureField.feature_frame`, so one
        implementation of a matrix-level operation serves both granularities.
        Here the frame is already the native representation; the supervoxel
        index is dropped to a positional one so callers cannot accidentally
        depend on label values during a column-wise computation.

        Returns:
            Feature matrix with a positional index, in column order.
        """
        return self.features.reset_index(drop=True)

    def with_feature_frame(
        self,
        frame: pd.DataFrame,
        *,
        produced_by: str,
        spec_fingerprint: str,
    ) -> "Supervoxelization":
        """
        Return a copy carrying a recomputed feature matrix.

        Args:
            frame: Replacement matrix, row-aligned with the current features.
                Columns may be fewer; the row count must match, since each
                row still describes one label of ``label_array``.
            produced_by: Provenance label of the step that produced ``frame``.
            spec_fingerprint: Fingerprint of that step's specification.

        Returns:
            A new partition with the same regions described differently:
            ``label_array`` is inherited unchanged, because describing
            supervoxels never redraws them.

        Raises:
            HABITAPIError: If ``frame`` has a different number of rows.
        """
        if len(frame) != len(self.features):
            raise HABITAPIError(
                f"{produced_by} returned {len(frame)} rows for a "
                f"{len(self.features)}-supervoxel partition. Replacing a "
                "feature matrix must preserve supervoxels: dropping rows "
                "would desynchronise the matrix from label_array."
            )
        restored = frame.copy()
        restored.index = self.features.index
        return Supervoxelization(
            subject_id=self.subject_id,
            label_array=self.label_array,
            features=restored,
            geometry=self.geometry,
            provenance=self.provenance.derive(
                produced_by=produced_by,
                spec_fingerprint=spec_fingerprint,
            ),
        )


@dataclass(frozen=True, eq=False)
class HabitatMap:
    """
    Habitat label image for one subject.

    Attributes:
        subject_id: Owning subject.
        label_array: Habitat id per voxel; ``0`` denotes background.
        geometry: Grid ``label_array`` refers to.
        model_id: Identifier of the :class:`HabitatModel` that assigned these
            labels. Without it, habitat ids from different runs are not
            comparable, which is the most common silent error in habitat
            studies.
        habitat_ids: Habitat ids the model can assign, in canonical order.
            Note that a given subject need not contain all of them.
        provenance: How this map was produced.
    """

    subject_id: str
    label_array: np.ndarray
    geometry: Geometry
    model_id: str
    habitat_ids: Tuple[int, ...]
    provenance: Provenance

    def __post_init__(self) -> None:
        """Coerce the label array and canonicalise the habitat id tuple."""
        object.__setattr__(self, "label_array", np.asarray(self.label_array))
        object.__setattr__(self, "habitat_ids", tuple(int(v) for v in self.habitat_ids))


def _to_jsonable(value: Any) -> Any:
    """
    Convert common scientific-Python values into JSON-serialisable form.

    NumPy arrays become explicit ``{"__ndarray__": ...}`` payloads carrying
    dtype and shape so they round-trip faithfully; scalars become native
    Python numbers; dataclass-like records with ``__dict__`` are NOT touched
    here (the dedicated provenance/fingerprint serialisers handle those).

    Args:
        value: Arbitrary value from a spec payload or preprocessing state.

    Returns:
        A JSON-serialisable equivalent.
    """
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _from_jsonable(value: Any) -> Any:
    """
    Restore values converted by :func:`_to_jsonable`.

    Args:
        value: JSON-decoded value.

    Returns:
        The restored value, with ``__ndarray__`` payloads rebuilt as arrays.
    """
    if isinstance(value, Mapping):
        if value.get("__ndarray__"):
            array = np.array(value["data"], dtype=np.dtype(value["dtype"]))
            return array.reshape(tuple(value["shape"]))
        return {key: _from_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_jsonable(item) for item in value]
    return value


def _provenance_to_dict(provenance: Provenance) -> Dict[str, Any]:
    """Serialise a provenance DAG into a nested JSON-able mapping."""
    return {
        "produced_by": provenance.produced_by,
        "spec_fingerprint": provenance.spec_fingerprint,
        "inputs": [_provenance_to_dict(item) for item in provenance.inputs],
        "software": dict(provenance.software),
        "random_seed": provenance.random_seed,
        "created_at": provenance.created_at,
        "notes": _to_jsonable(dict(provenance.notes)),
    }


def _provenance_from_dict(payload: Mapping[str, Any]) -> Provenance:
    """Rebuild a provenance DAG from :func:`_provenance_to_dict` output."""
    return Provenance(
        produced_by=str(payload["produced_by"]),
        spec_fingerprint=str(payload["spec_fingerprint"]),
        inputs=tuple(
            _provenance_from_dict(item) for item in payload.get("inputs", ())
        ),
        software=dict(payload.get("software", {})),
        random_seed=payload.get("random_seed"),
        created_at=payload.get("created_at"),
        notes=dict(payload.get("notes", {})),
    )


@dataclass(frozen=True, eq=False)
class HabitatModel:
    """
    Population-level habitat definition -- HABIT's primary scientific artefact.

    In v0.1 this was serialised as an opaque ``habitat_pipeline.pkl``
    byproduct. Promoting it to a first-class, self-describing object is what
    enables the strategic goal: a habitat definition published alongside a
    paper can be loaded by other groups and applied to their own cohorts.

    Attributes:
        model_id: Stable identifier derived from the specification
            fingerprint.
        n_habitats: Number of habitats this model can assign.
        feature_names: Features consumed for assignment, in required order.
        centroids: Population cluster centres, shape
            ``(n_habitats, n_features)``.
        preprocessing_state: State learned at fit time and required at apply
            time, e.g. binning edges and normalisation statistics. Keeping
            this inside the model is what guarantees train/predict
            consistency.
        spec_payload: Serialisable form of the full algorithm specification,
            so the model can describe itself and be exported back to YAML.
        cohort_fingerprint: Non-identifiable description of the defining
            cohort.
        provenance: Software, dependency, and seed fingerprint.

    Examples:
        Models are produced by the habitat recipes and round-trip through a
        self-describing ``.habitatmodel`` archive:

        >>> from habit import HabitatModel
        >>> model = HabitatModel.load("out/habitat_model.habitatmodel")  # doctest: +SKIP
        >>> model.n_habitats, model.feature_names  # doctest: +SKIP
        (3, ('T1', 'T2'))
        >>> print(model.summary())  # doctest: +SKIP
        >>> assigner = model.assigner()  # doctest: +SKIP

        See :meth:`habit.recipes.Study.predict` (via ``Study.from_model``)
        for projecting a reloaded model onto new subjects.
    """

    model_id: str
    n_habitats: int
    feature_names: Tuple[str, ...]
    centroids: np.ndarray
    preprocessing_state: Mapping[str, Any]
    spec_payload: Mapping[str, Any]
    cohort_fingerprint: CohortFingerprint
    provenance: Provenance

    def __post_init__(self) -> None:
        """Validate the centroid matrix against the declared dimensions."""
        centroids = np.asarray(self.centroids)
        if centroids.ndim != 2:
            raise HABITAPIError(
                f"HabitatModel.centroids must be 2D; received {centroids.ndim}D."
            )
        if centroids.shape[0] != self.n_habitats:
            raise HABITAPIError(
                f"HabitatModel declares {self.n_habitats} habitats but "
                f"centroids has {centroids.shape[0]} rows."
            )
        if centroids.shape[1] != len(self.feature_names):
            raise HABITAPIError(
                f"HabitatModel declares {len(self.feature_names)} features but "
                f"centroids has {centroids.shape[1]} columns."
            )
        object.__setattr__(self, "centroids", centroids)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        object.__setattr__(
            self, "preprocessing_state", dict(self.preprocessing_state)
        )
        object.__setattr__(self, "spec_payload", dict(self.spec_payload))

    def summary(self) -> str:
        """
        Return a human-readable model card.

        Named ``summary`` (statsmodels convention) rather than ``describe``,
        because in scientific Python ``DataFrame.describe()`` already returns
        a statistics table, and this returns prose. Intended for both
        notebook inspection and inclusion in a manuscript's supplementary
        material.

        Returns:
            Multi-line English description of the model.
        """
        lines = [
            f"HabitatModel {self.model_id}",
            f"  habitats           : {self.n_habitats}",
            f"  features ({len(self.feature_names)})    : {', '.join(self.feature_names)}",
            f"  defining cohort    : n={self.cohort_fingerprint.n_subjects}"
            + (
                f", name={self.cohort_fingerprint.name}"
                if self.cohort_fingerprint.name
                else ""
            ),
            f"  modalities         : {', '.join(self.cohort_fingerprint.modalities) or 'n/a'}",
            f"  cohort digest      : {self.cohort_fingerprint.subject_id_digest[:16]}...",
            f"  produced by        : {self.provenance.produced_by}",
            f"  habit version      : {self.provenance.software.get('habit', 'unknown')}",
            f"  random seed        : {self.provenance.random_seed}",
        ]
        preprocessing_keys = sorted(self.preprocessing_state)
        if preprocessing_keys:
            lines.append(
                f"  preprocessing state: {', '.join(preprocessing_keys)}"
            )
        return "\n".join(lines)

    def with_cohort_preprocessing(
        self,
        state: Mapping[str, Any],
        spec_payload: Mapping[str, Any],
    ) -> "HabitatModel":
        """
        Bind the cohort-level feature preprocessing into this model.

        A habitat definition is a set of centroids TOGETHER WITH the feature
        space they live in. Storing the fitted cohort chain here is what lets
        the model be applied to a new cohort at all: without it, prediction
        would compute raw features, compare them against centroids fitted on
        preprocessed features, and return labels that look entirely
        reasonable.

        The model id is recomputed, because two models whose centroids came
        from differently preprocessed features are different definitions and
        must not collide. Provenance is derived rather than replaced, so the
        chain back to each fitting unit stays intact.

        Args:
            state: Fitted chain state, from
                ``CohortPreprocessingChain.state``.
            spec_payload: The chain's specification, recorded alongside the
                fitter's so the model card states both.

        Returns:
            A new model carrying the chain. Callers that need the original
            still hold it -- this contract is frozen.
        """
        merged_state = {
            **dict(self.preprocessing_state),
            "cohort_feature_preprocessor": dict(state),
        }
        merged_spec = {
            **dict(self.spec_payload),
            "cohort_feature_preprocessor": dict(spec_payload),
        }
        fitter_name = self.model_id.split("-", 1)[0]
        chain_fingerprint = hashlib.sha256(
            json.dumps(_to_jsonable(dict(spec_payload)), sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()
        rebound_id = hashlib.sha256(
            f"{self.model_id}:{chain_fingerprint}".encode("utf-8")
        ).hexdigest()[:16]
        return replace(
            self,
            model_id=f"{fitter_name}-{rebound_id}",
            preprocessing_state=merged_state,
            spec_payload=merged_spec,
            # Cohort preprocessing is deterministic given the fitted chain
            # state; it must not wipe the fitter seed from the model card.
            # derive() inherits the parent seed when omitted -- pass it
            # explicitly so the contract of this method is local and obvious.
            provenance=self.provenance.derive(
                produced_by=f"{self.provenance.produced_by}+cohort_preprocessing",
                spec_fingerprint=chain_fingerprint,
                random_seed=self.provenance.random_seed,
            ),
        )

    def assigner(self, name: str = "nearest_centroid", **params: Any) -> Any:
        """
        Build an assigner that projects this model onto individual subjects.

        Assigners take their model at construction time, so this factory is
        the ordinary way to obtain one and keeps the common case to a single
        call: ``labels = model.assigner()(supervoxel_map)``. The registry
        import is lazy: the contracts layer must stay importable without the
        domain layer.

        Args:
            name: Registered ``habitat_assigner`` implementation name.
            **params: Parameters for that implementation.

        Returns:
            A one-argument callable from a supervoxel map to a habitat map.
        """
        from habit.domain.assignment import HabitatAssignerRegistry

        return HabitatAssignerRegistry.create(name, model=self, **params)

    def save(self, path: Union[str, Path]) -> Path:
        """
        Persist the model in a versioned, self-describing format.

        Deliberately not a bare pickle: a shared scientific artefact must
        remain readable across HABIT versions, or fail with an explicit
        incompatibility message rather than a deserialisation error. The
        ``.habitatmodel`` file is a ZIP archive holding a JSON manifest
        (format name, format version, producing HABIT version, and every
        scalar field) plus the centroid matrix as a ``.npy`` member.

        Args:
            path: Destination file path.

        Returns:
            The written path.
        """
        import io

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format": _FORMAT_NAME,
            "format_version": _FORMAT_VERSION,
            "habit_version": _habit_version,
            "model_id": self.model_id,
            "n_habitats": self.n_habitats,
            "feature_names": list(self.feature_names),
            "preprocessing_state": _to_jsonable(dict(self.preprocessing_state)),
            "spec_payload": _to_jsonable(dict(self.spec_payload)),
            "cohort_fingerprint": asdict(self.cohort_fingerprint),
            "provenance": _provenance_to_dict(self.provenance),
        }
        buffer = io.BytesIO()
        np.save(buffer, self.centroids, allow_pickle=False)
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "manifest.json",
                json.dumps(manifest, indent=2, sort_keys=True),
            )
            zf.writestr("arrays/centroids.npy", buffer.getvalue())
        return destination

    @classmethod
    def load(cls, path: Union[str, Path]) -> "HabitatModel":
        """
        Load a model previously written by :meth:`save`.

        Args:
            path: Source file path.

        Returns:
            The reconstructed model.

        Raises:
            CompatibilityError: If the file was produced by an incompatible
                format or HABIT version, with guidance on which version can
                read it.
        """
        import io

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"HabitatModel file not found: {source}")
        try:
            archive = zipfile.ZipFile(source, "r")
        except zipfile.BadZipFile as exc:
            raise CompatibilityError(
                f"{source} is not a {_FORMAT_NAME} file. HABIT v1.0 expects "
                "a self-describing .habitatmodel archive produced by train; "
                "legacy v0.1 habitat_pipeline.pkl files are not supported. "
                "Re-train and apply the model via Study.from_model(...).predict(...) "
                "or point pipeline_path at habitat_model.habitatmodel."
            ) from exc
        with archive:
            try:
                manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            except KeyError as exc:
                raise CompatibilityError(
                    f"{source} lacks a manifest.json; it is not a valid "
                    f"{_FORMAT_NAME} file."
                ) from exc
            if manifest.get("format") != _FORMAT_NAME:
                raise CompatibilityError(
                    f"{source} has format {manifest.get('format')!r}; expected "
                    f"{_FORMAT_NAME!r}."
                )
            file_version = int(manifest.get("format_version", 0))
            if file_version > _FORMAT_VERSION:
                raise CompatibilityError(
                    f"{source} was written with format version {file_version}, "
                    f"but this HABIT (v{_habit_version}) reads up to version "
                    f"{_FORMAT_VERSION}. Upgrade HABIT to load this model."
                )
            centroids = np.load(
                io.BytesIO(archive.read("arrays/centroids.npy")),
                allow_pickle=False,
            )
        fingerprint_payload = manifest["cohort_fingerprint"]
        return cls(
            model_id=str(manifest["model_id"]),
            n_habitats=int(manifest["n_habitats"]),
            feature_names=tuple(str(v) for v in manifest["feature_names"]),
            centroids=centroids,
            preprocessing_state=_from_jsonable(manifest["preprocessing_state"]),
            spec_payload=_from_jsonable(manifest["spec_payload"]),
            cohort_fingerprint=CohortFingerprint(
                n_subjects=int(fingerprint_payload["n_subjects"]),
                modalities=tuple(fingerprint_payload["modalities"]),
                subject_id_digest=str(fingerprint_payload["subject_id_digest"]),
                name=fingerprint_payload.get("name"),
                description=fingerprint_payload.get("description"),
            ),
            provenance=_provenance_from_dict(manifest["provenance"]),
        )
