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
"""Per-supervoxel radiomics: the family that motivated this domain.

Describing every supervoxel with a full PyRadiomics feature vector -- rather
than the mean of the voxel features -- is what lets the population clustering
separate regions by texture instead of by intensity alone. It is also the
family that cannot be expressed as a ``Supervoxelizer``: it reads the
subject's ORIGINAL INTENSITIES, which a ``VoxelFeatureField`` does not carry.

The numerics are the v0.1 implementation unchanged. The batched union-mask
binning, the habit native C extension for texture matrices and the
TorchRadiomics GPU path all live in :mod:`habit.kernels.radiomics`, which the
v0.1 extractor calls too; only the plumbing (in-memory contracts instead of
file paths) is new, so previously published numbers stay reproducible.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from habit.exceptions import HABITAPIError
from habit.contracts.habitat import Supervoxelization
from habit.contracts.subject import Subject
from habit.supervoxel.features_base import (
    SUPERVOXEL_INDEX_NAME,
    partition_labels,
    resolve_modality_names,
    with_features,
)
from habit.supervoxel.features_registry import (
    SupervoxelFeatureExtractorRegistry,
)
from habit.spec.specs import Spec

__all__ = ["SupervoxelRadiomicsFeatures"]

#: Column the v0.1 extractors emit for the supervoxel identifier.
_V01_LABEL_COLUMN = "supervoxel_id"


@SupervoxelFeatureExtractorRegistry.register("supervoxel_radiomics")
class SupervoxelRadiomicsFeatures:
    """
    Describe each supervoxel by PyRadiomics features of the original images.

    One extraction pass runs per modality and the resulting columns are
    suffixed with ``-{modality}``, which is the v0.1 column scheme for
    ``concat(supervoxel_radiomics(m1), supervoxel_radiomics(m2))``. All
    modalities of a subject are used when none are named.

    The single-modality form ``modality="T1"`` is the tree-friendly
    alternative to ``modalities=["T1"]``; ``as_`` renames the column
    suffix so the same modality can be extracted under two parameter
    sets without a name clash.

    Args:
        modality: Single modality name; mutually exclusive with
            ``modalities``.
        modalities: Modality names to extract from; empty selects all the
            subject carries.
        as_: Alias used as the column suffix instead of the modality name;
            requires ``modality``.
        params_file: Path to a PyRadiomics parameter YAML, or ``None`` for
            PyRadiomics defaults.
        params: Inline PyRadiomics settings mapping, for API users holding
            settings in memory. Mutually exclusive with ``params_file``.
        supervoxel_batch: Labels processed per batch. Larger batches trade
            memory for speed and never change the numbers.
        supervoxel_union_bbox_crop: Crop image and masks to the bounding box
            of the union of all supervoxels before extraction.
        supervoxel_pad_distance: Padding around that bounding box; ``None``
            keeps the PyRadiomics ``padDistance`` setting.
        use_supervoxel_cext: ``True`` (default), ``"auto"``, or ``False`` -- whether to
            use the habit native C extension for texture matrices.
        union_bin: When False (default) each supervoxel is discretized
            with its own ``binWidth`` edges, matching PyRadiomics
            ``execute()``. When True, all labels share one union-mask bin.
        use_torch_radiomics: ``"auto"``, ``True`` or ``False`` -- whether to
            use the TorchRadiomics GPU path when torch and CUDA are present.
        torch_device: Torch device string, or ``"auto"`` to select one.
        torch_dtype: ``"float64"`` (default) or ``"float32"`` for the torch path.
        output_float32: Downcast the resulting feature columns to float32,
            the v0.1 default that keeps large supervoxel tables manageable.
    """

    def __init__(
        self,
        modality: Optional[str] = None,
        modalities: Sequence[str] = (),
        as_: Optional[str] = None,
        params_file: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        supervoxel_batch: int = 64,
        supervoxel_union_bbox_crop: bool = True,
        supervoxel_pad_distance: Optional[int] = None,
        use_supervoxel_cext: Union[str, bool] = True,
        union_bin: bool = False,
        use_torch_radiomics: Union[str, bool] = "auto",
        torch_device: str = "auto",
        torch_dtype: str = "float64",
        output_float32: bool = True,
    ) -> None:
        if modality is not None and modalities:
            raise HABITAPIError(
                "supervoxel_radiomics: 'modality' and 'modalities' are "
                "mutually exclusive; use 'modality' for the single-modality "
                "form."
            )
        if as_ is not None and modality is None:
            raise HABITAPIError(
                "supervoxel_radiomics: 'as_' requires the single-modality "
                "form; pass 'modality' as well."
            )
        if params_file is not None and params is not None:
            raise HABITAPIError(
                "supervoxel_radiomics: params_file and params are mutually "
                "exclusive; pass the PyRadiomics settings as a file path OR "
                "as a mapping."
            )
        if (
            isinstance(supervoxel_batch, bool)
            or not isinstance(supervoxel_batch, int)
            or supervoxel_batch < 1
        ):
            raise HABITAPIError(
                "supervoxel_batch must be a positive integer; "
                f"got {supervoxel_batch!r}."
            )
        self.modality = modality
        self.as_ = as_
        self.modalities = (modality,) if modality is not None else tuple(modalities)
        self.params_file = params_file
        self.params = dict(params) if params is not None else None
        self.supervoxel_batch = int(supervoxel_batch)
        self.supervoxel_union_bbox_crop = bool(supervoxel_union_bbox_crop)
        self.supervoxel_pad_distance = supervoxel_pad_distance
        self.use_supervoxel_cext = use_supervoxel_cext
        self.union_bin = bool(union_bin)
        self.use_torch_radiomics = use_torch_radiomics
        self.torch_device = str(torch_device)
        self.torch_dtype = str(torch_dtype)
        self.output_float32 = bool(output_float32)

    def _resolved_params_file(self) -> Optional[str]:
        """
        Return the parameter file to use, falling back to the supervoxel preset.

        Matches v0.1 ``SupervoxelRadiomicsExtractor`` behaviour: when the user
        omits ``params_file``, the bundled ``params_supervoxel_radiomics.yaml``
        preset is used instead of PyRadiomics bare defaults (which enable no
        feature classes and yield empty matrices).

        Returns:
            A path, or ``None`` when inline ``params`` were supplied.
        """
        if self.params is not None:
            return None
        from habit.utils.radiomics_preset_utils import resolve_params_file

        return resolve_params_file(self.params_file, preset="supervoxel")

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        params: Dict[str, Any] = {
            "modalities": list(self.modalities),
            "params_file": self.params_file,
            "params": self.params,
            "supervoxel_batch": self.supervoxel_batch,
            "supervoxel_union_bbox_crop": self.supervoxel_union_bbox_crop,
            "supervoxel_pad_distance": self.supervoxel_pad_distance,
            "use_supervoxel_cext": self.use_supervoxel_cext,
            "union_bin": self.union_bin,
            "use_torch_radiomics": self.use_torch_radiomics,
            "torch_device": self.torch_device,
            "torch_dtype": self.torch_dtype,
            "output_float32": self.output_float32,
        }
        # Fold the single-modality spelling in only when used, so existing
        # configurations keep their historical fingerprints.
        if self.modality is not None:
            params["modality"] = self.modality
        if self.as_ is not None:
            params["as_"] = self.as_
        return Spec(name="supervoxel_radiomics", params=params)

    def _radiomics_settings(self, extractor_settings: Any) -> Dict[str, object]:
        """
        Merge the habit-specific extraction keys into PyRadiomics settings.

        Args:
            extractor_settings: Settings of the initialised PyRadiomics
                extractor.

        Returns:
            The settings mapping handed to the batched helpers.
        """
        from habit.kernels.radiomics.settings import merge_supervoxel_settings

        overrides: Dict[str, object] = {
            "supervoxel_union_bbox_crop": self.supervoxel_union_bbox_crop,
            "use_supervoxel_cext": self.use_supervoxel_cext,
            "union_bin": self.union_bin,
        }
        if self.supervoxel_pad_distance is not None:
            overrides["supervoxel_pad_distance"] = int(self.supervoxel_pad_distance)
        return merge_supervoxel_settings(extractor_settings, overrides)

    def _extract_one_modality(
        self,
        subject: Subject,
        partition: Supervoxelization,
        modality: str,
        labels: np.ndarray,
        column_label: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run one PyRadiomics pass over every supervoxel of one modality.

        Args:
            subject: Subject supplying the intensity image.
            partition: Supervoxel partition defining the regions.
            modality: Modality to extract from.
            labels: Supervoxel ids to extract, ascending.
            column_label: Column suffix for the output frame; defaults to
                the modality name. The ``as_`` alias lands here.

        Returns:
            Feature frame indexed by supervoxel id.
        """
        from habit.kernels.radiomics.supervoxel_batch import (
            extract_batched_supervoxel_features,
            extract_supervoxel_features_pyradiomics,
        )
        from habit.radiomics._domain import (
            build_pyradiomics_extractor,
            sitk_image_from_contract,
        )
        from habit.utils.torch_radiomics_utils import (
            injected_torch_radiomics,
            resolve_torch_dtype,
            resolve_voxel_radiomics_backend,
        )

        volume = subject.image(modality)
        image_sitk = sitk_image_from_contract(volume.load(), volume.geometry)
        supervoxel_sitk = sitk_image_from_contract(
            np.asarray(partition.label_array), partition.geometry
        )
        # v0.1 semantics: the label map adopts the image's metadata verbatim,
        # so PyRadiomics reads every label off one multi-label mask.
        supervoxel_sitk.CopyInformation(image_sitk)

        extractor = build_pyradiomics_extractor(
            self._resolved_params_file(), self.params, owner="supervoxel_radiomics"
        )
        extractor.settings.update({"geometryTolerance": 1e-3})
        settings = self._radiomics_settings(extractor.settings)
        suffix = column_label if column_label is not None else modality

        from habit.kernels.radiomics.cext import is_cext_available

        use_cext = (
            self.use_supervoxel_cext is True
            or (self.use_supervoxel_cext == "auto" and is_cext_available())
        )
        if use_cext:
            from habit.kernels.radiomics.native_batch import (
                extract_native_supervoxel_features,
            )

            frame = extract_native_supervoxel_features(
                image_sitk,
                supervoxel_sitk,
                labels,
                enabled_features=extractor.enabledFeatures,
                settings=settings,
                image_name=suffix,
                union_bin=self.union_bin,
            )
        else:
            from habit.utils.parallel_gpu_utils import read_worker_gpu_slot_index

            backend, device = resolve_voxel_radiomics_backend(
                use_torch_radiomics=self.use_torch_radiomics,
                torch_device=self.torch_device,
                subject=partition.subject_id,
                gpu_slot_index=read_worker_gpu_slot_index(),
            )
            if backend == "torch" and device is not None:
                extractor.settings["device"] = device
                extractor.settings["dtype"] = resolve_torch_dtype(self.torch_dtype)
            settings = self._radiomics_settings(extractor.settings)

            with injected_torch_radiomics(enabled=(backend == "torch")):
                if backend == "torch":
                    frame = extract_batched_supervoxel_features(
                        image_sitk,
                        supervoxel_sitk,
                        labels,
                        enabled_features=extractor.enabledFeatures,
                        image_name=suffix,
                        settings=settings,
                        device=str(device),
                        dtype_name=self.torch_dtype,
                        batch_size=self.supervoxel_batch,
                    )
                else:
                    frame = extract_supervoxel_features_pyradiomics(
                        image_sitk,
                        supervoxel_sitk,
                        labels,
                        enabled_features=extractor.enabledFeatures,
                        image_name=suffix,
                        settings=settings,
                        batch_size=self.supervoxel_batch,
                    )
        if _V01_LABEL_COLUMN not in frame.columns:
            raise HABITAPIError(
                "supervoxel_radiomics: extraction returned no "
                f"{_V01_LABEL_COLUMN!r} column for modality {modality!r}."
            )
        frame = frame.set_index(_V01_LABEL_COLUMN)
        frame.index = frame.index.astype(np.int64, copy=False)
        frame.index.name = SUPERVOXEL_INDEX_NAME
        return frame.apply(pd.to_numeric, errors="coerce")

    def __call__(
        self,
        subject: Subject,
        partition: Supervoxelization,
    ) -> Supervoxelization:
        """
        Compute per-supervoxel radiomics for one subject.

        Args:
            subject: Subject supplying the original intensity images.
            partition: The subject's supervoxel partition.

        Returns:
            The partition carrying one radiomics feature vector per
            supervoxel, across every requested modality.

        Raises:
            HABITAPIError: If a requested modality is absent or the partition
                holds no supervoxel.
        """
        modalities = resolve_modality_names(
            tuple(subject.images.keys()),
            self.modalities,
            owner="supervoxel_radiomics",
            subject_id=subject.subject_id,
        )
        if not modalities:
            raise HABITAPIError(
                f"supervoxel_radiomics: subject {subject.subject_id!r} carries "
                "no image to extract from."
            )
        labels = partition_labels(partition)

        # The single-modality form may rename the column suffix via ``as_``;
        # the modalities resolution itself is untouched.
        column_labels: Tuple[Optional[str], ...] = (
            (self.as_,) if self.as_ is not None else (None,) * len(modalities)
        )
        frames = [
            self._extract_one_modality(
                subject, partition, modality, labels, column_label=label
            )
            for modality, label in zip(modalities, column_labels)
        ]
        features = frames[0]
        for frame in frames[1:]:
            features = features.join(frame, how="outer")
        if self.output_float32:
            features = features.astype(np.float32)
        return with_features(partition, features, self.spec)

