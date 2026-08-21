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
"""Voxel-wise radiomics: a texture feature vector for every voxel.

This is the family behind texture-driven habitats. PyRadiomics computes one
feature map per enabled feature by sliding a kernel over the ROI; the maps are
then read back into a voxel-by-feature table. The map alignment and table
assembly live in :mod:`habit.kernels.radiomics.voxel_maps`, shared with the
v0.1 extractor, so both paths yield identical numbers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from habit.contracts.habitat import VoxelFeatureField
from habit.contracts.subject import Subject
from habit.domain.voxel_features._base import (
    build_voxel_field,
    resolve_source_modalities,
    resolve_voxel_modalities,
    roi_voxels,
)
from habit.domain.voxel_features.registry import VoxelFeatureExtractorRegistry
from habit.exceptions import HABITAPIError
from habit.spec.specs import Spec

__all__ = ["VoxelRadiomicsFeatures", "VoxelRadiomicsFeaturesParams"]

#: v0.1 default: a 7x7x7 neighbourhood (radius 3), the CT habitat setting of
#: Prior O, et al., Radiol Artif Intell 2024;6(2):e230118.
DEFAULT_KERNEL_RADIUS = 3

#: v0.1 default voxel batch; balances memory against speed. PyRadiomics reads
#: -1 as "all ROI voxels at once".
DEFAULT_VOXEL_BATCH = 1000


class VoxelRadiomicsFeaturesParams(BaseModel):
    """Constructor parameters for :class:`VoxelRadiomicsFeatures`."""

    model_config = ConfigDict(extra="forbid")
    modality: Optional[str] = None
    modalities: Sequence[str] = ()
    as_: Optional[str] = None
    roi: Optional[str] = None
    params_file: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    kernel_radius: int = Field(default=DEFAULT_KERNEL_RADIUS, gt=0)
    voxel_batch: int = DEFAULT_VOXEL_BATCH
    use_torch_radiomics: Union[str, bool] = "auto"
    torch_device: str = "auto"
    torch_dtype: str = "float32"
    use_gpu_matrices: Union[str, bool] = "auto"
    output_float32: bool = True
    class_progress: bool = False


@VoxelFeatureExtractorRegistry.register("voxel_radiomics")
class VoxelRadiomicsFeatures:
    """
    Describe every ROI voxel by a PyRadiomics feature vector.

    One extraction pass runs per modality and each column is suffixed with
    ``-{modality}``, the v0.1 scheme for
    ``concat(voxel_radiomics(m1), voxel_radiomics(m2))``.

    Args:
        modality: Single modality key -- the explicit form used inside
            feature trees. Mutually exclusive with ``modalities``.
        modalities: Modality keys to extract from, in feature order; empty
            selects every image the subject carries.
        as_: Optional output-column alias. Valid only with exactly one
            resolved modality; the column suffix then uses the alias.
        roi: Mask key defining the region of interest; ``None`` uses the
            subject's single mask.
        params_file: Path to a PyRadiomics parameter YAML; ``None`` selects the
            bundled voxel preset.
        params: Inline PyRadiomics settings, for API callers holding settings
            in memory. Mutually exclusive with ``params_file``.
        kernel_radius: Neighbourhood radius in voxels; radius 1 is a 3x3x3
            cube, radius 3 a 7x7x7 cube.
        voxel_batch: ROI voxels PyRadiomics processes per batch.
        use_torch_radiomics: ``"auto"``, ``True`` or ``False`` -- whether to
            use the TorchRadiomics path when torch and CUDA are present.
        torch_device: Torch device string, or ``"auto"`` to select one.
        torch_dtype: ``"float32"`` or ``"float64"`` for the torch path.
        use_gpu_matrices: ``"auto"``, ``True`` or ``False`` -- whether the
            TorchRadiomics texture matrices (GLCM, ...) are built on GPU by
            ``habit.kernels.radiomics.gpumatrices`` instead of the
            single-threaded PyRadiomics C extension. ``"auto"`` follows the
            torch device. Bit-identical counts either way.
        output_float32: Downcast the feature columns to float32, the v0.1
            default that keeps large voxel tables manageable.
        class_progress: When True, print and tqdm each PyRadiomics class
            (firstorder, glcm, ...). Default False: one ``execute()`` with
            no per-class lines; ``Cohort.map`` still shows subject progress.
    """

    def __init__(
        self,
        modalities: Sequence[str] = (),
        roi: Optional[str] = None,
        params_file: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        kernel_radius: int = DEFAULT_KERNEL_RADIUS,
        voxel_batch: int = DEFAULT_VOXEL_BATCH,
        use_torch_radiomics: Union[str, bool] = "auto",
        torch_device: str = "auto",
        torch_dtype: str = "float32",
        use_gpu_matrices: Union[str, bool] = "auto",
        output_float32: bool = True,
        class_progress: bool = False,
        modality: Optional[str] = None,
        as_: Optional[str] = None,
    ) -> None:
        if params_file is not None and params is not None:
            raise HABITAPIError(
                "voxel_radiomics: params_file and params are mutually "
                "exclusive; pass the PyRadiomics settings as a file path OR "
                "as a mapping."
            )
        if kernel_radius < 1:
            raise HABITAPIError(
                f"kernel_radius must be positive; got {kernel_radius}."
            )
        resolved, labels = resolve_source_modalities(
            modality, modalities, as_, owner="voxel_radiomics"
        )
        self.modalities = resolved
        self.source_labels = labels
        self.modality = str(modality) if modality is not None else None
        self.as_ = str(as_) if as_ is not None else None
        self.roi = roi
        self.params_file = params_file
        self.params = dict(params) if params is not None else None
        self.kernel_radius = int(kernel_radius)
        self.voxel_batch = int(voxel_batch)
        self.use_torch_radiomics = use_torch_radiomics
        self.torch_device = str(torch_device)
        self.torch_dtype = str(torch_dtype)
        self.use_gpu_matrices = use_gpu_matrices
        self.output_float32 = bool(output_float32)
        self.class_progress = bool(class_progress)

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification used for provenance."""
        spec_params: Dict[str, Any] = {
            "modalities": list(self.modalities),
            "roi": self.roi,
            "params_file": self.params_file,
            "params": self.params,
            "kernel_radius": self.kernel_radius,
            "voxel_batch": self.voxel_batch,
            "use_torch_radiomics": self.use_torch_radiomics,
            "torch_device": self.torch_device,
            "torch_dtype": self.torch_dtype,
            "use_gpu_matrices": self.use_gpu_matrices,
            "output_float32": self.output_float32,
        }
        # Fold the singular/alias forms in only when set so the historical
        # ``modalities=[...]`` fingerprint stays byte-identical.
        if self.modality is not None:
            spec_params["modality"] = self.modality
        if self.as_ is not None:
            spec_params["as_"] = self.as_
        # Default False is omitted so quiet extraction keeps the old fingerprint.
        if self.class_progress:
            spec_params["class_progress"] = True
        return Spec(name="voxel_radiomics", params=spec_params)

    def _resolved_params_file(self) -> Optional[str]:
        """
        Return the parameter file to use, falling back to the voxel preset.

        Returns:
            A path, or ``None`` when inline ``params`` were supplied.
        """
        if self.params is not None:
            return None
        from habit.utils.radiomics_preset_utils import resolve_params_file

        return resolve_params_file(self.params_file, preset="voxel")

    def _extract_one_modality(
        self,
        subject: Subject,
        modality: str,
        mask_sitk: Any,
        column_label: Optional[str] = None,
    ) -> Any:
        """
        Run one voxel-based PyRadiomics pass over the ROI of one modality.

        Args:
            subject: Subject supplying the intensity image.
            modality: Modality to extract from.
            mask_sitk: ROI mask as a SimpleITK image.
            column_label: Output column suffix; defaults to the modality
                name. The ``as_`` alias lands here.

        Returns:
            A voxel-by-feature frame for this modality, ROI rows in C order.
        """
        from habit.domain.habitat_features._radiomics import (
            build_pyradiomics_extractor,
            sitk_image_from_contract,
        )
        from habit.kernels.radiomics.voxel_maps import voxel_feature_frame
        from habit.utils.radiomics_params_utils import (
            configure_voxel_glcm_on_extractor,
        )
        from habit.utils.parallel_gpu_utils import read_worker_gpu_slot_index
        from habit.utils.torch_radiomics_utils import (
            execute_voxel_based_with_class_progress,
            injected_torch_radiomics,
            release_cuda_cache,
            resolve_torch_dtype,
            resolve_voxel_radiomics_backend,
        )

        volume = subject.image(modality)
        image_sitk = sitk_image_from_contract(volume.load(), volume.geometry)
        # v0.1 semantics: the mask adopts the image's metadata verbatim so
        # PyRadiomics accepts the pair without a geometry complaint.
        mask_sitk.CopyInformation(image_sitk)

        extractor = build_pyradiomics_extractor(
            self._resolved_params_file(), self.params, owner="voxel_radiomics"
        )
        configure_voxel_glcm_on_extractor(extractor)
        backend, device = resolve_voxel_radiomics_backend(
            use_torch_radiomics=self.use_torch_radiomics,
            torch_device=self.torch_device,
            subject=subject.subject_id,
            # Process-pool workers export HABIT_GPU_SLOT_INDEX; honour it so
            # multi-GPU pools do not all pile onto cuda:0.
            gpu_slot_index=read_worker_gpu_slot_index(),
        )
        extractor.settings.update(
            {
                "kernelRadius": self.kernel_radius,
                "voxelBatch": self.voxel_batch,
                "geometryTolerance": 1e-3,
                "use_gpu_matrices": self.use_gpu_matrices,
            }
        )
        if backend == "torch" and device is not None:
            extractor.settings["device"] = device
            extractor.settings["dtype"] = resolve_torch_dtype(self.torch_dtype)

        configured_label = extractor.settings.get("label", 1)
        mask_label = 1 if configured_label is None else int(configured_label)

        with injected_torch_radiomics(enabled=(backend == "torch")):
            # Quiet by default: one execute() with no per-class tqdm / stderr.
            # class_progress=True restores the instrumented per-class loop.
            if self.class_progress:
                result = execute_voxel_based_with_class_progress(
                    extractor, image_sitk, mask_sitk, voxel_based=True
                )
            else:
                result = extractor.execute(image_sitk, mask_sitk, voxelBased=True)
                release_cuda_cache()
        del extractor

        return voxel_feature_frame(
            result,
            mask_sitk,
            image_name=column_label if column_label is not None else modality,
            mask_label=mask_label,
            output_float32=self.output_float32,
        )

    def __call__(self, subject: Subject) -> VoxelFeatureField:
        """
        Compute per-voxel radiomics for one subject.

        Args:
            subject: Subject providing the requested modalities and mask.

        Returns:
            One row per ROI voxel, one column per feature and modality.

        Raises:
            HABITAPIError: If a requested modality is absent, or a modality's
                extraction does not cover every ROI voxel.
        """
        from habit.domain.habitat_features._radiomics import sitk_image_from_contract

        modalities = resolve_voxel_modalities(
            subject, self.modalities, owner="voxel_radiomics"
        )
        # ``resolve_voxel_modalities`` may expand an empty request to every
        # subject image; labels track that expansion one-to-one.
        labels = (
            self.source_labels
            if len(self.source_labels) == len(modalities)
            else modalities
        )
        mask, _, voxel_index = roi_voxels(subject, self.roi)
        mask_array = np.asarray(mask.data)

        names: List[str] = []
        columns: List[np.ndarray] = []
        for modality, label in zip(modalities, labels):
            # A fresh mask image per modality: PyRadiomics rewrites the mask
            # metadata to match the image it is paired with.
            frame = self._extract_one_modality(
                subject,
                modality,
                sitk_image_from_contract(mask_array, mask.geometry),
                column_label=label,
            )
            if frame.shape[0] != voxel_index.shape[0]:
                raise HABITAPIError(
                    f"voxel_radiomics: modality {modality!r} of subject "
                    f"{subject.subject_id!r} produced {frame.shape[0]} rows "
                    f"for {voxel_index.shape[0]} ROI voxels."
                )
            names.extend(str(column) for column in frame.columns)
            columns.append(frame.to_numpy())

        values = np.concatenate(columns, axis=1)
        return build_voxel_field(
            subject, mask, voxel_index, names, values, self.spec
        )


VoxelFeatureExtractorRegistry.register_params_model(
    "voxel_radiomics", VoxelRadiomicsFeaturesParams
)
