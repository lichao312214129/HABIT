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
"""Per-habitat radiomics features (PyRadiomics within each habitat label)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from habit.contracts.habitat import HabitatMap
from habit.contracts.subject import Subject
from habit.contracts.table import FeatureTable
from habit.exceptions import HABITAPIError
from habit.habitat_features._base import single_subject_table
from habit.radiomics._domain import (
    DEFAULT_USE_TORCH_RADIOMICS,
    build_pyradiomics_extractor,
    harmonize_mask_geometry,
    resolve_modalities,
    sitk_image_from_contract,
)
from habit.habitat_features.registry import HabitatFeatureExtractorRegistry
from habit.spec.specs import Spec

__all__ = ["EachHabitatRadiomicsFeatures"]


def _extract_each_habitat_modality(
    extractor: Any,
    image_sitk: Any,
    mask_sitk: Any,
    measured: Sequence[int],
    *,
    use_torch_radiomics: Union[str, bool],
    torch_device: str,
    torch_dtype: str,
    subject_id: str,
) -> Dict[int, Dict[str, float]]:
    """
    Extract per-habitat radiomics for one modality via the batched C path.

    Applies the same ``loadImage`` preprocess as ``execute()`` (normalize on
    the full image, optional resample), then one multi-label pass with
    ``union_bin=False`` so ``binWidth`` gray levels stay per habitat.
    Shape features are computed separately (they do not use intensity bins).

    Args:
        extractor: Initialised PyRadiomics ``RadiomicsFeatureExtractor``.
        image_sitk: Intensity SimpleITK image.
        mask_sitk: Multi-label habitat SimpleITK image (already harmonised).
        measured: Habitat ids present in this subject, model order.
        use_torch_radiomics: Torch switch; default False keeps CPU PyRadiomics.
        torch_device: Torch device string or ``"auto"``.
        torch_dtype: ``"float32"`` or ``"float64"``.
        subject_id: Subject id for backend-resolution logging.

    Returns:
        Dict[int, Dict[str, float]]: Habitat id -> execute()-style feature dict
        (keys such as ``original_firstorder_Mean``).
    """
    from radiomics import imageoperations

    from habit.kernels.radiomics.supervoxel_batch import (
        extract_batched_supervoxel_features,
        extract_supervoxel_features_pyradiomics,
    )
    from habit.utils.torch_radiomics_utils import (
        injected_torch_radiomics,
        resolve_torch_dtype,
        resolve_voxel_radiomics_backend,
    )

    settings: Dict[str, Any] = dict(extractor.settings)
    # execute() normalizes the full volume before any crop (whole-image
    # mean/std). Doing it once here is identical for every habitat.
    if settings.get("normalize", False):
        image_sitk = imageoperations.normalizeImage(image_sitk, **settings)
    if (
        settings.get("interpolator") is not None
        and settings.get("resampledPixelSpacing") is not None
    ):
        image_sitk, mask_sitk = imageoperations.resampleImage(
            image_sitk, mask_sitk, **settings
        )

    labels_arr = np.asarray(list(measured), dtype=np.int64)
    settings["use_supervoxel_cext"] = "auto"
    settings["union_bin"] = False

    backend, device = resolve_voxel_radiomics_backend(
        use_torch_radiomics=use_torch_radiomics,
        torch_device=torch_device,
        subject=subject_id or None,
    )
    if backend == "torch" and device is not None:
        settings["device"] = device
        settings["dtype"] = resolve_torch_dtype(torch_dtype)

    with injected_torch_radiomics(enabled=(backend == "torch")):
        if backend == "torch" and device is not None:
            frame = extract_batched_supervoxel_features(
                image_sitk,
                mask_sitk,
                labels_arr,
                enabled_features=extractor.enabledFeatures,
                image_name="",
                settings=settings,
                device=str(device),
                dtype_name=torch_dtype,
                union_bin=False,
            )
        else:
            frame = extract_supervoxel_features_pyradiomics(
                image_sitk,
                mask_sitk,
                labels_arr,
                enabled_features=extractor.enabledFeatures,
                image_name="",
                settings=settings,
                union_bin=False,
            )

    rows: Dict[int, Dict[str, float]] = {}
    for _, row in frame.iterrows():
        habitat_id = int(row["supervoxel_id"])
        features: Dict[str, float] = {}
        for column, value in row.items():
            if column == "supervoxel_id":
                continue
            features[str(column)] = float(value)
        rows[habitat_id] = features

    shape_enabled = any(
        str(name).startswith("shape") for name in extractor.enabledFeatures
    )
    if shape_enabled:
        for habitat_id in measured:
            shape_settings = dict(settings)
            shape_settings["label"] = int(habitat_id)
            try:
                bounding_box, corrected_mask = imageoperations.checkMask(
                    image_sitk, mask_sitk, **shape_settings
                )
                mask_for_shape = (
                    corrected_mask if corrected_mask is not None else mask_sitk
                )
                shape_feats = extractor.computeShape(
                    image_sitk, mask_for_shape, bounding_box, **shape_settings
                )
                dest = rows.setdefault(int(habitat_id), {})
                for key, value in shape_feats.items():
                    dest[str(key)] = float(value)
            except Exception:
                dest = rows.setdefault(int(habitat_id), {})
                dest.setdefault("original_shape_VoxelVolume", float("nan"))

    return rows


@HabitatFeatureExtractorRegistry.register("each_habitat")
class EachHabitatRadiomicsFeatures:
    """
    PyRadiomics features of the raw image(s) within each habitat label.

    This is the v1 form of the v0.1 ``each_habitat`` feature type: for every
    habitat id the model can assign, PyRadiomics runs on each raw modality
    with the multi-label habitat map as mask and the habitat id as label,
    replicating
    ``HabitatRadiomicsExtractor.extract_radiomics_features_from_each_habitat``
    (mask-metadata harmonisation included).

    Where v0.1 wrote one CSV per habitat plus a ``habitat_count.csv``, the
    v1 single-row table carries everything at once:

    * ``has_habitat_{id}`` -- 1.0 when the habitat is present in this
      subject, else 0.0 (the v0.1 habitat-count semantics);
    * ``habitat_{id}_{feature}_of_{modality}`` -- one column per PyRadiomics
      feature, ``NaN`` when the habitat is absent (``NaN`` is the honest
      "not measured"; zero would be a fabricated measurement).

    Columns and their order are canonical for a given extractor
    configuration -- every subject of the same model yields the same layout
    regardless of which habitats it contains. A subject whose map has no
    habitat label at all yields only the ``has_habitat_*`` columns.

    Extraction is one multi-label pass per modality (union-bbox crop, then
    per-habitat ``_applyBinning`` + native C matrices). ``binWidth`` gray
    levels stay per-habitat, matching ``execute(label=id)``. Torch is off
    unless the caller sets ``use_torch_radiomics``.
    """

    def __init__(
        self,
        params_file: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        modalities: Optional[Sequence[str]] = None,
        modality: Optional[str] = None,
        as_: Optional[str] = None,
        use_torch_radiomics: Union[str, bool] = DEFAULT_USE_TORCH_RADIOMICS,
        torch_device: str = "auto",
        torch_dtype: str = "float64",
    ) -> None:
        if modality is not None and modalities is not None:
            raise HABITAPIError(
                "each_habitat: 'modality' and 'modalities' are mutually "
                "exclusive; use 'modality' for the single-modality form."
            )
        if as_ is not None and modality is None:
            raise HABITAPIError(
                "each_habitat: 'as_' requires the single-modality form; "
                "pass 'modality' as well."
            )
        self.params_file = params_file
        self.params = dict(params) if params is not None else None
        self.modalities = (
            (modality,)
            if modality is not None
            else tuple(modalities) if modalities is not None else None
        )
        self.modality = modality
        self.as_ = as_
        self.use_torch_radiomics = use_torch_radiomics
        self.torch_device = str(torch_device)
        self.torch_dtype = str(torch_dtype)
        self._params_file = self.params_file
        self._params = self.params
        self._modalities = self.modalities
        self._modality = self.modality
        self._as = self.as_
        self._use_torch_radiomics = self.use_torch_radiomics
        self._torch_device = self.torch_device
        self._torch_dtype = self.torch_dtype

    @property
    def spec(self) -> Spec:
        """Return the algorithm specification."""
        params: Dict[str, Any] = {
            "params_file": self._params_file,
            "params": self._params,
            "modalities": self._modalities,
            "use_torch_radiomics": self._use_torch_radiomics,
            "torch_device": self._torch_device,
            "torch_dtype": self._torch_dtype,
        }
        # Fold the single-modality spelling in only when used, so existing
        # configurations keep their historical fingerprints.
        if self._modality is not None:
            params["modality"] = self._modality
        if self._as is not None:
            params["as_"] = self._as
        return Spec(name="each_habitat", params=params)

    def __call__(self, subject: Subject, habitat_map: HabitatMap) -> FeatureTable:
        """
        Compute the per-habitat radiomics family for one subject.

        Args:
            subject: Owning subject; every selected modality is extracted.
            habitat_map: Habitat labels; used as the multi-label mask.

        Returns:
            One-row table of per-habitat radiomics plus presence indicators.
        """
        owner = f"habitat_feature_extractor.{self.spec.name}"
        modalities = resolve_modalities(subject, self._modalities, owner=owner)
        extractor = build_pyradiomics_extractor(self._params_file, self._params, owner=owner)

        labels = np.asarray(habitat_map.label_array)
        present = {int(v) for v in np.unique(labels) if v != 0}
        # Extraction runs only for ids the model can assign; labels outside
        # the model's id set indicate a contract violation and are ignored
        # (consistent with the MSI family's use of habitat_ids).
        measured = [h for h in habitat_map.habitat_ids if h in present]

        # Pass 1: one multi-label extraction per modality (union-bbox crop,
        # per-habitat discretize, native C matrices). Column names stay the
        # execute() contract: ``{feature}_of_{modality}``.
        per_habitat: Dict[int, Dict[str, float]] = {}
        base_names: List[str] = []
        for modality in modalities:
            # The ``as_`` alias only renames the column suffix; the image
            # read and the mask handling are untouched.
            suffix = self._as if self._as is not None else modality
            volume = subject.image(modality)
            image_sitk = sitk_image_from_contract(volume.load(), volume.geometry)
            mask_sitk = sitk_image_from_contract(labels, habitat_map.geometry)
            # v0.1 semantics: the mask adopts the raw image's metadata.
            harmonize_mask_geometry(image_sitk, mask_sitk)
            if not measured:
                continue
            for habitat_id, habitat_raw in _extract_each_habitat_modality(
                extractor,
                image_sitk,
                mask_sitk,
                measured,
                use_torch_radiomics=self._use_torch_radiomics,
                torch_device=self._torch_device,
                torch_dtype=self._torch_dtype,
                subject_id=subject.subject_id,
            ).items():
                habitat_features = per_habitat.setdefault(habitat_id, {})
                for key, value in habitat_raw.items():
                    base_name = f"{key}_of_{suffix}"
                    if base_name not in base_names:
                        base_names.append(base_name)
                    habitat_features[base_name] = value

        # Pass 2: emit the canonical column layout for every model habitat,
        # NaN-filling the ones absent from this subject.
        features: Dict[str, float] = {}
        for habitat_id in habitat_map.habitat_ids:
            features[f"has_habitat_{habitat_id}"] = 1.0 if habitat_id in per_habitat else 0.0
            habitat_features = per_habitat.get(habitat_id, {})
            for base_name in base_names:
                features[f"habitat_{habitat_id}_{base_name}"] = habitat_features.get(
                    base_name, float("nan")
                )
        return single_subject_table(
            subject_id=subject.subject_id,
            features=features,
            habitat_map=habitat_map,
            spec=self.spec,
        )

