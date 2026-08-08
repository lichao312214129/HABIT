Traditional Radiomics Configuration
===================================

This page documents the **traditional whole-ROI radiomics** workflow
(``habit radiomics``). It is separate from habitat voxel / supervoxel
radiomics configured under :doc:`habitat`.

Demo: ``config/radiomics/config_traditional_radiomics.yaml``.
PyRadiomics parameter presets live beside it (for example ``parameter.yaml``)
and as bundled defaults under ``habit/resources/radiomics/``.

Command usage: :doc:`../how_to/radiomics`.

**Example configuration:**

.. code-block:: yaml

   paths:
     # params_file optional: omit to use the bundled ROI preset
     # params_file: ./parameter.yaml
     images_folder: ../../demo_data/preprocessed
     out_dir: ../../demo_data/results/radiomics_traditional

   processing:
     n_processes: 2
     save_every_n_files: 5
     process_image_types:
       - pre_contrast
       - LAP
       - PVP
       - delay_3min

   export:
     export_by_image_type: true
     export_combined: true
     export_format: csv
     add_timestamp: false

   logging:
     level: INFO
     console_output: true
     file_output: true

``paths``
---------

- ``images_folder`` (**required**): root that contains ``images/`` and
  ``masks/`` with matching ``<subject>/<modality>/`` layout
- ``out_dir`` (**required**): feature table output directory
- ``params_file`` (optional): PyRadiomics parameter YAML; when omitted, HABIT
  uses the bundled ROI preset

``processing``
--------------

- ``n_processes`` (default ``2``)
- ``save_every_n_files`` (default ``5``): flush intermediate results every N files
- ``process_image_types``: modality folder names to process; ``null`` means all
- ``target_labels``: mask label IDs to extract (default ``[1]``)

``export``
----------

- ``export_by_image_type``: write one table per modality
- ``export_combined``: also write a wide combined table
- ``export_format``: ``csv`` | ``json`` | ``pickle``
- ``add_timestamp``: append a timestamp to output filenames

``logging``
-----------

- ``level``: ``DEBUG`` / ``INFO`` / …
- ``console_output``, ``file_output``

Backward-compatible flat keys
-----------------------------

Deprecated top-level aliases still accepted: ``params_file``, ``images_folder``,
``out_dir``, ``n_processes`` (same meaning as the nested fields above).

PyRadiomics parameter YAML
--------------------------

Files such as ``config/radiomics/parameter.yaml`` follow the upstream
PyRadiomics schema (``imageType``, ``featureClass``, ``setting``). Habitat
voxel / supervoxel presets are documented under :doc:`habitat` and shipped in
``habit/resources/radiomics/``.
