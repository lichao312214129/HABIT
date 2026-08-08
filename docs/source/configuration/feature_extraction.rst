Feature Extraction Configuration
==================================

This section documents **feature extraction** configuration. CLI: ``habit extract -c <yaml>``. Demo example: ``config/feature_extraction/config_extract_features_demo.yaml``.

**Example configuration file:**

.. code-block:: yaml

   params_file_of_non_habitat: ./parameter.yaml
   params_file_of_habitat: ./parameter_habitat.yaml

   raw_img_folder: ./demo_data/preprocessed
   habitats_map_folder: ./demo_data/results/habitat_two_step
   out_dir: ./demo_data/results/features

   n_processes: 3
   habitat_pattern: '*_habitats.nrrd'

   feature_types:
     - volume
     - msi
     - ith_score
     - non_radiomics
     # Heavy PyRadiomics families (opt-in; require pyradiomics):
     # - traditional
     # - whole_habitat
     # - each_habitat

   n_habitats:

   debug: false

**params_file_of_non_habitat**: parameter file for features extracted from raw images

- **Type**: string
- **Required**: no
- **Default**: ``null`` (bundled ``roi`` preset → ``habit/resources/radiomics/parameter.yaml``)
- **Description**: PyRadiomics parameter file for traditional / each_habitat radiomics on raw images
- **Example**: ``./parameter.yaml``

**params_file_of_habitat**: parameter file for features extracted from habitat maps

- **Type**: string
- **Required**: no
- **Default**: ``null`` (bundled ``habitat`` preset → ``habit/resources/radiomics/parameter_habitat.yaml``)
- **Description**: PyRadiomics parameter file for whole_habitat radiomics on the label map
- **Example**: ``./parameter_habitat.yaml``

**raw_img_folder**: root directory of raw images

- **Type**: string
- **Required**: yes
- **Default**: none (required)
- **Description**: contains preprocessed images
- **Example**: ``./demo_data/preprocessed``

**habitats_map_folder**: root directory of habitat maps

- **Type**: string
- **Required**: yes
- **Default**: none (required)
- **Description**: contains generated habitat maps
- **Example**: ``./results/habitat``

**out_dir**: output directory

- **Type**: string
- **Required**: yes
- **Default**: none (required)
- **Description**: feature files are saved here
- **Example**: ``./results/features``

**debug** (``FeatureExtractionConfig``)

- **Type**: boolean
- **Default**: ``false``

**n_processes**: number of parallel processes

- **Type**: integer
- **Required**: no
- **Default**: ``4`` (built-in default for feature extraction config)
- **Description**: number of processes for parallel processing
- **Example**: ``3``

**habitat_pattern**: habitat file glob pattern

- **Type**: string
- **Required**: no
- **Default**: ``'*_habitats.nrrd'``
- **Description**: pattern to match habitat map files; supports wildcards (``*``)
- **Example**: ``*_habitats.nrrd``

**feature_types**: list of feature types

- **Type**: list
- **Required**: yes
- **Default**: none (required; at least one item)
- **Description**: types not in the list are not extracted
- **Allowed values**: ``volume``, ``msi``, ``ith_score``, ``non_radiomics``, ``traditional``, ``whole_habitat``, ``each_habitat``
- **Example**: ``[volume, msi, ith_score, non_radiomics]`` (uncomment heavy radiomics when needed)
- **Meanings and references per type**: see :doc:`../reference/features/index`

**n_habitats**: number of habitats

- **Type**: integer or null
- **Required**: no
- **Default**: ``null`` (auto-detect)
- **Description**: can manually specify habitat count
- **Example**: ``null``
