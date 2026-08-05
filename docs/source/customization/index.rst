Customization and Extension Guide
=================================

This section explains how to customize and extend HABIT components, including preprocessors, feature extractors, clustering algorithms, models, and more.

.. seealso::

   Base classes and interfaces are defined in source code. If custom model metrics depend on **scikit-learn conventions**, see the `sklearn developer guide <https://scikit-learn.org/stable/developers/develop.html>`_ and :doc:`../reference/upstream_libraries`.
   For arithmetic formulas without writing a plugin, use the built-in
   ``expression`` voxel extractor (:doc:`../examples/custom_voxel_features`).

Overview
--------

One of HABIT's design goals is extensibility. Factory patterns and registration let users add custom components easily.

**v1.0 (preferred) — protocol registries**

- Domain string = snake_case singular protocol name
  (e.g. ``voxel_feature_extractor``, ``habitat_model_fitter``)
- Entry-point group = ``habit.<domain>``
- Discover with ``list_plugins("voxel_feature_extractor")`` / ``load_plugins()``

**v0.1 (legacy, still honoured) — factory registries**

- Plural domains such as ``feature_extractors``, ``models``, ``metrics``
- Documented in the sections below that still mention
  ``BaseClusteringExtractor`` / ``FeatureExtractorRegistry``

**Extensible components (v1):**

- **Voxel feature extractors**: ``VoxelFeatureExtractorRegistry``
- **Supervoxelizers / supervoxel features**: matching domain registries
- **Habitat model fitters / assigners / habitat features**
- **Table preprocessors, feature selectors, classifiers, metrics**

**Extension mechanism:**

1. **Registry**: create components with ``Registry.create(name, **params)``
2. **Registration**: ``@Registry.register("name")`` (in-process) or entry points
3. **Protocol**: implement the matching ``habit.domain.protocols`` protocol
4. **Plug and play**: reference the name from ``HabitatSpec`` / YAML

v1 custom voxel feature extractors
----------------------------------

Use this path for DIY formulas that need neighbourhoods, embeddings, or
logic beyond the built-in ``expression`` DSL.

**Step 1: Implement the protocol and register**

.. code-block:: python

   import numpy as np
   from habit.contracts import VoxelFeatureField
   from habit.contracts.subject import Subject
   from habit.domain.voxel_features import (
       VoxelFeatureExtractorRegistry,
       aligned_image,
       build_voxel_field,
       roi_voxels,
   )
   from habit.spec import Spec

   @VoxelFeatureExtractorRegistry.register("t1_t2_contrast")
   class T1T2Contrast:
       def __init__(self, modalities=("T1", "T2"), roi=None, eps=1e-8):
           self.modalities = tuple(modalities)
           self.roi = roi
           self.eps = float(eps)

       @property
       def spec(self) -> Spec:
           return Spec(
               name="t1_t2_contrast",
               params={
                   "modalities": list(self.modalities),
                   "roi": self.roi,
                   "eps": self.eps,
               },
           )

       def __call__(self, subject: Subject) -> VoxelFeatureField:
           mask, inside, index = roi_voxels(subject, self.roi)
           a = aligned_image(subject, self.modalities[0], mask, owner="t1_t2_contrast")
           b = aligned_image(subject, self.modalities[1], mask, owner="t1_t2_contrast")
           values = ((a[inside] - b[inside]) / (a[inside] + b[inside] + self.eps))
           return build_voxel_field(
               subject, mask, index, ("t1_t2_contrast",),
               np.asarray(values).reshape(-1, 1), self.spec,
           )

**Step 2: Use it from a HabitatSpec (or YAML after migration)**

.. code-block:: python

   from habit import HabitatSpec, Spec
   import habit.recipes as recipes

   spec = HabitatSpec(
       name="diy",
       voxel_feature_extractor=Spec("t1_t2_contrast", {"modalities": ["T1", "T2"]}),
       supervoxelizer=Spec("kmeans", {"n_supervoxels": 8, "n_init": 5}),
       habitat_model_fitter=Spec("kmeans", {"n_habitats": 3}),
       habitat_assigner=Spec("nearest_centroid"),
   )
   result = recipes.two_step(cohort, spec)

**Step 3 (optional): ship as a third-party package**

In the plugin's ``pyproject.toml``::

   [project.entry-points."habit.voxel_feature_extractor"]
   t1_t2_contrast = "my_package.features:register"

where ``register()`` performs the ``@VoxelFeatureExtractorRegistry.register``
call (or imports the module that does). Users then::

   from habit import load_plugins
   load_plugins()

See :doc:`../examples/custom_voxel_features` for a runnable demo covering both
``expression`` and a custom plugin, and :doc:`../api/plugins` for discovery.

Extension principles
--------------------

**1. Follow the interface**

All custom components must inherit the appropriate base class and implement required methods:

- **Preprocessor**: inherit ``BasePreprocessor``, implement ``__call__``
- **Feature extractor**: inherit ``BaseClusteringExtractor``, implement ``extract_features``
- **Clustering**: inherit ``BaseClustering``, implement ``fit_predict``
- **Model**: inherit ``BaseModel``, implement ``fit``, ``predict``, ``predict_proba``

**2. Use registration decorators**

Register custom components with the appropriate decorator:

- **Preprocessor**: ``@PreprocessorFactory.register("name")``
- **Feature extractor**: ``@FeatureExtractorRegistry.register('name')``
- **Clustering**: ``@ClusteringAlgorithmFactory.register("name")``
- **Model**: ``@ModelFactory.register("name")``
- **Feature selector**: ``@SelectorRegistry.register('name')``
- **Metric**: ``@MetricRegistry.register('name', display_name='Display')``

**3. Provide clear documentation**

Document custom components clearly:

- **Purpose**: what the component does and when to use it
- **Parameters**: meaning and defaults
- **Examples**: usage examples
- **Notes**: caveats and limitations

**4. Test and validate**

Test custom components thoroughly:

- **Unit tests**: basic functionality
- **Integration tests**: interaction with other components
- **Performance tests**: performance characteristics
- **Validation**: correctness checks

Custom preprocessors
--------------------

**Step 1: Create a custom preprocessor**

.. code-block:: python

   from habit.compat.engines.preprocessing.preprocessor_factory import PreprocessorFactory
   from habit.compat.engines.preprocessing.base_preprocessor import BasePreprocessor

   @PreprocessorFactory.register("my_preprocessor")
   class MyPreprocessor(BasePreprocessor):
       def __init__(self, keys, allow_missing_keys=False,**kwargs):
           super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
           self.param1 = kwargs.get('param1', default_value)
           self.param2 = kwargs.get('param2', default_value)

       def __call__(self, data):
           self._check_keys(data)
           for key in self.keys:
               data[key] = self._process_item(data[key])
           return data

       def _process_item(self, item):
           # Implement your preprocessing logic here
           return processed_item

**Step 2: Use in configuration**

.. code-block:: yaml

   preprocessing:
     my_preprocessor:
       images: [T1, T2]
       param1: value1
       param2: value2

**Step 3: Run preprocessing**

.. code-block:: bash

   habit preprocess --config config_with_custom_preprocessor.yaml

**Example: custom Gaussian filter preprocessor**

.. code-block:: python

   import numpy as np
   from scipy.ndimage import gaussian_filter
   from habit.compat.engines.preprocessing.preprocessor_factory import PreprocessorFactory
   from habit.compat.engines.preprocessing.base_preprocessor import BasePreprocessor

   @PreprocessorFactory.register("gaussian_filter")
   class GaussianFilterPreprocessor(BasePreprocessor):
       def __init__(self, keys, allow_missing_keys=False,**kwargs):
           super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
           self.sigma = kwargs.get('sigma', 1.0)
           self.order = kwargs.get('order', 0)

       def __call__(self, data):
           self._check_keys(data)
           for key in self.keys:
               data[key] = self._process_item(data[key])
           return data

       def _process_item(self, item):
           return gaussian_filter(item, sigma=self.sigma, order=self.order)

Custom feature extractors
---------------------------

**Step 1: Create a custom feature extractor**

.. code-block:: python

   from habit.compat.engines.habitat_analysis.clustering_features.base_extractor import BaseClusteringExtractor
   from habit.compat.engines.habitat_analysis.clustering_features.base_extractor import FeatureExtractorRegistry

   @FeatureExtractorRegistry.register('my_feature_extractor')
   class MyFeatureExtractor(BaseClusteringExtractor):
       def __init__(self,**kwargs):
           super().__init__(**kwargs)
           self.feature_names = ['feature1', 'feature2', 'feature3']

       def extract_features(self, image_data,**kwargs):
           # Implement feature extraction logic.
           n_samples = image_data.shape[0]
           features = np.random.random((n_samples, 3))
           return features

**Step 2: Use in configuration**

.. code-block:: yaml

   feature_construction:
     voxel_level:
       method: my_feature_extractor(raw(delay2), raw(delay3))
       params:
         param1: value1

**Step 3: Run habitat analysis**

.. code-block:: bash

   habit get-habitat --config config_with_custom_extractor.yaml

**Example: custom local contrast feature extractor**

.. code-block:: python

   import numpy as np
   from habit.compat.engines.habitat_analysis.clustering_features.base_extractor import BaseClusteringExtractor
   from habit.compat.engines.habitat_analysis.clustering_features.base_extractor import FeatureExtractorRegistry

   @FeatureExtractorRegistry.register('local_contrast')
   class LocalContrastExtractor(BaseClusteringExtractor):
       def __init__(self,**kwargs):
           super().__init__(**kwargs)
           self.radius = kwargs.get('radius', 3)
           self.feature_names = ['local_contrast']

       def extract_features(self, image_data,**kwargs):
           n_samples = image_data.shape[0]
           features = np.zeros((n_samples, 1))
           for i in range(n_samples):
               features[i, 0] = self._compute_local_contrast(image_data[i])
           return features

       def _compute_local_contrast(self, image):
           local_mean = self._compute_local_mean(image)
           local_contrast = np.abs(image - local_mean)
           return local_contrast

       def _compute_local_mean(self, image):
           from scipy.ndimage import uniform_filter
           return uniform_filter(image, size=self.radius * 2 + 1)

Custom clustering algorithms
----------------------------

**Step 1: Create a custom clustering algorithm**

.. code-block:: python

   from habit.compat.engines.habitat_analysis.clustering.base_clustering import BaseClustering
   from habit.compat.engines.habitat_analysis.clustering.base_clustering import ClusteringAlgorithmFactory

   @ClusteringAlgorithmFactory.register("my_clustering")
   class MyClusteringAlgorithm(BaseClustering):
       def __init__(self, n_clusters=3, random_state=None,**kwargs):
           super().__init__(n_clusters=n_clusters, random_state=random_state)
           self.param1 = kwargs.get('param1', default_value)

       def fit_predict(self, X,**kwargs):
           # Implement clustering logic.
           labels = self._cluster(X)
           return labels

       def _cluster(self, X):
           # Implement the concrete clustering algorithm.
           return labels

**Step 2: Use in configuration**

.. code-block:: yaml

   habitat_segmentation:
     clustering_mode: two_step
     supervoxel:
       algorithm: my_clustering
       n_clusters: 50
       param1: value1

**Step 3: Run habitat analysis**

.. code-block:: bash

   habit get-habitat --config config_with_custom_clustering.yaml

**Example: custom spectral clustering**

.. code-block:: python

   import numpy as np
   from sklearn.cluster import SpectralClustering
   from habit.compat.engines.habitat_analysis.clustering.base_clustering import BaseClustering
   from habit.compat.engines.habitat_analysis.clustering.base_clustering import ClusteringAlgorithmFactory

   @ClusteringAlgorithmFactory.register("spectral")
   class SpectralClusteringAlgorithm(BaseClustering):
       def __init__(self, n_clusters=3, random_state=None,**kwargs):
           super().__init__(n_clusters=n_clusters, random_state=random_state)
           self.gamma = kwargs.get('gamma', 1.0)
           self.n_neighbors = kwargs.get('n_neighbors', 10)

       def fit_predict(self, X,**kwargs):
           clustering = SpectralClustering(
               n_clusters=self.n_clusters,
               gamma=self.gamma,
               n_neighbors=self.n_neighbors,
               random_state=self.random_state
           )
           labels = clustering.fit_predict(X)
           return labels

Custom models
-------------

**Step 1: Create a custom model**

.. code-block:: python

   from habit.compat.engines.machine_learning.models.base import BaseModel
   from habit.compat.engines.machine_learning.models.factory import ModelFactory

   @ModelFactory.register("my_model")
   class MyModel(BaseModel):
       def __init__(self,**kwargs):
           super().__init__(**kwargs)
           self.param1 = kwargs.get('param1', default_value)
           self.model = None

       def fit(self, X, y,**kwargs):
           # Implement training logic
           self.model = self._train(X, y)
           return self

       def predict(self, X,**kwargs):
           # Implement prediction logic
           return self.model.predict(X)

       def predict_proba(self, X,**kwargs):
           # Implement probability prediction logic
           return self.model.predict_proba(X)

       def _train(self, X, y):
           # Implement the concrete training algorithm
           return model

**Step 2: Use in configuration**

.. code-block:: yaml

   models:
     my_model:
       params:
         param1: value1

**Step 3: Run machine learning**

.. code-block:: bash

   habit model --config config_with_custom_model.yaml

**Example: custom neural network model**

.. code-block:: python

   import numpy as np
   from sklearn.neural_network import MLPClassifier
   from habit.compat.engines.machine_learning.models.base import BaseModel
   from habit.compat.engines.machine_learning.models.factory import ModelFactory

   @ModelFactory.register("neural_network")
   class NeuralNetworkModel(BaseModel):
       def __init__(self,**kwargs):
           super().__init__(**kwargs)
           self.hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (100,))
           self.activation = kwargs.get('activation', 'relu')
           self.solver = kwargs.get('solver', 'adam')
           self.max_iter = kwargs.get('max_iter', 200)
           self.random_state = kwargs.get('random_state', None)
           self.model = None

       def fit(self, X, y,**kwargs):
           self.model = MLPClassifier(
               hidden_layer_sizes=self.hidden_layer_sizes,
               activation=self.activation,
               solver=self.solver,
               max_iter=self.max_iter,
               random_state=self.random_state
           )
           self.model.fit(X, y)
           return self

       def predict(self, X,**kwargs):
           return self.model.predict(X)

       def predict_proba(self, X,**kwargs):
           return self.model.predict_proba(X)

Custom feature selectors
------------------------

**Step 1: Create a custom feature selector**

.. code-block:: python

   from typing import List
   from habit.compat.engines.machine_learning.feature_selectors.selector_registry import (
       SelectorRegistry,
       SelectorContext,
   )

   # A feature selector is a function: it receives a SelectorContext and
   # returns the list of feature names to KEEP.
   @SelectorRegistry.register('my_selector')
   def my_selector(context: SelectorContext, param1=1.0, param2=10) -> List[str]:
       X = context.X                      # pandas DataFrame (current features)
       y = context.y                      # pandas Series (target)
       candidate_features = context.selected_features

       # Implement the concrete selection algorithm.
       kept = [f for f in candidate_features if _keep(X[f], y, param1, param2)]
       return kept

**Step 2: Use in configuration**

.. code-block:: yaml

   feature_selection_methods:
     - method: my_selector
       params:
         param1: value1
         param2: value2

**Step 3: Run machine learning**

.. code-block:: bash

   habit model --config config_with_custom_selector.yaml

**Example: custom mutual information feature selector**

.. code-block:: python

   import numpy as np
   from typing import List
   from sklearn.feature_selection import mutual_info_classif
   from habit.compat.engines.machine_learning.feature_selectors.selector_registry import (
       SelectorRegistry,
       SelectorContext,
   )

   @SelectorRegistry.register('mutual_info')
   def mutual_info_selector(
       context: SelectorContext, k_features: int = 10, random_state: int = None
   ) -> List[str]:
       X = context.X
       y = context.y
       scores = mutual_info_classif(X.values, y, random_state=random_state)
       top_idx = np.argsort(scores)[-k_features:]
       return [X.columns[i] for i in top_idx]

Best practices
--------------

**1. Naming conventions**

- Use clear, descriptive names
- Use lowercase letters and underscores
- Avoid abbreviations

**Examples:**

.. code-block:: python

   # Good naming
   @PreprocessorFactory.register("gaussian_filter")
   @FeatureExtractorRegistry.register('local_contrast')
   @ClusteringAlgorithmFactory.register("spectral")

   # Poor naming
   @PreprocessorFactory.register("gf")
   @FeatureExtractorRegistry.register('lc')
   @ClusteringAlgorithmFactory.register("spec")

**2. Parameter validation**

Validate input parameters to ensure they are valid.

**Example:**

.. code-block:: python

   def __init__(self, sigma=1.0,**kwargs):
       super().__init__(**kwargs)
       if sigma <= 0:
           raise ValueError("sigma must be positive")
       self.sigma = sigma

**3. Docstrings**

Provide clear docstrings for custom components.

**Example:**

.. code-block:: python

   @PreprocessorFactory.register("gaussian_filter")
   class GaussianFilterPreprocessor(BasePreprocessor):
       """
       Gaussian filter preprocessor.

       Applies Gaussian smoothing to reduce noise.

       Parameters
       ----------
       sigma : float, default=1.0
           Standard deviation of the Gaussian kernel. Larger values smooth more.
       order : int, default=0
           Order of the Gaussian filter. 0 = smoothing, 1 = first derivative, 2 = second derivative.

       Notes
       -----
       - Gaussian filtering blurs fine detail
       - Larger sigma values produce stronger smoothing
       """

       def __init__(self, keys, allow_missing_keys=False,**kwargs):
           super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
           self.sigma = kwargs.get('sigma', 1.0)
           self.order = kwargs.get('order', 0)

**4. Error handling**

Provide clear error messages for debugging.

**Example:**

.. code-block:: python

   def __call__(self, data):
       self._check_keys(data)
       for key in self.keys:
           try:
               data[key] = self._process_item(data[key])
           except Exception as e:
               raise RuntimeError(f"Failed to process {key}: {str(e)}")
       return data

**5. Testing**

Write tests for custom components to ensure correctness.

**Example:**

.. code-block:: python

   import unittest
   import numpy as np

   class TestGaussianFilterPreprocessor(unittest.TestCase):
       def setUp(self):
           self.preprocessor = GaussianFilterPreprocessor(
               keys=['image'],
               sigma=1.0
           )

       def test_gaussian_filter(self):
           data = {'image': np.random.random((10, 10, 10))}
           result = self.preprocessor(data)
           self.assertIn('image', result)
           self.assertEqual(result['image'].shape, (10, 10, 10))

   if __name__ == '__main__':
       unittest.main()

FAQ
---

**Q1: How do I debug a custom component?**

A: You can:

1. Enable verbose logging with ``debug`` mode
2. Add ``print`` statements in code
3. Use the Python debugger (pdb)
4. Write unit tests

**Q2: How do I share a custom component?**

A: You can:

1. Share code with other researchers
2. Create a GitHub repository
3. Submit to the HABIT project
4. Write documentation and examples

**Q3: How do I optimize performance of a custom component?**

A: Try:

1. Vectorized operations
2. Parallel computation
3. C/C++ extensions
4. Algorithm optimization

**Q4: How do I ensure correctness of a custom component?**

A: You can:

1. Write unit tests
2. Compare against known results
3. Use visualization for validation
4. Run cross-validation

**Q5: How do I handle dependencies for a custom component?**

A: You can:

1. Document dependencies
2. Provide installation instructions
3. Use virtual environments
4. Provide a requirements file (requirements.txt)

Next steps
----------

After extending HABIT, you may:

- :doc:`../configuration/index`: detailed configuration reference
- :doc:`../development/index`: HABIT architecture and extension mechanisms
- :doc:`../reference/cli`: CLI commands
