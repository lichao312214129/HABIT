Customization and Extension Guide
=================================

This section explains how to customize and extend HABIT components, including preprocessors, feature extractors, clustering algorithms, models, and more.

.. seealso::

   Base classes and interfaces are defined in source code. If custom model metrics depend on **scikit-learn conventions**, see the `sklearn developer guide <https://scikit-learn.org/stable/developers/develop.html>`_ and :doc:`../reference/upstream_libraries`.

Overview
--------

One of HABIT's design goals is extensibility. Factory patterns and registration let users add custom components easily.

**Extensible components:**

- **Preprocessors**: custom image preprocessing methods
- **Feature extractors**: custom clustering feature extractors
- **Clustering algorithms**: custom clustering algorithms
- **Strategies**: custom habitat segmentation strategies
- **Models**: custom machine learning models
- **Feature selectors**: custom feature selection methods

**Extension mechanism:**

HABIT uses factory patterns and registration:

1. **Factory pattern**: all extensible components are created via factories
2. **Registration**: register custom components with decorators
3. **Unified interface**: custom components follow a common interface
4. **Plug and play**: once registered, use in configuration files

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
- **Feature extractor**: ``@register_feature_extractor('name')``
- **Clustering**: ``@register_clustering("name")``
- **Model**: ``@ModelFactory.register("name")``

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

   from habit.core.preprocessing.preprocessor_factory import PreprocessorFactory
   from habit.core.preprocessing.base_preprocessor import BasePreprocessor

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

   Preprocessing:
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
   from habit.core.preprocessing.preprocessor_factory import PreprocessorFactory
   from habit.core.preprocessing.base_preprocessor import BasePreprocessor

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

   from habit.core.habitat_analysis.clustering_features.base_extractor import BaseClusteringExtractor
   from habit.core.habitat_analysis.clustering_features.base_extractor import register_feature_extractor

   @register_feature_extractor('my_feature_extractor')
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

   FeatureConstruction:
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
   from habit.core.habitat_analysis.clustering_features.base_extractor import BaseClusteringExtractor
   from habit.core.habitat_analysis.clustering_features.base_extractor import register_feature_extractor

   @register_feature_extractor('local_contrast')
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

   from habit.core.habitat_analysis.clustering.base_clustering import BaseClustering
   from habit.core.habitat_analysis.clustering.base_clustering import register_clustering

   @register_clustering("my_clustering")
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

   HabitatSegmentation:
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
   from habit.core.habitat_analysis.clustering.base_clustering import BaseClustering
   from habit.core.habitat_analysis.clustering.base_clustering import register_clustering

   @register_clustering("spectral")
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

   from habit.core.machine_learning.models.base import BaseModel
   from habit.core.machine_learning.models.factory import ModelFactory

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
   from habit.core.machine_learning.models.base import BaseModel
   from habit.core.machine_learning.models.factory import ModelFactory

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

   from sklearn.base import BaseEstimator, TransformerMixin
   from habit.core.machine_learning.feature_selectors.selector_registry import register_selector

   @register_selector('my_selector')
   class MyFeatureSelector(BaseEstimator, TransformerMixin):
       def __init__(self, param1=default_value, param2=default_value):
           self.param1 = param1
           self.param2 = param2
           self.selected_features_ = None

       def fit(self, X, y=None):
           # Implement feature selection logic
           self.selected_features_ = self._select_features(X, y)
           return self

       def transform(self, X):
           # Implement feature transformation logic
           return X[:, self.selected_features_]

       def _select_features(self, X, y):
           # Implement the concrete selection algorithm
           return selected_indices

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
   from sklearn.feature_selection import mutual_info_classif
   from sklearn.base import BaseEstimator, TransformerMixin
   from habit.core.machine_learning.feature_selectors.selector_registry import register_selector

   @register_selector('mutual_info')
   class MutualInfoSelector(BaseEstimator, TransformerMixin):
       def __init__(self, k_features=10, random_state=None):
           self.k_features = k_features
           self.random_state = random_state
           self.selected_features_ = None
           self.scores_ = None

       def fit(self, X, y):
           scores = mutual_info_classif(X, y, random_state=self.random_state)
           self.scores_ = scores
           self.selected_features_ = np.argsort(scores)[-self.k_features:]
           return self

       def transform(self, X):
           return X[:, self.selected_features_]

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
   @register_feature_extractor('local_contrast')
   @register_clustering("spectral")

   # Poor naming
   @PreprocessorFactory.register("gf")
   @register_feature_extractor('lc')
   @register_clustering("spec")

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
