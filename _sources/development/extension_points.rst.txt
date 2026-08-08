Extension and Plugin Guide
==========================

HABIT is highly decoupled. Developers can inject custom algorithm components
through the **Registry** mechanism without modifying core source code.

This guide is based on HABIT's code contracts and explains how to extend
preprocessors, machine-learning models, and feature selectors.

Core extension contracts
------------------------

HABIT extensions follow three contracts:

1. **Behavior contract**: subclass the required base class, such as
   ``BasePreprocessor``, or implement the required signature, such as
   ``SelectorContext -> List[str]``.
2. **Registration contract**: use the appropriate decorator, such as
   ``@PreprocessorFactory.register``, to register the component name globally.
3. **Parameter contract**: define a Pydantic schema and register it with
   ``ParamSchemaRegistry.register`` for strongly typed YAML validation.

.. important::
   
   **Module import behavior**: registration decorators run only when the
   Python module is imported. External plugins must be imported before the
   workflow starts. For in-tree development, import the module from the
   corresponding subpackage ``__init__.py``.

Example 1: Custom preprocessing step (ClassRegistry)
----------------------------------------------------

Preprocessors are class-based components managed by ``PreprocessorFactory``.

**1. Implement the behavior and registration name**

Subclass ``BasePreprocessor``, implement ``__call__``, and register the class
with ``@PreprocessorFactory.register``.

.. code-block:: python

   import numpy as np
   from scipy.ndimage import gaussian_filter
   from habit.compat.engines.preprocessing.preprocessor_factory import PreprocessorFactory
   from habit.compat.engines.preprocessing.base_preprocessor import BasePreprocessor

   @PreprocessorFactory.register("my_gaussian_filter")
   class MyGaussianFilter(BasePreprocessor):
       def __init__(self, keys, allow_missing_keys=False, **kwargs):
           super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
           # Extract parameters.
           self.sigma = kwargs.get('sigma', 1.0)

       def __call__(self, data):
           self._check_keys(data)
           for key in self.keys:
               data[key] = gaussian_filter(data[key], sigma=self.sigma)
           return data

**2. Define and register the parameter schema**

Define and register the parameter model in
``habit/schemas/steps/preprocessing.py`` (or in your plugin module).

.. code-block:: python

   from pydantic import BaseModel, Field
   from habit.schemas.registry import ParamSchemaRegistry

   class MyGaussianFilterParams(BaseModel):
       sigma: float = Field(default=1.0, description="Gaussian kernel standard deviation.")

   # Register with ParamSchemaRegistry (domain="preprocessing").
   ParamSchemaRegistry.register("preprocessing", "my_gaussian_filter", MyGaussianFilterParams)

**3. Call the component from YAML**

.. code-block:: yaml

   preprocessing:
     my_gaussian_filter:
       images: [T1, T2]
       sigma: 2.5

Example 2: Custom feature selector (CallableRegistry)
-----------------------------------------------------

Unlike scikit-learn, which requires a complete ``BaseEstimator`` class,
**HABIT feature selectors are designed as pure functions**. The underlying
``pipeline_builder.py`` wraps them as sklearn-compatible transformers.

**1. Implement the function and registration name**

The selector receives a ``SelectorContext`` containing ``X``, ``y``, and
``selected_features`` and returns the retained feature names as ``List[str]``.

.. code-block:: python

   from typing import List
   from habit.compat.engines.machine_learning.feature_selectors.selector_registry import (
       SelectorRegistry, 
       SelectorContext
   )

   @SelectorRegistry.register("my_variance_selector", display_name="Custom Variance")
   def my_variance_selector(context: SelectorContext, threshold: float = 0.0) -> List[str]:
       """
       Custom variance feature selector.
       """
       X = context.X
       # Compute variance.
       variances = X.var(axis=0)
       # Keep features above the threshold.
       retained_features = variances[variances > threshold].index.tolist()
       
       context.logger.info(f"Retained {len(retained_features)} features.")
       return retained_features

**2. Define and register the parameter schema**

.. code-block:: python

   from pydantic import BaseModel, Field
   from habit.schemas.registry import ParamSchemaRegistry

   class MyVarianceParams(BaseModel):
       threshold: float = Field(default=0.0, description="Variance threshold.")

   # Register with ParamSchemaRegistry (domain="feature_selection").
   ParamSchemaRegistry.register("feature_selection", "my_variance_selector", MyVarianceParams)

**3. Call the component from YAML**

.. code-block:: yaml

   feature_selection_methods:
     - method: my_variance_selector
       params:
         threshold: 0.5

Example 3: Custom machine-learning model (ModelFactory)
-------------------------------------------------------

The ``ModelFactory`` inherits from ``ClassRegistry[BaseModel]``. Its
constructor contract is a single ``config`` dictionary.

**1. Implement the model and registration name**

Subclass ``BaseModel`` and implement ``fit``, ``predict``, and ``predict_proba``.

.. code-block:: python

   from sklearn.neural_network import MLPClassifier
   from habit.compat.engines.machine_learning.models.base import BaseModel
   from habit.compat.engines.machine_learning.models.factory import ModelFactory

   @ModelFactory.register("my_mlp")
   class MyMLPModel(BaseModel):
       def __init__(self, config: dict):
           super().__init__(config)
           # Read values from the config dictionary.
           hidden_layer_sizes = config.get('hidden_layer_sizes', (100,))
           random_state = config.get('random_state', 42)
           
           self.model = MLPClassifier(
               hidden_layer_sizes=hidden_layer_sizes,
               random_state=random_state
           )

       def fit(self, X, y, **kwargs):
           self.model.fit(X, y)
           return self

       def predict(self, X, **kwargs):
           return self.model.predict(X)

       def predict_proba(self, X, **kwargs):
           return self.model.predict_proba(X)

**2. Define and register the parameter schema**

.. code-block:: python

   from typing import Tuple
   from pydantic import BaseModel, Field
   from habit.schemas.registry import ParamSchemaRegistry

   class MyMLPParams(BaseModel):
       hidden_layer_sizes: Tuple[int, ...] = Field(default=(100,))
       random_state: int = Field(default=42)

   # Register with ParamSchemaRegistry (domain="model").
   ParamSchemaRegistry.register("model", "my_mlp", MyMLPParams)

Example 4: Custom clustering feature extractor
(``FeatureExtractorRegistry`` + ``method_param_spec``)
-------------------------------------------------------------------------------------------------

Clustering feature extractors are discovered lazily through
``FeatureExtractorRegistry`` in ``*_extractor.py`` modules. In addition to
implementing ``BaseClusteringExtractor``, declare ``method_param_spec`` on the
class so functional ``method`` expressions can validate bindings and inject
defaults:

.. code-block:: python

   from habit.compat.engines.habitat_analysis.clustering_features.base_extractor import (
       BaseClusteringExtractor,
       FeatureExtractorRegistry,
   )
   from habit.compat.engines.habitat_analysis.clustering_features.method_param_spec import (
       MethodParamSpec,
   )

   @FeatureExtractorRegistry.register("my_texture")
   class MyTextureExtractor(BaseClusteringExtractor):
       method_param_spec = MethodParamSpec(
           required=(),                         # names that MUST appear in F(...)
           optional={"window_size": 5},         # built-in defaults when omitted
           default_params_file_preset=None,       # or "voxel"/"supervoxel"/"roi"/"habitat"
           combiner=False,
           takes_image=True,
       )

       def extract_features(self, image_data, mask_data, **kwargs):
           ...

YAML usage (parentheses declare bindings; ``params`` assigns values only):

.. code-block:: yaml

   feature_construction:
     voxel_level:
       method: concat(my_texture(T2, window_size))
       params:
         window_size: 7

``params_file`` is optional; radiomics methods use bundled presets from
``habit/resources/radiomics/``.

Extension point reference
-------------------------

HABIT has **eight registries** (six class-based factories and two
function-based registries). The table below is the complete reference.

.. list-table::
   :header-rows: 1
   :widths: 22 26 28 24

   * - Component type
     - Registration decorator
     - Behavior contract (base class/signature)
     - Schema domain
   * - **Preprocessing step**
     - ``@PreprocessorFactory.register("name")``
     - Subclass ``BasePreprocessor``
     - ``preprocessing``
   * - **Machine-learning model**
     - ``@ModelFactory.register("name")``
     - Subclass ``BaseModel``
     - ``model``
   * - **Clustering algorithm**
     - ``@ClusteringAlgorithmFactory.register("name")``
     - Subclass ``BaseClustering``
     - (Habitat subsystem; no independent domain)
   * - **Clustering feature extractor**
     - ``FeatureExtractorRegistry`` (lazy discovery of ``*_extractor.py``)
     - Subclass ``BaseClusteringExtractor``
     - (Habitat subsystem)
   * - **Feature-table preprocessing method**
     - ``@PreprocessingMethodFactory.register("name")``
     - Subclass ``BaseFeaturePreprocessing``
     - (Habitat subsystem)
   * - **Post-segmentation habitat feature**
     - ``@HabitatFeatureFactory.register("name")``
     - Subclass ``BaseHabitatFeature``; resolve with ``get_handler()``
     - (Habitat subsystem)
   * - **Feature selector**
     - ``@SelectorRegistry.register("name")``
     - Function ``(SelectorContext) -> List[str]``
     - ``feature_selection``
   * - **Evaluation metric**
     - ``@MetricRegistry.register("name")``
     - Function ``(y_true, y_pred, y_prob, cm=None) -> float``
     - (Pure function; usually no parameter schema)

.. note::

   **Registry categories**: the first six are **class-based factories**
   (subclasses of ``ClassRegistry`` whose products are instantiated with
   ``create()``); the final two are **function-based registries**
   (subclasses of ``CallableRegistry`` whose products are functions). Both
   follow the shared registry interface and are protected by architecture
   contract tests; see :doc:`invariants`.

.. seealso::

   See :doc:`../customization/index` for complete copy-ready component
   templates and :doc:`dev_workflow` for adding new components to contract
   tests.