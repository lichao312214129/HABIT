Ecosystem adapters (``habit.compat``)
=====================================

Optional bridges. Core HABIT never imports these backends unless you do.

scikit-learn (``habit.compat.sklearn``)
---------------------------------------

.. code-block:: python

   from sklearn.linear_model import LogisticRegression
   from sklearn.pipeline import Pipeline

   from habit.compat.sklearn import as_classifier, as_estimator, as_transformer
   from habit.domain import ZScorePreprocessor
   from habit.spec import HabitatSpec, Spec

   habitat_spec = HabitatSpec(
       name="direct",
       voxel_feature_extractor=Spec("raw", {"modalities": ["T1"]}),
       supervoxelizer=None,
       habitat_model_fitter=Spec("kmeans", {"n_habitats": 2}),
       habitat_assigner=Spec("nearest_centroid"),
       habitat_features=(
           Spec("volume"),
           Spec("msi"),
           Spec("ith_score"),
           Spec("non_radiomics"),
           # Heavy PyRadiomics families (opt-in; require pyradiomics):
           # Spec("traditional"),
           # Spec("whole_habitat"),
           # Spec("each_habitat"),
       ),
       random_seed=0,
   )

   # HabitatSpec -> sklearn Transformer; fit learns HabitatModel once
   habitat_est = as_estimator(
       habitat_spec,
       n_habitats=2,
       random_seed=1,
   )
   X = habitat_est.fit_transform(list(cohort))  # Sequence[Subject]
   names = habitat_est.get_feature_names_out()
   model = habitat_est.model_  # HabitatModel

   table_step = as_transformer(ZScorePreprocessor())
   clf_step = as_classifier(domain_classifier)

   pipe = Pipeline(
       [
           ("habitat", as_estimator(habitat_spec)),
           ("scale", table_step),
           ("clf", LogisticRegression()),
       ]
   )

Also exported: ``HabitatFeaturesEstimator``, ``TableTransformerEstimator``,
``TableClassifierEstimator``.

MONAI (``habit.compat.monai``)
------------------------------

.. code-block:: python

   from habit.compat.monai import (
       AsDictTransform,
       AsMonaiDict,
       FromMonaiDict,
       from_monai_dict,
       to_monai_dict,
   )

   monai_dict = to_monai_dict(subject, channel_first=False)
   subject2 = from_monai_dict(monai_dict, subject_id="P001")

   as_monai = AsMonaiDict()
   from_monai = FromMonaiDict(subject_id_key="subject_id")
   # Wrap a subject-level HABIT operator as a dict transform:
   # transform = AsDictTransform(habit_operator)

nnU-Net (``habit.compat.nnunet``)
---------------------------------

.. code-block:: python

   from habit.compat.nnunet import NnUNetDataSource

   source = NnUNetDataSource(
       "Dataset001_Tumor",   # path to nnU-Net raw dataset root
       roi_label=1,          # int, name, or union of ints
   )
   cohort = source.load()    # contracts.Cohort with FileImageRef images
