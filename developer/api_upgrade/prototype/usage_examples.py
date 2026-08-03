"""
HABIT v1.0 design prototype -- what calling code looks like at each layer.

STATUS: illustrative. These functions are the design deliverable that shows the
ergonomics of the proposed API; they are not executable against v0.1.x.

The four layers are concentric, not alternatives. Every layer is implemented in
terms of the one below it, so there is exactly one execution path and no risk of
the CLI and the API drifting apart -- the failure mode that made v0.1's API a
mere alias table for CLI commands.

    L0  zero-code      YAML + CLI + GUI            clinicians and trainees
    L1  recipes        one call per study design   notebook researchers
    L2  domain ops     the five protocols          methodologists
    L3  contracts      data model + backends       integrators and agents
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

# Illustrative imports; the shipped package would expose these from ``habit``.
from .contracts import Cohort, FeatureTable, HabitatModel, Subject
from .spec import HabitatSpec, Spec


# ---------------------------------------------------------------------------
# L0 -- zero-code layer. UNCHANGED from v0.1 by design.
# ---------------------------------------------------------------------------
#
#   habit get-habitat -c config/habitat/config_habitat_two_step.yaml -m train
#   habit extract     -c config/feature_extraction/config_extract_features_demo.yaml
#   habit cv          -c config/machine_learning/config_machine_learning_kfold_demo.yaml
#
# The commands, their options, and the existing YAML files keep working. A user
# who does not program cannot tell that the internals were replaced. This is a
# consequence of the architecture rather than an extra compatibility effort:
# the CLI only parses a file and calls L1, so there is nothing in it to break.


# ---------------------------------------------------------------------------
# L1 -- recipes. One call expresses one standard study design.
# ---------------------------------------------------------------------------


def example_l1_two_step_study(data_dir: Path, out_dir: Path) -> None:
    """
    Run a standard two-step habitat study from a notebook.

    This is the layer most clinical researchers should live in: it assumes the
    published two-step design and asks only for the decisions that are genuinely
    the researcher's -- which modalities, how many habitats, which feature
    families.
    """
    import habit

    training = habit.cohort_from_directory(data_dir, name="training")

    study = habit.recipes.two_step_habitat(
        modalities=["delay2", "delay3", "delay5"],
        n_supervoxels=50,
        n_habitats="auto",
        habitat_features=["msi", "ith_score", "non_radiomics"],
        random_seed=42,
    )

    result = study.fit(training)

    # Everything below is in memory. Writing to disk is a separate, explicit act,
    # which is what allows the same code to run inside someone else's pipeline.
    habitat_model: HabitatModel = result.habitat_model
    features: FeatureTable = result.features

    result.save(out_dir)

    # The reporting layer is a first-class deliverable, not an afterthought.
    print(result.manifest.describe_methods(style="radiology"))
    print(result.manifest.checklist("CLEAR"))

    # The exact analysis can be handed to a colleague who only uses the CLI.
    study.spec.to_yaml(out_dir / "habitat_spec.yaml")


def example_l1_external_validation(model_path: Path, external_dir: Path) -> None:
    """
    Apply a previously fitted habitat definition to an external cohort.

    This is the scenario that makes ``HabitatModel`` worth promoting to a
    first-class artefact: external validation becomes two lines, and the habitat
    labels are guaranteed comparable with the development cohort because the
    population-level definition is carried inside the model rather than
    recomputed.
    """
    import habit

    model = habit.HabitatModel.load(model_path)
    external = habit.cohort_from_directory(external_dir, name="external")

    result = habit.recipes.apply_habitat_model(model, external)
    result.features.frame.to_csv("external_habitat_features.csv", index=False)


def example_l1_published_model() -> None:
    """
    Reproduce a published habitat definition on one's own data.

    The strategic goal of the redesign: a habitat definition published with a
    paper circulates the way a pretrained segmentation model does today, so that
    "the five-habitat DCE-MRI model from that study" becomes something other
    groups can actually apply rather than approximately reimplement.
    """
    import habit

    model = habit.HabitatModel.load(Path("hcc_dce_5habitat_v1.habitatmodel"))

    # The model can be inspected before use: which features it needs, which
    # cohort defined it, which software version produced it.
    print(model.summary())


# ---------------------------------------------------------------------------
# L2 -- domain components. Swap any single step for method comparison.
# ---------------------------------------------------------------------------


def example_l2_custom_voxel_features(cohort: Cohort) -> HabitatModel:
    """
    Replace HABIT's voxel feature step with an external model's embeddings.

    A methodologist comparing hand-crafted voxel features against foundation
    model embeddings only has to implement ``VoxelFeatureExtractor``. Nothing
    else in the habitat pipeline changes, and the resulting habitat model is
    still comparable, shareable, and reportable.
    """
    import habit
    from habit.domain import HabitatModelEstimatorFactory, SupervoxelizerFactory

    class FoundationModelVoxelFeatures:
        """Voxel embeddings from a third-party encoder."""

        @property
        def spec(self) -> Spec:
            return Spec(
                domain="feature_extractors",
                name="external.foundation_encoder",
                params={"checkpoint": "encoder_v3", "layer": -2},
            )

        def __call__(self, subject: Subject):
            raise NotImplementedError("user-supplied implementation")

        extract = __call__  # domain-named alias; same function, not a copy

    voxel_features = FoundationModelVoxelFeatures()
    supervoxelizer = SupervoxelizerFactory.create("slic", n_supervoxels=50)
    estimator = HabitatModelEstimatorFactory.create("kmeans", n_clusters=5)

    # Debug the new extractor on one subject first -- no cohort, no backend.
    _probe = voxel_features(cohort[0])

    # Then run it over the cohort. Serial by default; pass ``backend=`` only if
    # parallelism, per-subject timeouts or resume are actually needed.
    fields = cohort.map(voxel_features)
    units = [supervoxelizer(f) for f in fields]
    return estimator.fit(units, cohort=cohort)


def example_l2_habitat_features_only(subject: Subject, model: HabitatModel) -> FeatureTable:
    """
    Use only HABIT's habitat feature mathematics inside a foreign pipeline.

    This is impossible in v0.1, where computing MSI requires accepting HABIT's
    directory layout, configuration file, and output directory. Here MSI is just
    a function of a subject and a habitat map.
    """
    from habit.domain import (
        FeatureExtractorRegistry,
        HabitatFeatureFactory,
        SupervoxelizerFactory,
    )

    supervoxelizer = SupervoxelizerFactory.create("slic", n_supervoxels=50)
    voxel_features = FeatureExtractorRegistry.create("raw", modalities=["T2"])
    assigner = model.assigner()  # the model is bound at construction, not at call

    # Each step is an ordinary callable on one subject, so the whole chain reads
    # as function composition and every intermediate value can be inspected.
    habitat_map = assigner(supervoxelizer(voxel_features(subject)))

    msi = HabitatFeatureFactory.create("msi", voxel_cutoff=10)
    ith = HabitatFeatureFactory.create("ith_score")

    return msi(subject, habitat_map).join(ith(subject, habitat_map))


# ---------------------------------------------------------------------------
# L3 -- contracts. Embedding into the wider ecosystem.
# ---------------------------------------------------------------------------


def example_l3_from_nnunet(dataset_dir: Path) -> Cohort:
    """
    Build a cohort directly from an nnU-Net dataset.

    Segmentation is usually produced by a dedicated tool, so requiring users to
    re-arrange those outputs into HABIT's own folder layout is friction with no
    scientific purpose. A ``DataSource`` implementation removes it entirely.
    """
    import habit

    source = habit.compat.nnunet.NnUNetDataSource(dataset_dir, roi_label=1)
    return source.load()


def example_l3_monai_interop(cohort: Cohort, model: HabitatModel) -> Any:
    """
    Drive HABIT operators with MONAI's ``Compose`` and torch's ``DataLoader``.

    This example only works because subject-level operators are plain callables
    on one sample. That single convention is what lets a PyTorch user keep the
    data plumbing they already have -- caching, batching, worker processes,
    augmentation -- and treat habitat mapping as one more transform, instead of
    handing control of execution over to HABIT.

    It also means HABIT's own execution backend is genuinely optional: torch's
    ``DataLoader`` is a perfectly good parallel runner for subject-level work.
    """
    import habit
    from habit.domain import FeatureExtractorRegistry, SupervoxelizerFactory

    supervoxelizer = SupervoxelizerFactory.create("slic", n_supervoxels=50)
    voxel_features = FeatureExtractorRegistry.create("raw", modalities=["T2"])

    # Compose the subject-level chain into one callable that MONAI can hold.
    to_habitat_map = habit.SubjectPipeline(
        voxel_feature_extractor=voxel_features,
        supervoxelizer=supervoxelizer,
        habitat_assigner=model.assigner(),
    )

    from monai.data import DataLoader, Dataset
    from monai.transforms import Compose

    transform = Compose([habit.compat.monai.AsMonaiDict(), to_habitat_map])
    loader = DataLoader(Dataset(list(cohort), transform=transform), num_workers=4)
    return loader


def example_l3_sklearn_pipeline(cohort: Cohort, labels: Sequence[int]) -> Any:
    """
    Tune the number of habitats with scikit-learn's own model selection.

    Because the domain components expose ``get_params``/``set_params``, the
    habitat step becomes an ordinary stage of a scikit-learn pipeline, and
    questions like "how many habitats generalise best" can be answered with the
    tooling researchers already trust instead of a bespoke HABIT loop.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    import habit

    pipeline = Pipeline(
        [
            ("habitats", habit.compat.sklearn.HabitatFeatures(n_habitats=4)),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )

    search = GridSearchCV(
        pipeline,
        param_grid={"habitats__n_habitats": [3, 4, 5, 6]},
        scoring="roc_auc",
        cv=5,
    )
    return search.fit(cohort, labels)


def example_l3_in_memory_no_disk(subject_arrays: Any) -> FeatureTable:
    """
    Run a habitat analysis with no filesystem involvement at all.

    Required by anyone embedding HABIT in a service or a larger pipeline, and
    structurally impossible in v0.1 where every workflow is defined by its input
    and output directories.
    """
    import habit

    cohort = habit.cohort_from_arrays(subject_arrays)

    study = habit.recipes.two_step_habitat(
        modalities=["T2"],
        n_supervoxels=40,
        n_habitats=4,
        habitat_features=["msi", "ith_score"],
    )
    # No sink is supplied, so nothing is written; the backend defaults to serial.
    return study.fit(cohort).features


def example_l3_agent_discovery() -> None:
    """
    Let an automated research agent discover and configure HABIT.

    Introspection turns the API into something a language model can use
    correctly without transcribing prose documentation: it can enumerate the
    available components, read their JSON Schemas, and emit a specification that
    is validated before anything runs.
    """
    import habit

    # Same three introspection functions v0.1.x already exports; v1.0 only
    # widens their coverage. Note the argument order: (name, domain).
    habit.load_plugins()  # pick up third-party entry points as well

    for info in habit.list_plugins("habitat_features"):
        schema = habit.get_param_schema(info.name, "habitat_features")
        print(info.name, schema.model_json_schema())

    spec = HabitatSpec.from_yaml(Path("agent_generated_spec.yaml"))
    print(spec.describe_methods())
