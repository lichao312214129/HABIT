# Same pattern as CLI: load validated config then run via HabitatConfigurator.
# On Windows ``multiprocessing`` uses spawn; top-level ``fit()`` would re-run this
# file in worker processes unless guarded by ``__main__`` (runtime error otherwise).

from habit.core.habitat_analysis.config_schemas import HabitatAnalysisConfig
from habit.core.common.configurators.habitat import HabitatConfigurator


def main() -> None:
    config = HabitatAnalysisConfig.from_file("./demo_data/config_habitat_two_step.yaml")
    configurator = HabitatConfigurator(config=config)
    analysis = configurator.create_habitat_analysis()
    analysis.fit()


if __name__ == "__main__":
    main()
