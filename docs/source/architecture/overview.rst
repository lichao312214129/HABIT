Architecture Overview
=====================

System Architecture
------------------

HABIT follows a modular architecture with clear separation of concerns:

- **Entry Layer**: CLI commands and configuration loading
- **Service Layer**: Dependency injection and service creation
- **Manager Layer**: Feature, clustering, and result management
- **Strategy Layer**: Different clustering strategies
- **Pipeline Layer**: sklearn-style pipeline execution

For detailed architecture documentation, see the ARCHITECTURE.md file in the source code:
```
habit/core/habitat_analysis/ARCHITECTURE.md
```
