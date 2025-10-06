# Course Equivalency Mapping System

This document provides a high-level overview of the project architecture and workflow.

## Architecture

```
Streamlit UI (app.py)
│
├── UI Components (src/ui/components.py)
├── Visualisations (src/ui/visualizations.py)
│
├── Similarity Engine (src/models/similarity_engine.py)
├── Course Processor (src/models/course_processor.py)
│
├── Data Utilities (src/utils/data_loader.py, validators.py, logger.py)
└── Configuration (src/config/settings.py)
```

### Data flow
1. Course data is loaded from JSON/CSV files and validated using the utilities in `src/utils`.
2. `CourseProcessor` normalises course metadata, including prerequisite and learning outcome extraction.
3. `SemanticSimilarityEngine` computes multiple similarity signals (semantic, structural, learning outcome) and aggregates them with configurable weights and confidence intervals.
4. The Streamlit UI presents ranked equivalency results, visualisations, and download options.

## Extensibility
- The configuration objects (`src/config/settings.py`) centralise adjustable weights and model parameters for easy experimentation.
- `SemanticSimilarityEngine` exposes an interchangeable embedding backend with a hashing fallback, making it compatible with GPU-enabled or CPU-only environments.
- Logging utilities ensure that additional pipelines (e.g., Pinecone integration, PostgreSQL storage) can reuse the same structured logging configuration.
- The project structure supports future additions such as graph-based similarity (`torch-geometric`) and experiment tracking through `mlflow`.

## Testing strategy
Unit tests target the main transformation and similarity modules (`tests/`). They validate course normalisation, similarity computation integrity, and data loading edge cases to ensure deterministic behaviour in production deployments.
