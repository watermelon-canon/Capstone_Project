"""Configuration objects for the Course Equivalency Mapping System."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class SimilarityWeights:
    """Weight configuration for the different similarity strategies."""

    semantic: float = 0.6
    structural: float = 0.2
    learning_outcomes: float = 0.2

    def normalized(self) -> Dict[str, float]:
        """Return a normalized weight dictionary."""
        total = self.semantic + self.structural + self.learning_outcomes
        if total == 0:
            raise ValueError("Similarity weights must sum to a non-zero value.")
        return {
            "semantic": self.semantic / total,
            "structural": self.structural / total,
            "learning_outcomes": self.learning_outcomes / total,
        }


@dataclass
class ConfidenceIntervalSettings:
    """Settings related to confidence interval estimation."""

    z_score: float = 1.96
    minimum_pairs: int = 3


@dataclass
class ModelSettings:
    """Settings for the similarity model and optional vector index."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 16
    cache_embeddings: bool = True
    pinecone_index_name: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_api_key: Optional[str] = None


@dataclass
class AppSettings:
    """Aggregate application settings."""

    similarity_weights: SimilarityWeights = field(default_factory=SimilarityWeights)
    confidence: ConfidenceIntervalSettings = field(default_factory=ConfidenceIntervalSettings)
    model: ModelSettings = field(default_factory=ModelSettings)


DEFAULT_SETTINGS = AppSettings()
# Module-level default settings used throughout the application.


def load_settings() -> AppSettings:
    """Return the default settings.

    The function exists for future overrides (for example environment-based
    configuration loading). Keeping it simple ensures deterministic behaviour
    during testing while remaining extendable.
    """

    return DEFAULT_SETTINGS
