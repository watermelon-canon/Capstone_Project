"""Similarity engine for comparing course records."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency not installed
    SentenceTransformer = None

from src.config.settings import AppSettings, load_settings
from src.models.course_processor import CourseRecord
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class SimilarityResult:
    """Container for similarity results."""

    reference: CourseRecord
    target: CourseRecord
    scores: Dict[str, float]
    weighted_score: float
    confidence_interval: float
    explanation: str


class SemanticSimilarityEngine:
    """Compute similarity between courses using multiple strategies."""

    def __init__(self, settings: Optional[AppSettings] = None) -> None:
        self.settings = settings or load_settings()
        self._model = self._load_model()
        self._hashing_vectorizer = HashingVectorizer(
            n_features=512, alternate_sign=False, norm="l2"
        )
        self._semantic_cache: Dict[str, np.ndarray] = {}
        self._outcomes_cache: Dict[str, np.ndarray] = {}

    def _load_model(self) -> Optional[SentenceTransformer]:
        if not SentenceTransformer:  # pragma: no cover - optional branch
            LOGGER.warning(
                "sentence-transformers is not available. Falling back to hashing embeddings."
            )
            return None
        try:  # pragma: no cover - depends on environment
            LOGGER.info(
                "Loading sentence transformer model: %s",
                self.settings.model.model_name,
            )
            return SentenceTransformer(self.settings.model.model_name)
        except Exception as exc:  # pragma: no cover - optional branch
            LOGGER.warning("Failed to load transformer model. Fallback activated: %s", exc)
            return None

    def _encode_text(self, text: str, cache: Dict[str, np.ndarray]) -> np.ndarray:
        if text in cache:
            return cache[text]
        if self._model:
            embedding = self._model.encode(text, convert_to_numpy=True)
        else:
            embedding = self._hashing_vectorizer.transform([text]).toarray()[0]
        cache[text] = embedding
        return embedding

    def _semantic_embedding(self, course: CourseRecord) -> np.ndarray:
        return self._encode_text(course.combined_text(), self._semantic_cache)

    def _learning_outcome_embedding(self, course: CourseRecord) -> np.ndarray:
        text = "; ".join(course.learning_outcomes) or course.description
        return self._encode_text(text, self._outcomes_cache)

    @staticmethod
    def _structural_features(course: CourseRecord) -> np.ndarray:
        prereq_count = len(course.prerequisites)
        outcome_count = len(course.learning_outcomes)
        description_length = len(course.description.split())
        return np.array([course.credits, prereq_count, outcome_count, description_length])

    @staticmethod
    def _calculate_confidence(scores: Dict[str, float], settings: AppSettings) -> float:
        components = np.array(list(scores.values()))
        std = components.std(ddof=0)
        if std == 0:
            return 0.0
        stderr = std / np.sqrt(len(components))
        return settings.confidence.z_score * stderr

    def _score_pair(self, reference: CourseRecord, target: CourseRecord) -> SimilarityResult:
        semantic_sim = float(
            cosine_similarity(
                [self._semantic_embedding(reference)],
                [self._semantic_embedding(target)],
            )[0][0]
        )
        structural_sim = self._structural_similarity(reference, target)
        outcome_sim = float(
            cosine_similarity(
                [self._learning_outcome_embedding(reference)],
                [self._learning_outcome_embedding(target)],
            )[0][0]
        )
        scores = {
            "semantic": semantic_sim,
            "structural": structural_sim,
            "learning_outcomes": outcome_sim,
        }
        weights = self.settings.similarity_weights.normalized()
        weighted_score = sum(scores[name] * weight for name, weight in weights.items())
        confidence_interval = self._calculate_confidence(scores, self.settings)
        explanation = self._build_explanation(reference, target, scores, weights)
        return SimilarityResult(
            reference=reference,
            target=target,
            scores=scores,
            weighted_score=weighted_score,
            confidence_interval=confidence_interval,
            explanation=explanation,
        )

    def _structural_similarity(self, reference: CourseRecord, target: CourseRecord) -> float:
        ref_features = self._structural_features(reference)
        tgt_features = self._structural_features(target)
        if np.all(ref_features == 0) and np.all(tgt_features == 0):
            return 0.0
        norm = np.linalg.norm(ref_features) * np.linalg.norm(tgt_features)
        if norm == 0:
            return 0.0
        similarity = float(np.dot(ref_features, tgt_features) / norm)
        return similarity

    @staticmethod
    def _build_explanation(
        reference: CourseRecord,
        target: CourseRecord,
        scores: Dict[str, float],
        weights: Dict[str, float],
    ) -> str:
        highlights = [
            f"Semantic alignment {scores['semantic']:.2f}",
            f"Structural match {scores['structural']:.2f}",
            f"Outcome overlap {scores['learning_outcomes']:.2f}",
        ]
        weight_info = ", ".join(f"{name} weight {weight:.2f}" for name, weight in weights.items())
        return (
            f"Compared '{reference.title}' with '{target.title}'. "
            f"Key signals: {', '.join(highlights)}. Weights applied: {weight_info}."
        )

    def rank_courses(
        self,
        reference_courses: Iterable[CourseRecord],
        target_courses: Iterable[CourseRecord],
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return a ranked dataframe of course equivalencies."""

        results: List[SimilarityResult] = []
        for reference in reference_courses:
            for target in target_courses:
                results.append(self._score_pair(reference, target))

        data = [
            {
                "reference_course_id": result.reference.course_id,
                "reference_institution": result.reference.institution,
                "target_course_id": result.target.course_id,
                "target_institution": result.target.institution,
                "weighted_score": result.weighted_score,
                "semantic_similarity": result.scores["semantic"],
                "structural_similarity": result.scores["structural"],
                "learning_outcome_similarity": result.scores["learning_outcomes"],
                "confidence_interval": result.confidence_interval,
                "explanation": result.explanation,
            }
            for result in results
        ]

        frame = pd.DataFrame(data)
        frame.sort_values(by="weighted_score", ascending=False, inplace=True)
        if top_k is not None:
            frame = frame.head(top_k)
        frame.reset_index(drop=True, inplace=True)
        return frame


__all__ = ["SemanticSimilarityEngine", "SimilarityResult"]
