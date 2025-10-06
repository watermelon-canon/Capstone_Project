"""Tests for the semantic similarity engine."""
from __future__ import annotations

from src.models.course_processor import CourseRecord
from src.models.similarity_engine import SemanticSimilarityEngine


def make_course(course_id: str, description: str, **kwargs) -> CourseRecord:
    return CourseRecord(
        course_id=course_id,
        institution=kwargs.get("institution", "A"),
        title=kwargs.get("title", course_id),
        description=description,
        credits=kwargs.get("credits", 3.0),
        prerequisites=kwargs.get("prerequisites", []),
        learning_outcomes=kwargs.get("learning_outcomes", []),
    )


def test_rank_courses_returns_scored_dataframe() -> None:
    reference = make_course(
        "REF-1",
        "Programming fundamentals covering variables, loops, and conditionals.",
        learning_outcomes=["Write basic programs"],
    )
    target = make_course(
        "TGT-1",
        "Intro to programming with focus on Python, loops, and conditionals.",
        learning_outcomes=["Develop scripts using control flow"],
    )
    engine = SemanticSimilarityEngine()
    results = engine.rank_courses([reference], [target])
    assert not results.empty
    row = results.iloc[0]
    assert 0.0 <= row["weighted_score"] <= 1.0
    assert row["semantic_similarity"] > 0
    assert row["confidence_interval"] >= 0
