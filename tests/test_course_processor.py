"""Tests for the course processor."""
from __future__ import annotations

from src.models.course_processor import CourseProcessor


def test_process_course_extracts_prerequisites_and_code() -> None:
    processor = CourseProcessor(language_model=None)
    raw_course = {
        "course_id": "TEST-100",
        "institution": "Test University",
        "title": "Intro to Testing",
        "description": "Course code: TEST-100. Learn testing. Prerequisites: None; Experience in programming. Outcomes: students will apply unit testing.",
        "credits": 3,
        "prerequisites": [],
        "learning_outcomes": [],
    }
    record = processor.process_course(raw_course)
    assert record.course_id == "TEST-100"
    assert "Experience in programming" in record.prerequisites
    assert record.metadata.get("course_code") == "TEST-100"
    assert any("unit testing" in outcome.lower() for outcome in record.learning_outcomes)
