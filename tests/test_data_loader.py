"""Tests for the data loading utilities."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.data_loader import load_courses


def test_load_courses_from_sample_dataset() -> None:
    courses = load_courses(Path("data/sample_courses.json"))
    assert len(courses) >= 30
    first = courses[0]
    assert {"course_id", "title", "description", "institution", "credits"}.issubset(first)


def test_load_courses_invalid_path(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_courses(missing_file)
