"""Data loading utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .logger import get_logger
from .validators import validate_course_dict

LOGGER = get_logger(__name__)


def _load_json(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        payload = payload.get("courses", [])
    if not isinstance(payload, list):
        raise ValueError("JSON course file must contain a list or 'courses' key.")
    return payload


def _load_csv(path: Path) -> Iterable[Dict[str, Any]]:
    frame = pd.read_csv(path)
    return frame.to_dict(orient="records")


def load_courses(path: str | Path) -> List[Dict[str, Any]]:
    """Load courses from a CSV or JSON file and validate them."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Course file not found: {path}")

    LOGGER.info("Loading courses from %s", file_path)
    if file_path.suffix.lower() == ".json":
        raw_courses = _load_json(file_path)
    elif file_path.suffix.lower() == ".csv":
        raw_courses = _load_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or JSON.")

    validated = [validate_course_dict(course) for course in raw_courses]
    LOGGER.info("Loaded %s course records", len(validated))
    return validated


__all__ = ["load_courses"]
