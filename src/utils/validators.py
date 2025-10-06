"""Validation helpers for course data."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List


REQUIRED_FIELDS = {"course_id", "title", "description", "institution", "credits"}


class ValidationError(ValueError):
    """Raised when a validation error occurs."""


def _ensure_iterable_strings(items: Any) -> List[str]:
    """Return a list of strings from a potential iterable or scalar."""

    if items is None:
        return []
    if isinstance(items, str):
        return [items.strip()] if items.strip() else []
    if isinstance(items, Iterable):
        return [str(item).strip() for item in items if str(item).strip()]
    raise ValidationError("Expected an iterable of strings or a string value.")


def validate_course_dict(raw_course: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalise a raw course dictionary."""

    missing = REQUIRED_FIELDS.difference(raw_course)
    if missing:
        raise ValidationError(f"Missing required course fields: {sorted(missing)}")

    course = dict(raw_course)
    try:
        course["credits"] = float(course["credits"])
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
        raise ValidationError("Credits must be numeric") from exc

    course["prerequisites"] = _ensure_iterable_strings(course.get("prerequisites"))
    course["learning_outcomes"] = _ensure_iterable_strings(
        course.get("learning_outcomes")
    )

    return course


__all__ = ["ValidationError", "validate_course_dict"]
