"""Course processing utilities."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    import spacy
    from spacy.language import Language
except Exception:  # pragma: no cover - optional dependency not installed
    spacy = None
    Language = None

from src.utils.logger import get_logger
from src.utils.validators import ValidationError, validate_course_dict

LOGGER = get_logger(__name__)


@dataclass
class CourseRecord:
    """Normalized course representation used throughout the system."""

    course_id: str
    institution: str
    title: str
    description: str
    credits: float
    prerequisites: List[str] = field(default_factory=list)
    learning_outcomes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def combined_text(self) -> str:
        """Return a concatenated text representation for embedding."""

        parts = [self.title, self.description]
        if self.prerequisites:
            parts.append("Prerequisites: " + "; ".join(self.prerequisites))
        if self.learning_outcomes:
            parts.append("Learning Outcomes: " + "; ".join(self.learning_outcomes))
        return " \n".join(part.strip() for part in parts if part.strip())


class CourseProcessor:
    """Transform raw course dictionaries into :class:`CourseRecord` objects."""

    def __init__(self, language_model: str | None = "en_core_web_sm") -> None:
        self._nlp: Optional[Language] = None
        if spacy and language_model:
            try:  # pragma: no cover - depends on runtime environment
                self._nlp = spacy.load(language_model)
            except Exception as exc:  # pragma: no cover - optional branch
                LOGGER.warning("Falling back to regex metadata extraction: %s", exc)
        self._course_pattern = re.compile(r"course\s*(?:code|id)[:\s]+([A-Za-z0-9\-]+)", re.I)
        self._prereq_pattern = re.compile(r"prereq(?:uisite)?s?:?(.+)", re.I)
        self._outcome_pattern = re.compile(r"outcome(?:s)?[:\-]?(.+)", re.I)

    @staticmethod
    def clean_text(text: str) -> str:
        """Return a cleaned version of the provided text."""

        cleaned = re.sub(r"\s+", " ", text or "").strip()
        return cleaned

    def _extract_with_spacy(self, text: str) -> Dict[str, Any]:
        """Extract entities using spaCy when available."""

        if not self._nlp:  # pragma: no cover - optional branch
            return {}
        doc = self._nlp(text)
        credits = None
        prerequisites: List[str] = []
        outcomes: List[str] = []
        for ent in doc.ents:
            label = ent.label_.lower()
            if label in {"quantity", "cardinal"} and "credit" in ent.sent.text.lower():
                try:
                    credits = float(ent.text)
                except ValueError:  # pragma: no cover - defensive
                    continue
            if label in {"work_of_art", "product"} and "prereq" in ent.sent.text.lower():
                prerequisites.append(ent.text)
        if not outcomes:
            outcomes.extend(self._split_clauses(text, ["students will", "able to"]))
        payload: Dict[str, Any] = {}
        if credits is not None:
            payload["credits_override"] = credits
        if prerequisites:
            payload["prerequisites"] = prerequisites
        if outcomes:
            payload["learning_outcomes"] = outcomes
        return payload

    @staticmethod
    def _split_clauses(text: str, markers: Iterable[str]) -> List[str]:
        """Split text into clauses around the provided markers."""

        lowered = text.lower()
        for marker in markers:
            if marker in lowered:
                parts = re.split(r"[\.;]\s*", text)
                return [part.strip() for part in parts if marker in part.lower()]
        return []

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata such as prerequisites and learning outcomes."""

        metadata = {}
        if not text:
            return metadata

        if self._nlp:  # pragma: no branch - depends on runtime
            metadata.update(self._extract_with_spacy(text))

        prereq_match = self._prereq_pattern.search(text)
        if prereq_match and "prerequisites" not in metadata:
            prereq_text = prereq_match.group(1).split("Outcome")[0]
            metadata["prerequisites"] = [
                chunk.strip().rstrip(".")
                for chunk in re.split(r"[;,]", prereq_text)
                if chunk.strip()
            ]

        outcome_match = self._outcome_pattern.search(text)
        if outcome_match and "learning_outcomes" not in metadata:
            outcome_text = outcome_match.group(1)
            metadata["learning_outcomes"] = [
                chunk.strip().rstrip(".")
                for chunk in re.split(r"[.;]", outcome_text)
                if chunk.strip()
            ]

        code_match = self._course_pattern.search(text)
        if code_match:
            metadata.setdefault("course_code", code_match.group(1))

        return metadata

    def process_course(self, raw_course: Dict[str, Any]) -> CourseRecord:
        """Validate and transform a single raw course."""

        try:
            course = validate_course_dict(raw_course)
        except ValidationError as exc:
            LOGGER.error("Invalid course data for %s: %s", raw_course.get("course_id"), exc)
            raise

        description = self.clean_text(course["description"])
        metadata = self.extract_metadata(description)
        if "credits_override" in metadata:
            course["credits"] = metadata.pop("credits_override")

        prerequisites = metadata.get("prerequisites", course.get("prerequisites", []))
        learning_outcomes = metadata.get(
            "learning_outcomes", course.get("learning_outcomes", [])
        )

        record = CourseRecord(
            course_id=str(course["course_id"]),
            institution=str(course["institution"]),
            title=self.clean_text(course["title"]),
            description=description,
            credits=float(course["credits"]),
            prerequisites=prerequisites,
            learning_outcomes=learning_outcomes,
            metadata={key: value for key, value in metadata.items() if key not in {"prerequisites", "learning_outcomes"}},
        )
        return record

    def process_courses(self, courses: Iterable[Dict[str, Any]]) -> List[CourseRecord]:
        """Process a sequence of raw course dictionaries."""

        processed = [self.process_course(course) for course in courses]
        LOGGER.info("Processed %s courses", len(processed))
        return processed


__all__ = ["CourseProcessor", "CourseRecord"]
