"""Reusable Streamlit UI components."""
from __future__ import annotations

from typing import Sequence

import pandas as pd
import streamlit as st

from src.models.course_processor import CourseRecord


def sidebar_filters(courses: Sequence[CourseRecord]) -> dict:
    """Render sidebar filters and return selected options."""

    institutions = sorted({course.institution for course in courses})
    default_institution = institutions[0] if institutions else None
    reference_institution = st.sidebar.selectbox(
        "Reference institution", institutions, index=0 if default_institution else -1
    )
    target_institution = st.sidebar.selectbox(
        "Target institution",
        institutions,
        index=1 if len(institutions) > 1 else 0,
    )
    top_k = st.sidebar.slider("Number of matches", min_value=5, max_value=50, value=10)
    return {
        "reference_institution": reference_institution,
        "target_institution": target_institution,
        "top_k": top_k,
    }


def render_course_summary(course: CourseRecord) -> None:
    """Display course information in the main panel."""

    st.subheader(f"{course.course_id} – {course.title}")
    st.markdown(f"**Institution:** {course.institution}")
    st.markdown(f"**Credits:** {course.credits}")
    if course.prerequisites:
        st.markdown("**Prerequisites:** " + ", ".join(course.prerequisites))
    if course.learning_outcomes:
        st.markdown("**Learning outcomes:**")
        for outcome in course.learning_outcomes:
            st.markdown(f"- {outcome}")
    st.markdown("**Description:**")
    st.write(course.description)


def render_results_table(results: pd.DataFrame) -> None:
    """Display ranked similarity results."""

    if results.empty:
        st.info("No matching courses were found for the selected filters.")
    else:
        st.dataframe(
            results[
                [
                    "reference_course_id",
                    "target_course_id",
                    "weighted_score",
                    "confidence_interval",
                    "semantic_similarity",
                    "structural_similarity",
                    "learning_outcome_similarity",
                    "explanation",
                ]
            ],
            width="stretch",
        )


def render_download_button(results: pd.DataFrame) -> None:
    """Render a download button for the results dataframe."""

    if results.empty:
        return
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download equivalency report (CSV)",
        data=csv_bytes,
        file_name="course_equivalency_report.csv",
        mime="text/csv",
    )


__all__ = [
    "sidebar_filters",
    "render_course_summary",
    "render_results_table",
    "render_download_button",
]
