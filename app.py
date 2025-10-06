"""Streamlit application entry point for the Course Equivalency Mapping System."""
from __future__ import annotations

import json
from typing import Iterable, List

import pandas as pd
import streamlit as st

from src.config.settings import SimilarityWeights, load_settings
from src.models.course_processor import CourseProcessor, CourseRecord
from src.models.similarity_engine import SemanticSimilarityEngine
from src.ui.components import (
    render_course_summary,
    render_download_button,
    render_results_table,
    sidebar_filters,
)
from src.ui.visualizations import component_score_bar, similarity_heatmap
from src.utils.data_loader import load_courses
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@st.cache_data(show_spinner=False)
def load_sample_data() -> List[CourseRecord]:
    """Load and process the bundled sample dataset."""

    processor = CourseProcessor()
    courses = load_courses("data/sample_courses.json")
    return processor.process_courses(courses)


def parse_uploaded_file(file) -> List[CourseRecord]:
    """Parse and process an uploaded file from the Streamlit interface."""

    if file is None:
        return []

    processor = CourseProcessor()
    filename = file.name.lower()
    try:
        if filename.endswith(".json"):
            payload = json.load(file)
            records = payload.get("courses", payload) if isinstance(payload, dict) else payload
            if not isinstance(records, list):
                st.error("JSON file must contain a list of courses or a 'courses' key.")
                return []
        elif filename.endswith(".csv"):
            frame = pd.read_csv(file)
            records = frame.to_dict(orient="records")
        else:
            st.error("Unsupported file type. Please upload CSV or JSON.")
            return []
        return processor.process_courses(records)
    except Exception as exc:  # pragma: no cover - user facing feedback
        st.error(f"Failed to process uploaded file: {exc}")
        LOGGER.exception("Failed to process uploaded file")
        return []


def filter_by_institution(courses: Iterable[CourseRecord], institution: str) -> List[CourseRecord]:
    return [course for course in courses if course.institution == institution]


def main() -> None:
    st.set_page_config(
        page_title="Course Equivalency Mapping System",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Course Equivalency Mapping System")
    st.write(
        "Compare academic courses across institutions using semantic, structural, "
        "and learning outcome similarity measures. Upload your catalog or use "
        "the provided sample data to explore equivalency insights."
    )

    sample_courses = load_sample_data()
    uploaded_file = st.sidebar.file_uploader(
        "Upload additional course catalog (CSV or JSON)",
        type=["csv", "json"],
    )
    uploaded_courses = parse_uploaded_file(uploaded_file)

    all_courses = sample_courses + uploaded_courses
    if not all_courses:
        st.warning("No courses available. Please upload a dataset.")
        st.stop()

    sidebar_options = sidebar_filters(all_courses)

    weights = load_settings().similarity_weights
    semantic_weight = st.sidebar.slider(
        "Semantic similarity weight",
        min_value=0.0,
        max_value=1.0,
        value=float(weights.semantic),
        step=0.05,
    )
    structural_weight = st.sidebar.slider(
        "Structural similarity weight",
        min_value=0.0,
        max_value=1.0,
        value=float(weights.structural),
        step=0.05,
    )
    outcome_weight = st.sidebar.slider(
        "Learning outcome similarity weight",
        min_value=0.0,
        max_value=1.0,
        value=float(weights.learning_outcomes),
        step=0.05,
    )

    reference_courses = filter_by_institution(
        all_courses, sidebar_options["reference_institution"]
    )
    target_courses = filter_by_institution(
        all_courses, sidebar_options["target_institution"]
    )

    if not reference_courses or not target_courses:
        st.info("Adjust the institution filters to include available courses.")
        st.stop()

    reference_lookup = {f"{course.course_id} – {course.title}": course for course in reference_courses}
    selected_reference_label = st.selectbox(
        "Select reference course",
        options=list(reference_lookup.keys()),
    )
    selected_reference = reference_lookup[selected_reference_label]

    st.markdown("---")
    st.header("Reference course overview")
    render_course_summary(selected_reference)

    engine = SemanticSimilarityEngine()
    engine.settings.similarity_weights = SimilarityWeights(
        semantic=semantic_weight,
        structural=structural_weight,
        learning_outcomes=outcome_weight,
    )

    results = engine.rank_courses(
        reference_courses=[selected_reference],
        target_courses=target_courses,
        top_k=sidebar_options["top_k"],
    )

    st.markdown("---")
    st.header("Similarity results")
    render_results_table(results)
    render_download_button(results)

    if not results.empty:
        st.plotly_chart(similarity_heatmap(results), width="stretch")
        top_result = results.iloc[0]
        st.subheader(
            f"Detailed breakdown: {top_result['reference_course_id']} vs {top_result['target_course_id']}"
        )
        st.plotly_chart(component_score_bar(top_result), width="stretch")
        st.info(top_result["explanation"])


if __name__ == "__main__":
    main()
