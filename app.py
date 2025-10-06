"""Streamlit UI for browsing course equivalency data."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st

DATA_CANDIDATES: Tuple[Path, ...] = (
    Path("data/course_equivalencies.csv"),
    Path("data/sample_courses_combined_catalog.csv"),
    Path("data/sample_courses_primary_catalog.csv"),
    Path("data/sample_courses_alternative_catalog.csv"),
)

FILTER_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "school": ("school", "school_name", "institution", "campus"),
    "department": ("department", "dept", "division", "domain"),
    "course": ("course", "course_id", "course_code", "course_number", "course_name", "title"),
}


def _resolve_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _prepare_options(df: pd.DataFrame, column: str) -> List[str]:
    return sorted(df[column].dropna().astype(str).unique())


@st.cache_data(show_spinner=True)
def load_course_data() -> Tuple[pd.DataFrame, Path]:
    errors: List[str] = []
    for candidate in DATA_CANDIDATES:
        if not candidate.exists():
            errors.append(f"{candidate.name} not found")
            continue
        try:
            data = pd.read_csv(candidate)
        except Exception as exc:  # pragma: no cover - defensive logging for unexpected formats
            errors.append(f"{candidate.name} could not be read ({exc})")
            continue
        if data.empty:
            errors.append(f"{candidate.name} contains no rows")
            continue
        data.columns = [str(col).strip() for col in data.columns]
        return data, candidate
    detail = "; ".join(errors) if errors else "No data files located"
    raise FileNotFoundError(detail)


def render_filters(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], List[str]]:
    with st.sidebar:
        st.header("Filters")
        st.caption("Choose course equivalency filters. Unavailable filters are disabled until the data provides the relevant columns.")

        school_col = _resolve_column(df, FILTER_CANDIDATES["school"])
        department_col = _resolve_column(df, FILTER_CANDIDATES["department"])
        course_col = _resolve_column(df, FILTER_CANDIDATES["course"])

        school_options = ["All"] + _prepare_options(df, school_col) if school_col else ["All"]
        selected_school = st.selectbox(
            "School",
            options=school_options,
            index=0,
            help="Limit results to a single school or institution.",
            disabled=school_col is None,
        )

        if school_col is None:
            st.caption("Add a column such as 'school' or 'institution' to enable this filter.")

        department_options = ["All"] + _prepare_options(df, department_col) if department_col else ["All"]
        selected_department = st.selectbox(
            "Department",
            options=department_options,
            index=0,
            help="Filter on the originating department or subject area.",
            disabled=department_col is None,
        )

        if department_col is None:
            st.caption("Add a column such as 'department', 'dept', or 'domain' to enable this filter.")

        course_options = _prepare_options(df, course_col) if course_col else []
        selected_courses = st.multiselect(
            "Course",
            options=course_options,
            help="Select one or more courses to narrow the results.",
            disabled=course_col is None,
        )

        if course_col is None:
            st.caption("Add a column such as 'course_id' or 'title' to enable this filter.")

    return (
        selected_school if school_col else None,
        selected_department if department_col else None,
        selected_courses,
    )


def apply_filters(
    df: pd.DataFrame,
    school_filter: Optional[str],
    department_filter: Optional[str],
    course_filter: List[str],
) -> pd.DataFrame:
    filtered = df.copy()

    school_col = _resolve_column(df, FILTER_CANDIDATES["school"])
    department_col = _resolve_column(df, FILTER_CANDIDATES["department"])
    course_col = _resolve_column(df, FILTER_CANDIDATES["course"])

    if school_col and school_filter and school_filter != "All":
        filtered = filtered[filtered[school_col].astype(str) == school_filter]

    if department_col and department_filter and department_filter != "All":
        filtered = filtered[filtered[department_col].astype(str) == department_filter]

    if course_col and course_filter:
        filtered = filtered[filtered[course_col].astype(str).isin(course_filter)]

    return filtered


def render_app() -> None:
    st.set_page_config(page_title="Course Equivalency Explorer", layout="wide")
    st.title("Course Equivalency Explorer")
    st.write(
        "Browse course equivalencies, filter by institution metadata, and identify matching courses across catalogs."
    )

    try:
        dataframe, source_path = load_course_data()
    except FileNotFoundError as exc:
        st.error("Unable to load course equivalency data.")
        st.info(
            "Upload a CSV to `data/course_equivalencies.csv` (or use one of the sample files) and restart the app."
        )
        st.code(str(exc))
        return
    except ValueError as exc:
        st.error("The course data could not be parsed.")
        st.code(str(exc))
        return

    school_filter, department_filter, course_filter = render_filters(dataframe)
    filtered_df = apply_filters(dataframe, school_filter, department_filter, course_filter)

    st.subheader("Results")
    total_count = len(dataframe)
    filtered_count = len(filtered_df)
    if filtered_count:
        st.success(f"Displaying {filtered_count} of {total_count} courses from {source_path.name}.")
        st.dataframe(filtered_df.reset_index(drop=True))
    else:
        st.warning("No course equivalencies match the selected filters.")
        st.caption("Adjust the filters or provide additional data to see results.")

    missing_filters: List[str] = []
    for key, candidates in FILTER_CANDIDATES.items():
        if _resolve_column(dataframe, candidates) is None:
            missing_filters.append(key)

    if missing_filters:
        st.info(
            "The data source is missing columns for: "
            + ", ".join(sorted(missing_filters))
            + ". Add the relevant columns to unlock these filters."
        )

    st.caption(f"Data source: {source_path}")


if __name__ == "__main__":
    render_app()
