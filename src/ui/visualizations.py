"""Plotly visualisations for similarity results."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def similarity_heatmap(results: pd.DataFrame) -> go.Figure:
    """Create a heatmap of weighted similarity scores."""

    if results.empty:
        return go.Figure()

    pivot = results.pivot_table(
        index="reference_course_id",
        columns="target_course_id",
        values="weighted_score",
        aggfunc="mean",
        fill_value=0.0,
    )
    figure = px.imshow(
        pivot,
        color_continuous_scale="Viridis",
        aspect="auto",
        labels=dict(x="Target course", y="Reference course", color="Score"),
    )
    figure.update_layout(title="Similarity heatmap")
    return figure


def component_score_bar(result_row: pd.Series) -> go.Figure:
    """Plot the component scores for a single comparison."""

    components = {
        "Semantic": result_row.get("semantic_similarity", 0.0),
        "Structural": result_row.get("structural_similarity", 0.0),
        "Learning outcomes": result_row.get("learning_outcome_similarity", 0.0),
    }
    figure = px.bar(
        x=list(components.keys()),
        y=list(components.values()),
        labels={"x": "Similarity component", "y": "Score"},
        range_y=[0, 1],
    )
    figure.update_layout(title="Component similarity breakdown")
    return figure


__all__ = ["similarity_heatmap", "component_score_bar"]
