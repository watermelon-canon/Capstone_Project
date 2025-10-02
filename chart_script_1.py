import plotly.graph_objects as go
import plotly.express as px
import json

# Data from the provided JSON
data = {
    "methods": [
        {"approach": "Semantic-only", "precision_at_5": 0.72, "ndcg_at_10": 0.68, "coverage": 0.89, "fairness": 0.73},
        {"approach": "Structural-only", "precision_at_5": 0.65, "ndcg_at_10": 0.62, "coverage": 0.94, "fairness": 0.81},
        {"approach": "Hybrid-basic", "precision_at_5": 0.79, "ndcg_at_10": 0.74, "coverage": 0.91, "fairness": 0.84},
        {"approach": "Hybrid-advanced", "precision_at_5": 0.87, "ndcg_at_10": 0.83, "coverage": 0.93, "fairness": 0.89}
    ]
}

# Extract data for the chart
approaches = [method["approach"] for method in data["methods"]]
precision_values = [method["precision_at_5"] for method in data["methods"]]
ndcg_values = [method["ndcg_at_10"] for method in data["methods"]]
coverage_values = [method["coverage"] for method in data["methods"]]
fairness_values = [method["fairness"] for method in data["methods"]]

# Create grouped bar chart
fig = go.Figure()

# Brand colors for the four metrics
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F']

# Add bars for each metric
fig.add_trace(go.Bar(
    name='Precision@5',
    x=approaches,
    y=precision_values,
    marker_color=colors[0],
    yaxis='y1'
))

fig.add_trace(go.Bar(
    name='nDCG@10',
    x=approaches,
    y=ndcg_values,
    marker_color=colors[1],
    yaxis='y1'
))

fig.add_trace(go.Bar(
    name='Coverage',
    x=approaches,
    y=coverage_values,
    marker_color=colors[2],
    yaxis='y1'
))

fig.add_trace(go.Bar(
    name='Fairness',
    x=approaches,
    y=fairness_values,
    marker_color=colors[3],
    yaxis='y1'
))

# Update layout
fig.update_layout(
    title='Performance Comparison',
    xaxis_title='Method',
    yaxis_title='Score',
    barmode='group',
    yaxis=dict(range=[0.6, 0.95]),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.05,
        xanchor='center',
        x=0.5
    )
)

# Update traces for better display
fig.update_traces(cliponaxis=False)

# Save as both PNG and SVG
fig.write_image("performance_chart.png")
fig.write_image("performance_chart.svg", format="svg")

fig.show()