import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional


def create_score_histogram(
    scores: pd.Series,
    title: str = "Score Distribution",
    bins: int = 10,
    color: str = "#636EFA"
) -> go.Figure:
    """Create a histogram of scores."""
    fig = px.histogram(
        x=scores.dropna(),
        nbins=bins,
        title=title,
        labels={"x": "Score", "y": "Count"},
        color_discrete_sequence=[color]
    )
    fig.update_layout(showlegend=False)
    return fig


def create_score_comparison_bar(
    scores_dict: dict,
    title: str = "Score Comparison"
) -> go.Figure:
    """Create a bar chart comparing multiple score types."""
    names = list(scores_dict.keys())
    values = [scores_dict[k].mean() if hasattr(scores_dict[k], 'mean') else scores_dict[k] for k in names]

    fig = go.Figure(data=[
        go.Bar(x=names, y=values, marker_color="#636EFA")
    ])
    fig.update_layout(
        title=title,
        xaxis_title="Metric",
        yaxis_title="Average Score",
        yaxis_range=[0, max(values) * 1.1] if values else [0, 1]
    )
    return fig


def create_score_boxplot(
    df: pd.DataFrame,
    score_columns: list,
    title: str = "Score Distributions"
) -> go.Figure:
    """Create a boxplot showing distributions of multiple scores."""
    fig = go.Figure()

    for col in score_columns:
        if col in df.columns:
            fig.add_trace(go.Box(
                y=df[col].dropna(),
                name=col,
                boxmean=True
            ))

    fig.update_layout(
        title=title,
        yaxis_title="Score",
        showlegend=False
    )
    return fig


def create_correlation_heatmap(
    df: pd.DataFrame,
    columns: list,
    title: str = "Score Correlations"
) -> go.Figure:
    """Create a correlation heatmap for score columns."""
    # Filter to only specified columns that exist
    valid_cols = [c for c in columns if c in df.columns]
    if len(valid_cols) < 2:
        return None

    corr_matrix = df[valid_cols].corr()

    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=valid_cols,
        y=valid_cols,
        color_continuous_scale="RdBu",
        title=title,
        zmin=-1,
        zmax=1
    )
    return fig


def create_score_over_samples(
    scores: pd.Series,
    title: str = "Scores Over Samples",
    window: int = 10
) -> go.Figure:
    """Create a line chart showing scores across samples with rolling average."""
    fig = go.Figure()

    # Individual scores
    fig.add_trace(go.Scatter(
        x=list(range(len(scores))),
        y=scores.values,
        mode="markers",
        name="Individual Scores",
        marker=dict(size=5, opacity=0.5)
    ))

    # Rolling average
    if len(scores) >= window:
        rolling_avg = scores.rolling(window=window, center=True).mean()
        fig.add_trace(go.Scatter(
            x=list(range(len(rolling_avg))),
            y=rolling_avg.values,
            mode="lines",
            name=f"Rolling Avg (n={window})",
            line=dict(width=2)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Sample Index",
        yaxis_title="Score"
    )
    return fig


def create_length_vs_score_scatter(
    df: pd.DataFrame,
    length_col: str,
    score_col: str,
    title: str = "Output Length vs Score"
) -> go.Figure:
    """Create a scatter plot of output length vs score."""
    if length_col not in df.columns or score_col not in df.columns:
        return None

    fig = px.scatter(
        df,
        x=length_col,
        y=score_col,
        title=title,
        labels={length_col: "Output Length", score_col: "Score"},
        trendline="ols"
    )
    return fig


def create_summary_metrics_card(metrics: dict) -> str:
    """Generate HTML for a summary metrics display."""
    cards_html = '<div style="display: flex; gap: 20px; flex-wrap: wrap;">'

    for name, value in metrics.items():
        if isinstance(value, float):
            display_value = f"{value:.3f}"
        else:
            display_value = str(value)

        cards_html += f'''
        <div style="background: #f0f2f6; padding: 15px 25px; border-radius: 8px; text-align: center;">
            <div style="font-size: 24px; font-weight: bold; color: #262730;">{display_value}</div>
            <div style="font-size: 14px; color: #6b7280;">{name}</div>
        </div>
        '''

    cards_html += '</div>'
    return cards_html
