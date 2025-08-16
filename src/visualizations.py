import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_missing_values_heatmap(df: pd.DataFrame):
    mask = df.isna()
    if mask.empty:
        return go.Figure()
    z = mask.astype(int).values
    fig = px.imshow(
        z,
        labels=dict(x="Columns", y="Rows", color="Missing"),
        x=list(df.columns),
        y=list(range(len(df))),
        aspect="auto",
        color_continuous_scale="Reds",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), title="Missing Values Heatmap")
    return fig

def plot_correlation_heatmap(corr_matrix: pd.DataFrame):
    if corr_matrix.shape[0] == 0:
        return go.Figure()
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        aspect="auto",
        title="Correlation Heatmap"
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    return fig

def create_data_quality_dashboard():
    # Optional: composite layout if you want multiple charts at once
    return go.Figure()

def plot_usability_gauge(score: int):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        gauge={'axis': {'range': [0,100]},
               'bar': {'thickness': 0.3},
               'steps': [
                   {'range': [0,60], 'color': "#ffcccc"},
                   {'range': [60,80], 'color': "#fff0b3"},
                   {'range': [80,100], 'color': "#ccffcc"},
               ]},
        title={'text': "Usability"},
        domain={'x':[0,1], 'y':[0,1]}
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig
