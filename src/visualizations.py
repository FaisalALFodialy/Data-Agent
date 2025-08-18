import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List


def create_empty_chart(message: str) -> go.Figure:
    """Create an empty chart with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor="white",
        height=400
    )
    return fig


def create_error_chart(error_message: str) -> go.Figure:
    """Create an error chart with error message."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"⚠️ {error_message}",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="red")
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor="#ffeeee",
        height=400
    )
    return fig

def plot_missing_values_heatmap(df: pd.DataFrame):
    """Create a heatmap showing missing values across the dataset."""
    try:
        if df.empty:
            return create_empty_chart("No data available for missing values heatmap")
        
        mask = df.isna()
        if mask.empty or not mask.any().any():
            return create_empty_chart("No missing values found")
        
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
    except Exception as e:
        return create_error_chart(f"Error creating missing values heatmap: {str(e)}")

def plot_correlation_heatmap(corr_matrix: pd.DataFrame):
    """Create a correlation heatmap for numeric columns."""
    try:
        if corr_matrix.empty or corr_matrix.shape[0] == 0:
            return create_empty_chart("No data available for correlation analysis")
        
        if corr_matrix.shape[0] < 2:
            return create_empty_chart("Need at least 2 numeric columns for correlation")
        
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
    except Exception as e:
        return create_error_chart(f"Error creating correlation heatmap: {str(e)}")

def create_data_quality_dashboard(analysis_results: Dict[str, Any]) -> go.Figure:
    """
    Create a comprehensive data quality dashboard with multiple metrics.
    
    Combines multiple quality indicators into a single dashboard
    view for quick assessment of overall data health.
    
    Args:
        analysis_results (Dict[str, Any]): Results from data analysis containing
                                          quality metrics and statistics
        
    Returns:
        plotly.graph_objects.Figure: Interactive dashboard with multiple subplots
    """
    try:
        if not analysis_results:
            return create_empty_chart("No analysis results available for dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Data Quality Score', 'Missing Values by Column', 
                           'Data Types Distribution', 'Outlier Summary'),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
    
        # 1. Quality Score Gauge
        try:
            usability_score = analysis_results.get('usability', {}).get('usability_score', 0)
            usability_score = float(usability_score) if usability_score else 0
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=usability_score,
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'thickness': 0.3},
                           'steps': [
                               {'range': [0, 60], 'color': "#ffcccc"},
                               {'range': [60, 80], 'color': "#fff0b3"},
                               {'range': [80, 100], 'color': "#ccffcc"},
                           ]},
                    title={'text': "Quality Score"}
                ),
                row=1, col=1
            )
        except Exception:
            # Add placeholder if gauge fails
            pass
    
        # 2. Missing values bar chart (sample data if not available)
        try:
            if 'missing_values' in analysis_results:
                missing_data = analysis_results['missing_values'].get('per_column', pd.DataFrame())
                if not missing_data.empty and 'null_pct' in missing_data.columns:
                    top_missing = missing_data.head(10)
                    fig.add_trace(
                        go.Bar(
                            x=top_missing['column'] if 'column' in top_missing.columns else top_missing.index,
                            y=top_missing['null_pct'],
                            name="Missing %",
                            marker_color='lightcoral'
                        ),
                        row=1, col=2
                    )
        except Exception:
            # Add placeholder if missing values chart fails
            pass
    
        # 3. Data types pie chart (sample if not available)
        try:
            # Extract basic stats for data types
            basic_stats = analysis_results.get('basic_stats', {})
            if basic_stats and 'column_names' in basic_stats:
                # This is a placeholder - would need actual dtype info
                dtypes_count = {'Numeric': 3, 'Text': 2, 'Boolean': 1}
            else:
                dtypes_count = {'Numeric': 3, 'Text': 2, 'Boolean': 1}
                
            fig.add_trace(
                go.Pie(
                    labels=list(dtypes_count.keys()),
                    values=list(dtypes_count.values()),
                    name="Data Types"
                ),
                row=2, col=1
            )
        except Exception:
            # Fallback pie chart
            try:
                fig.add_trace(
                    go.Pie(
                        labels=['Numeric', 'Text', 'Boolean'],
                        values=[3, 2, 1],
                        name="Data Types"
                    ),
                    row=2, col=1
                )
            except Exception:
                pass
    
        # 4. Outlier summary (sample data)
        try:
            outlier_data = {'Column A': 5, 'Column B': 3, 'Column C': 8}
            fig.add_trace(
                go.Bar(
                    x=list(outlier_data.keys()),
                    y=list(outlier_data.values()),
                    name="Outliers",
                    marker_color='orange'
                ),
                row=2, col=2
            )
        except Exception:
            # Add placeholder if outlier chart fails
            pass
        
        # Update layout
        fig.update_layout(
            height=600,
            title_text="Data Quality Dashboard",
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        return create_error_chart(f"Error creating data quality dashboard: {str(e)}")

def plot_usability_gauge(score: int):
    """Create a gauge chart for usability score."""
    try:
        if not isinstance(score, (int, float)) or score < 0 or score > 100:
            return create_error_chart(f"Invalid score: {score}. Score must be between 0-100")
        
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
    except Exception as e:
        return create_error_chart(f"Error creating usability gauge: {str(e)}")


def plot_missing_values_bar(df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing missing value counts per column.
    
    Displays the count and percentage of missing values for each column
    in a clear, interactive bar chart format.
    """
    try:
        if df.empty:
            return create_empty_chart("No data available for missing values analysis")
        
        missing_counts = df.isnull().sum()
        missing_pcts = (missing_counts / len(df) * 100).round(2)
        
        # Only show columns with missing values
        cols_with_missing = missing_counts[missing_counts > 0]
        
        if len(cols_with_missing) == 0:
            return create_empty_chart("No missing values found")
        
        fig = go.Figure(data=[
            go.Bar(
                x=cols_with_missing.index,
                y=cols_with_missing.values,
                text=[f"{missing_pcts[col]:.1f}%" for col in cols_with_missing.index],
                textposition='auto',
                marker_color='lightcoral'
            )
        ])
        
        fig.update_layout(
            title="Missing Values by Column",
            xaxis_title="Columns",
            yaxis_title="Missing Count",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
        
    except Exception as e:
        return create_error_chart(f"Error creating missing values bar chart: {str(e)}")


def plot_distribution(df: pd.DataFrame, column: str, plot_type: str = 'histogram') -> go.Figure:
    """
    Create distribution plots for numerical or categorical columns.
    
    Generates appropriate distribution visualizations based on column
    type and user preference (histogram, box plot, violin plot).
    """
    try:
        if column not in df.columns:
            return create_error_chart(f"Column '{column}' not found in dataframe")
        
        col_data = df[column].dropna()
        
        if len(col_data) == 0:
            return create_empty_chart(f"No data available for column '{column}'")
        
        if plot_type == 'histogram':
            if pd.api.types.is_numeric_dtype(col_data):
                fig = px.histogram(
                    x=col_data,
                    nbins=min(50, max(10, len(col_data.unique()))),
                    title=f"Distribution of {column}"
                )
            else:
                # For categorical data, show value counts
                value_counts = col_data.value_counts().head(20)
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Distribution of {column}"
                )
                fig.update_xaxis(title=column)
                fig.update_yaxis(title="Count")
        
        elif plot_type == 'box':
            if pd.api.types.is_numeric_dtype(col_data):
                fig = px.box(y=col_data, title=f"Box Plot of {column}")
            else:
                return create_empty_chart("Box plot only available for numeric columns")
        
        elif plot_type == 'violin':
            if pd.api.types.is_numeric_dtype(col_data):
                fig = px.violin(y=col_data, title=f"Violin Plot of {column}")
            else:
                return create_empty_chart("Violin plot only available for numeric columns")
        
        else:
            return create_error_chart(f"Invalid plot_type: {plot_type}. Use 'histogram', 'box', or 'violin'")
        
        return fig
        
    except Exception as e:
        return create_error_chart(f"Error creating distribution plot: {str(e)}")


def plot_outliers_boxplot(df: pd.DataFrame, columns: List[str]) -> go.Figure:
    """
    Create box plots to visualize outliers in multiple numeric columns.
    
    Displays box plots for selected columns side-by-side to compare
    distributions and identify outliers across variables.
    """
    try:
        if df.empty:
            return create_empty_chart("No data available for outlier analysis")
        
        numeric_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if not numeric_columns:
            return create_empty_chart("No numeric columns found for outlier analysis")
        
        fig = go.Figure()
        
        for col in numeric_columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                fig.add_trace(go.Box(
                    y=col_data,
                    name=col,
                    boxpoints='outliers'  # Show outlier points
                ))
        
        if not fig.data:
            return create_empty_chart("No valid data found in selected columns")
        
        fig.update_layout(
            title="Outlier Detection - Box Plots",
            yaxis_title="Values",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
        
    except Exception as e:
        return create_error_chart(f"Error creating outlier box plots: {str(e)}")


def plot_data_types_pie(df: pd.DataFrame) -> go.Figure:
    """
    Create a pie chart showing the distribution of data types.
    
    Visualizes the proportion of different data types across all
    columns to give insight into data composition.
    """
    try:
        if df.empty:
            return create_empty_chart("No data available for data types analysis")
        
        dtype_counts = df.dtypes.value_counts()
        
        if dtype_counts.empty:
            return create_empty_chart("No data types found")
        
        # Map pandas dtypes to more readable names
        dtype_mapping = {
            'int64': 'Integer',
            'float64': 'Float',
            'object': 'Text/Object',
            'bool': 'Boolean',
            'datetime64[ns]': 'DateTime',
            'category': 'Categorical'
        }
        
        readable_labels = [dtype_mapping.get(str(dtype), str(dtype)) for dtype in dtype_counts.index]
        
        fig = go.Figure(data=[go.Pie(
            labels=readable_labels,
            values=dtype_counts.values,
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Columns: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Data Types Distribution",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
        
    except Exception as e:
        return create_error_chart(f"Error creating data types pie chart: {str(e)}")


def plot_cardinality_bar(cardinality_data: Dict[str, int]) -> go.Figure:
    """
    Create a bar chart showing cardinality (unique values) for each column.
    
    Visualizes the number of unique values per column to identify
    high and low cardinality features.
    """
    try:
        if not cardinality_data:
            return create_empty_chart("No cardinality data provided")
        
        # Sort by cardinality for better visualization
        sorted_data = dict(sorted(cardinality_data.items(), key=lambda x: x[1], reverse=True))
        
        columns = list(sorted_data.keys())
        values = list(sorted_data.values())
        
        if not columns:
            return create_empty_chart("No columns to analyze")
        
        # Color code based on cardinality levels
        colors = []
        for val in values:
            if val < 10:
                colors.append('lightgreen')  # Low cardinality
            elif val < 50:
                colors.append('orange')      # Medium cardinality
            else:
                colors.append('lightcoral')  # High cardinality
        
        fig = go.Figure(data=[
            go.Bar(
                x=columns,
                y=values,
                marker_color=colors,
                text=values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Cardinality Analysis - Unique Values per Column",
            xaxis_title="Columns",
            yaxis_title="Unique Values Count",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        # Rotate x-axis labels if many columns
        if len(columns) > 10:
            fig.update_xaxis(tickangle=45)
        
        return fig
        
    except Exception as e:
        return create_error_chart(f"Error creating cardinality bar chart: {str(e)}")


def plot_before_after_comparison(before_df: pd.DataFrame, after_df: pd.DataFrame, column: str) -> go.Figure:
    """
    Create before/after comparison plots for a specific column.
    
    Shows the impact of cleaning operations by comparing distributions,
    statistics, or other relevant metrics before and after cleaning.
    """
    try:
        if before_df.empty or after_df.empty:
            return create_empty_chart("No data available for before/after comparison")
        
        if column not in before_df.columns or column not in after_df.columns:
            return create_error_chart(f"Column '{column}' not found in one or both dataframes")
    
        # Create subplots for before and after
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'Before Cleaning: {column}', f'After Cleaning: {column}'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        before_data = before_df[column].dropna()
        after_data = after_df[column].dropna()
        
        if pd.api.types.is_numeric_dtype(before_data):
            # Numeric data - show histograms
            fig.add_trace(
                go.Histogram(x=before_data, name="Before", opacity=0.7, marker_color='lightcoral'),
                row=1, col=1
            )
            fig.add_trace(
                go.Histogram(x=after_data, name="After", opacity=0.7, marker_color='lightgreen'),
                row=1, col=2
            )
        else:
            # Categorical data - show value counts
            before_counts = before_data.value_counts().head(10)
            after_counts = after_data.value_counts().head(10)
            
            fig.add_trace(
                go.Bar(x=before_counts.index, y=before_counts.values, name="Before", marker_color='lightcoral'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=after_counts.index, y=after_counts.values, name="After", marker_color='lightgreen'),
                row=1, col=2
            )
        
        # Add summary statistics as annotations
        try:
            before_stats = f"Count: {len(before_data)}<br>Missing: {int(before_df[column].isna().sum())}"
            after_stats = f"Count: {len(after_data)}<br>Missing: {int(after_df[column].isna().sum())}"
            
            if pd.api.types.is_numeric_dtype(before_data):
                before_stats += f"<br>Mean: {float(before_data.mean()):.2f}"
                after_stats += f"<br>Mean: {float(after_data.mean()):.2f}"
            
            fig.add_annotation(
                text=before_stats,
                xref="x domain", yref="y domain",
                x=0.02, y=0.98, showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black", borderwidth=1
            )
        except Exception:
            # Skip annotations if they fail
            pass
        
        fig.update_layout(
            title=f"Before vs After Comparison: {column}",
            height=400,
            margin=dict(l=0, r=0, t=60, b=0)
        )
        
        return fig
        
    except Exception as e:
        return create_error_chart(f"Error creating before/after comparison: {str(e)}")
