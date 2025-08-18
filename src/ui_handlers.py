"""
UI Handlers Module

All UI rendering functions separated from main.py following Single Responsibility Principle.
Handles all Streamlit UI components, tabs, and user interactions.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any
import math
import openai
import json
import logging
import datetime

# Import custom modules
from .data_analyzer import create_analysis_report, calculate_usability_score
from .visualizations import (
    plot_missing_values_heatmap, 
    plot_correlation_heatmap,
    create_data_quality_dashboard,
    plot_usability_gauge,
    plot_missing_values_bar,
    plot_data_types_pie,
    plot_outliers_boxplot
)
from .utils import (
    calculate_memory_usage,
    format_bytes,
    create_download_link,
    get_numeric_columns,
    safe_dataframe,
    get_current_dataframe,
    validate_session_state,
    ensure_session_state_integrity
)
from .ui_components import (
    render_cleaning_recommendations,
    render_missing_values_cleaning,
    render_outliers_cleaning,
    render_duplicates_cleaning,
    render_text_cleaning,
    render_data_types_cleaning,
    render_cleaning_pipeline_management,
    render_analysis_report_download,
    render_cleaning_report_download,
    render_pipeline_config_download,
    render_complete_package_export
)
from .config import (
    UPLOAD_FILE_TYPES, 
    MAX_FILE_SIZE,
    MISSING_VALUE_THRESHOLD,
    OUTLIER_IQR_MULTIPLIER
)
from .app_core import reset_application_state, load_sample_data

# Configure logging
logger = logging.getLogger(__name__)


def json_safe(obj, *, max_rows=200):
    """Convert objects to JSON-safe format."""
    # primitives
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    # pandas
    if isinstance(obj, pd.DataFrame):
        df = obj.head(max_rows).copy()
        for c in df.columns:
            df[c] = df[c].apply(lambda v: json_safe(v, max_rows=max_rows))
        return df.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return [json_safe(v, max_rows=max_rows) for v in obj.head(max_rows).tolist()]

    # numpy
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [json_safe(v, max_rows=max_rows) for v in obj.tolist()]

    # datetime-like
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)

    # mappings / sequences
    if isinstance(obj, dict):
        return {str(k): json_safe(v, max_rows=max_rows) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v, max_rows=max_rows) for v in obj]

    # fallback (e.g., Streamlit DeltaGenerator)
    return str(obj)


def send_json_to_gpt(json_data: dict) -> str:
    """Send analyzed JSON data to GPT and return the response."""
    try:
        safe_payload = json_safe(json_data, max_rows=200)
        user_msg = "Here is the JSON data:\n" + json.dumps(safe_payload, indent=2, ensure_ascii=False)

        resp = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
            {"role": "system", "content": """You are given an EDA report as JSON: {EDA_JSON}.
Write 6–13 bullet points. OUTPUT RULES:
- Output ONLY bullets; each line must start with "- " and be ONE short sentence.
- No paragraphs, headings, tables, code blocks, or extra text before/after the bullets.
- Use only facts found in the JSON; do not invent numbers; round to 1–2 decimals.
- Mention when available: dataset shape/memory; overall missing % and rows-with-missing %; top 2–3 columns by null % with brief imputation/drop hint; dtype mix (numeric/categorical/datetime); cardinality (flag ID-like unique_ratio ≥0.95 and low-card ≤10 with encoding tips); outliers/skewness and suggested transforms; duplicates; 2–4 strongest absolute correlations and any multicollinearity risk (|corr|>0.8); any quality/usability score and drivers; concrete next-step preprocessing actions.
- If some sections are absent in the JSON, omit them silently.
- Keep column lists to at most 3 names per line; keep language concise and actionable."""},
            {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message["content"]
    except Exception as e:
        return f"Error communicating with GPT: {e}"


def render_sidebar() -> Dict[str, Any]:
    """Render enhanced sidebar with file upload and settings."""
    # Byan branding
    try:
        st.sidebar.image("src/Byan.png", use_container_width=True)
    except:
        pass  # Image file not found, continue without logo
    
    st.sidebar.header("📁 File Upload")
    
    # File upload with enhanced validation
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=UPLOAD_FILE_TYPES,
        help=f"Maximum file size: {format_bytes(MAX_FILE_SIZE)}\n\nSupported formats: {', '.join(UPLOAD_FILE_TYPES)}",
        key="sidebar_file_uploader"
    )
    
    # Upload tips
    with st.sidebar.expander("Upload Tips"):
        st.markdown("""
        **For best results:**
        - Use UTF-8 encoding
        - Include column headers
        - Avoid special characters in filenames
        - Ensure consistent data types per column
        """)
    
    st.sidebar.markdown("---")

    st.sidebar.header("Settings")
    
    # Advanced mode toggle
    advanced_mode = st.sidebar.checkbox(
        "Advanced Mode", 
        value=st.session_state.get('advanced_mode', False),
        help="Enable advanced features and detailed controls",
        key="sidebar_advanced_mode"
    )
    st.session_state.advanced_mode = advanced_mode
    
    # Auto-analysis settings
    auto_analyze = st.sidebar.checkbox(
        "Auto-analyze on upload", 
        value=True,
        help="Automatically run analysis when a file is uploaded",
        key="sidebar_auto_analyze"
    )
    
    # Data quality thresholds
    with st.sidebar.expander("Quality Thresholds"):
        max_missing_threshold = st.slider(
            "Missing Value Alert (%)",
            min_value=0,
            max_value=100,
            value=int(MISSING_VALUE_THRESHOLD * 100),
            help="Flag columns with missing values above this threshold",
            key="sidebar_missing_threshold"
        )
        
        outlier_sensitivity = st.slider(
            "Outlier Sensitivity",
            min_value=1.0,
            max_value=3.0,
            value=OUTLIER_IQR_MULTIPLIER,
            step=0.1,
            help="IQR multiplier for outlier detection (higher = less sensitive)",
            key="sidebar_outlier_sensitivity"
        )
    
    # Performance settings
    if advanced_mode:
        with st.sidebar.expander("Performance"):
            chunk_size = st.selectbox(
                "Processing chunk size",
                [1000, 5000, 10000, 50000],
                index=2,
                help="Larger chunks = faster processing but more memory usage",
                key="sidebar_chunk_size"
            )
            
            enable_caching = st.checkbox(
                "Enable result caching",
                value=True,
                help="Cache analysis results for faster repeated operations",
                key="sidebar_enable_caching"
            )
    else:
        chunk_size = 10000
        enable_caching = True
    
    st.sidebar.markdown("---")
    
    # Data summary (if data is loaded)
    if st.session_state.current_df is not None:
        st.sidebar.header("Data Summary")
        df = get_current_dataframe()
        
        st.sidebar.metric("Rows", f"{int(df.shape[0]):,}")
        st.sidebar.metric("Columns", f"{int(df.shape[1]):,}")
        
        memory_info = calculate_memory_usage(df)
        st.sidebar.metric("Memory", memory_info['total'])
        
        # Quick quality indicator
        if st.session_state.usability_score:
            score = int(st.session_state.usability_score['overall_score'])
            color = "Good" if score >= 80 else "Fair" if score >= 60 else "Poor"
            st.sidebar.metric("Quality Score", f"{color}: {score}/100")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Help & Tips")
    
    with st.sidebar.expander("Quick Start"):
        st.markdown("""
        1. **Upload** your CSV file
        2. **Review** the analysis results
        3. **Clean** your data with suggested operations
        4. **Export** the cleaned dataset
        """)
    
    with st.sidebar.expander("Troubleshooting"):
        st.markdown("""
        **Common issues:**
        - File too large: Try smaller chunks
        - Encoding errors: Check file encoding
        - Memory issues: Enable advanced mode
        - Performance slow: Reduce chunk size
        """)
    
    # Reset button
    if st.sidebar.button("Reset Application"):
        reset_application_state()
        st.rerun()

    return {
        'uploaded_file': uploaded_file,
        'advanced_mode': advanced_mode,
        'auto_analyze': auto_analyze,
        'max_missing_threshold': max_missing_threshold / 100,
        'outlier_sensitivity': outlier_sensitivity,
        'chunk_size': chunk_size,
        'enable_caching': enable_caching
    }


def render_analysis_tab() -> None:
    """Render enhanced data analysis tab with comprehensive metrics using session state."""
    try:
        # Ensure session state integrity
        ensure_session_state_integrity()
        
        # Get current dataframe from session state
        df = get_current_dataframe()
        st.header("Data Analysis Dashboard")
        
        # Enhanced metrics display
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", f"{df.shape[1]:,}")
        with col3:
            memory_info = calculate_memory_usage(df)
            st.metric("Memory Usage", memory_info['total'])
        with col4:
            missing_pct = float((df.isnull().sum().sum() / max(df.size, 1)) * 100)
            color = "normal" if missing_pct < 5 else "inverse"
            st.metric("Missing Data", f"{missing_pct:.1f}%", delta_color=color)
        with col5:
            duplicates = int(df.duplicated().sum())
            st.metric("Duplicates", f"{duplicates:,}")

        # Run analysis with progress tracking
        if st.session_state.analysis_results is None:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Analyzing data..."):
                try:
                    status_text.text("Running comprehensive analysis...")
                    progress_bar.progress(25)
                    
                    st.session_state.analysis_results = create_analysis_report(df)
                    progress_bar.progress(75)
                    
                    status_text.text("Calculating quality scores...")
                    progress_bar.progress(100)
                    
                    status_text.text("Analysis complete!")
                    progress_bar.empty()
                    status_text.empty()
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    logger.error(f"Analysis error: {str(e)}")
                    return

        analysis_results = st.session_state.analysis_results

        # Calculate usability score
        if st.session_state.usability_score is None:
            st.session_state.usability_score = calculate_usability_score(analysis_results)

        # Enhanced usability score display
        st.subheader("Data Quality Assessment")
        
        score_col1, score_col2, score_col3 = st.columns([1, 2, 1])
        
        with score_col1:
            score = int(st.session_state.usability_score['overall_score'])
            st.metric("Overall Score", f"{score}/100")
            
            # Quality indicators
            if score >= 80:
                st.success("Excellent data quality!")
                quality_color = "success"
            elif score >= 60:
                st.warning("Good quality, room for improvement")
                quality_color = "warning"
            else:
                st.error("Poor quality - cleaning recommended")
                quality_color = "error"
        
        with score_col2:
            try:
                fig_gauge = plot_usability_gauge(score)
                st.plotly_chart(fig_gauge, use_container_width=True, key="analysis_tab_quality_gauge")
            except Exception as e:
                st.error(f"Failed to render quality gauge: {str(e)}")
        
        with score_col3:
            # Quality breakdown
            components = st.session_state.usability_score.get('components', {})
            st.markdown("**Quality Factors:**")
            st.markdown(f"Missing: {float(components.get('missing_pct', 0)):.1f}%")
            st.markdown(f"Duplicates: {float(components.get('duplicate_pct', 0)):.1f}%")
            st.markdown(f"Outliers: {float(components.get('avg_outlier_pct', 0)):.1f}%")

        # Data Quality Dashboard
        st.subheader("Comprehensive Analysis")
        
        try:
            fig_dashboard = create_data_quality_dashboard(analysis_results)
            st.plotly_chart(fig_dashboard, use_container_width=True, key="analysis_tab_dashboard")
        except Exception as e:
            st.warning(f"Could not render dashboard: {str(e)}")

        # Detailed analysis sections
        analysis_tabs = st.tabs([
            "Missing Values", "Data Types", "Statistics", "Outliers", 
            "Correlations", "Raw Results"
        ])
        
        with analysis_tabs[0]:
            render_missing_values_analysis(df, analysis_results)
            
        with analysis_tabs[1]:
            render_data_types_analysis(df, analysis_results)
            
        with analysis_tabs[2]:
            render_statistics_analysis(df, analysis_results)
            
        with analysis_tabs[3]:
            render_outliers_analysis(df, analysis_results)
            
        with analysis_tabs[4]:
            render_correlation_analysis(df, analysis_results)
            
        with analysis_tabs[5]:
            render_raw_analysis_results(analysis_results)
            
    except Exception as e:
        st.error(f"Error in analysis tab: {str(e)}")
        logger.error(f"Analysis tab error: {str(e)}")
        if st.session_state.advanced_mode:
            st.exception(e)


def render_cleaning_tab() -> None:
    """Render enhanced data cleaning tab with intelligent operations using session state."""
    try:
        # Ensure session state integrity
        ensure_session_state_integrity()
        
        # Validate session state before proceeding
        if not validate_session_state():
            st.error("❌ No valid data loaded. Please upload a file first.")
            return
        st.header("Data Cleaning Workshop")
        
        # Cleaning strategy recommendation
        render_cleaning_recommendations()
        
        # Cleaning operations tabs with state persistence
        cleaning_tab_names = [
            "Missing Values", "Outliers", "Duplicates", 
            "Text Cleaning", "Data Types", "Pipeline"
        ]
        
        # Use session state to track active cleaning sub-tab
        selected_cleaning_tab = st.selectbox(
            "Select Cleaning Operation:",
            options=cleaning_tab_names,
            index=st.session_state.active_cleaning_subtab,
            key="cleaning_tab_selector"
        )
        
        # Update session state when cleaning tab changes
        if selected_cleaning_tab != cleaning_tab_names[st.session_state.active_cleaning_subtab]:
            st.session_state.active_cleaning_subtab = cleaning_tab_names.index(selected_cleaning_tab)
        
        # Add some styling separation
        st.markdown("---")
        
        # Render content based on selected cleaning tab
        if selected_cleaning_tab == "Missing Values":
            render_missing_values_cleaning()
        elif selected_cleaning_tab == "Outliers":
            render_outliers_cleaning()
        elif selected_cleaning_tab == "Duplicates":
            render_duplicates_cleaning()
        elif selected_cleaning_tab == "Text Cleaning":
            render_text_cleaning()
        elif selected_cleaning_tab == "Data Types":
            render_data_types_cleaning()
        elif selected_cleaning_tab == "Pipeline":
            render_cleaning_pipeline_management()
            
    except Exception as e:
        st.error(f"Error in cleaning tab: {str(e)}")
        logger.error(f"Cleaning tab error: {str(e)}")
        if st.session_state.advanced_mode:
            st.exception(e)


def render_export_tab() -> None:
    """Render enhanced export tab with multiple format options using session state."""
    try:
        # Ensure session state integrity
        ensure_session_state_integrity()
        
        # Get current dataframe from session state
        df = get_current_dataframe()
        st.header("Export & Download")
        
        # Export summary
        st.subheader("Export Summary")
        if st.session_state.original_df is not None:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original Rows", f"{st.session_state.original_df.shape[0]:,}")
            with col2:
                st.metric("Cleaned Rows", f"{df.shape[0]:,}")
            with col3:
                rows_removed = int(st.session_state.original_df.shape[0] - df.shape[0])
                delta_color = "inverse" if rows_removed > 0 else "normal"
                st.metric("Rows Removed", f"{rows_removed:,}", delta_color=delta_color)
            with col4:
                memory_saved = int(calculate_memory_usage(st.session_state.original_df)['bytes'] - calculate_memory_usage(df)['bytes'])
                st.metric("Memory Saved", format_bytes(memory_saved) if memory_saved > 0 else "0 B")

        # Main export options
        export_col1, export_col2 = st.columns([2, 1])
        
        with export_col1:
            st.subheader("Download Cleaned Data")
            
            # File naming
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"cleaned_data_{timestamp}.csv"
            filename = st.text_input("Filename", value=default_filename, key="export_filename")
            
            # Quick download using modern Streamlit download button
            csv_data = create_download_link(df, filename)
            st.download_button(
                label="📥 Download Cleaned Data",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                use_container_width=True
            )
            st.info(f"File will contain {df.shape[0]:,} rows and {df.shape[1]:,} columns")
            
            # Preview data before export
            if st.checkbox("Preview data before export", key="export_preview_checkbox"):
                preview_rows = st.slider("Rows to preview", 5, min(100, len(df)), 10, key="export_preview_rows")
                safe_dataframe(df.head(preview_rows), use_container_width=True)
        
        with export_col2:
            st.subheader("Export Settings")
            
            include_index = st.checkbox("Include row index", value=False, key="export_include_index")
            
            date_format = st.selectbox(
                "Date format",
                ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"],
                help="Format for datetime columns",
                key="export_date_format"
            )
            
            encoding = st.selectbox(
                "File encoding",
                ["utf-8", "utf-8-sig", "latin-1", "cp1252"],
                help="Character encoding for the output file",
                key="export_encoding"
            )
            
            separator = st.selectbox(
                "CSV separator",
                [",", ";", "\t", "|"],
                format_func=lambda x: {',': 'Comma (,)', ';': 'Semicolon (;)', '\t': 'Tab', '|': 'Pipe (|)'}[x],
                key="export_separator"
            )

        # Reports section
        st.subheader("Download Reports")
        
        report_col1, report_col2, report_col3 = st.columns(3)
        
        with report_col1:
            if st.button("Analysis Report", use_container_width=True, key="export_analysis_report"):
                render_analysis_report_download()
                
        with report_col2:
            if st.button("Cleaning Report", use_container_width=True, key="export_cleaning_report"):
                render_cleaning_report_download()
                
        with report_col3:
            if st.button("Pipeline Config", use_container_width=True, key="export_pipeline_config"):
                render_pipeline_config_download()

        # Bulk export options
        if st.session_state.advanced_mode:
            st.subheader("Bulk Export")
            
            if st.button("Export Complete Package", key="export_complete_package"):
                render_complete_package_export()
                
    except Exception as e:
        st.error(f"Error in export tab: {str(e)}")
        logger.error(f"Export tab error: {str(e)}")
        if st.session_state.advanced_mode:
            st.exception(e)


def render_welcome_screen() -> None:
    """Render welcome screen when no data is loaded."""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h2>Welcome to the Byan Data Analysis & Cleaning Tool</h2>
        <p style="font-size: 1.2rem; color: #666;">Your comprehensive solution for data quality assessment and cleaning with AI insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ### Getting Started
        
        1. **Upload** your CSV file using the sidebar
        2. **Analyze** your data quality automatically
        3. **Clean** your data with intelligent suggestions
        4. **Export** your improved dataset
        
        ### Key Features
        
        - **Smart Quality Scoring**: Get instant data quality assessment
        - **GPT-4 Insights**: AI-powered analysis and recommendations
        - **Comprehensive Analysis**: Missing values, outliers, data types
        - **Intelligent Cleaning**: Automated suggestions and operations
        - **Rich Visualizations**: Interactive charts and dashboards
        - **Multiple Export Options**: CSV, reports, and configurations
        
        ### Supported Data
        
        - **File formats**: CSV, TSV
        - **Encodings**: UTF-8, Latin-1, CP1252
        - **Size limit**: Up to 200MB
        - **Columns**: Unlimited
        """)
        
        # Sample data option
        if st.button("Try with Sample Data", use_container_width=True, key="welcome_sample_data"):
            load_sample_data()


def render_data_preview() -> None:
    """Render collapsible data preview section using session state."""
    try:
        df = get_current_dataframe()
    except ValueError:
        st.warning("No data available for preview")
        return
    with st.expander("Data Preview", expanded=False):
        preview_col1, preview_col2 = st.columns([3, 1])
        
        with preview_col1:
            preview_rows = st.slider("Rows to preview", 5, min(100, len(df)), 20, key="data_preview_rows")
            safe_dataframe(df.head(preview_rows), use_container_width=True)
            
        with preview_col2:
            st.markdown("**Quick Info:**")
            st.markdown(f"Shape: {int(df.shape[0]):,} × {int(df.shape[1]):,}")
            st.markdown(f"Memory: {calculate_memory_usage(df)['total']}")
            st.markdown(f"Missing: {int(df.isnull().sum().sum()):,} values")
            st.markdown(f"Duplicates: {int(df.duplicated().sum()):,} rows")


# Analysis helper functions
def render_missing_values_analysis(df: pd.DataFrame, analysis_results: Dict) -> None:
    """Render missing values analysis section."""
    missing_info = analysis_results.get('missing_values', {})
    
    if missing_info.get('total_missing', 0) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                fig_heatmap = plot_missing_values_heatmap(df)
                st.plotly_chart(fig_heatmap, use_container_width=True, key="analysis_tab_missing_heatmap")
            except Exception as e:
                st.warning(f"Could not render missing values heatmap: {str(e)}")
        
        with col2:
            try:
                fig_bar = plot_missing_values_bar(df)
                st.plotly_chart(fig_bar, use_container_width=True, key="analysis_tab_missing_bar")
            except Exception as e:
                st.warning(f"Could not render missing values bar chart: {str(e)}")
        
        # Missing values table
        st.subheader("Missing Values by Column")
        missing_df = pd.DataFrame(missing_info.get('per_column', {})).T
        safe_dataframe(missing_df, use_container_width=True)
    else:
        st.success("No missing values found!")


def render_data_types_analysis(df: pd.DataFrame, analysis_results: Dict) -> None:
    """Render data types analysis section."""
    st.subheader("Data Types Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            fig_types = plot_data_types_pie(df)
            st.plotly_chart(fig_types, use_container_width=True, key="analysis_tab_datatypes")
        except Exception as e:
            st.warning(f"Could not render data types chart: {str(e)}")
    
    with col2:
        data_types_info = analysis_results.get('data_types', pd.DataFrame())
        if not data_types_info.empty:
            safe_dataframe(data_types_info, use_container_width=True)
        else:
            st.info("No data type information available")


def render_statistics_analysis(df: pd.DataFrame, analysis_results: Dict) -> None:
    """Render statistical analysis section."""
    st.subheader("Statistical Summary")
    
    # Standard deviation analysis
    std_info = analysis_results.get("standard_deviation", {})
    per_col = std_info.get("per_column", {})
    
    if per_col:
        col1, col2 = st.columns(2)
        
        with col1:
            std_df = pd.DataFrame({
                "Column": list(per_col.keys()), 
                "Standard Deviation": list(per_col.values())
            }).sort_values("Standard Deviation", ascending=False, ignore_index=True)
            
            safe_dataframe(std_df, use_container_width=True)
        
        with col2:
            try:
                import plotly.express as px
                fig_std = px.bar(
                    std_df, x="Column", y="Standard Deviation",
                    title="Standard Deviation per Numeric Column"
                )
                fig_std.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_std, use_container_width=True, key="analysis_tab_statistics")
            except Exception as e:
                st.info(f"Chart not available: {str(e)}")
    else:
        st.info("No numeric columns available for standard deviation analysis")
    
    # Column statistics
    st.subheader("Column Statistics")
    col_stats = analysis_results.get('column_statistics', pd.DataFrame())
    if not col_stats.empty:
        # Clean up the statistics display
        display_stats = col_stats.copy()
        
        # Format example values
        if 'example_values' in display_stats.columns:
            display_stats['example_values'] = display_stats['example_values'].apply(
                lambda v: ", ".join(map(str, v[:3])) + ("..." if len(v) > 3 else "") 
                if isinstance(v, (list, tuple)) else str(v)
            )
        
        # Format datetime columns
        for col in ["min", "max"]:
            if col in display_stats.columns:
                display_stats[col] = display_stats[col].apply(
                    lambda x: x.isoformat()[:19] if hasattr(x, "isoformat") else x
                )
        
        safe_dataframe(display_stats, use_container_width=True)
    else:
        st.info("No statistical information available")


def render_outliers_analysis(df: pd.DataFrame, analysis_results: Dict) -> None:
    """Render outliers analysis section."""
    st.subheader("Outliers Detection")
    
    numeric_columns = get_numeric_columns(df)
    
    if numeric_columns:
        selected_cols = st.multiselect(
            "Select columns for outlier analysis",
            numeric_columns,
            default=numeric_columns[:min(3, len(numeric_columns))],
            key="analysis_outlier_columns"
        )
        
        if selected_cols:
            try:
                fig = plot_outliers_boxplot(df, selected_cols)
                st.plotly_chart(fig, use_container_width=True, key="analysis_tab_outliers")
            except Exception as e:
                st.warning(f"Could not render outlier chart: {str(e)}")
        else:
            st.info("Select columns to analyze outliers")
    else:
        st.info("No numeric columns available for outlier analysis")


def render_correlation_analysis(df: pd.DataFrame, analysis_results: Dict) -> None:
    """Render correlation analysis section."""
    st.subheader("Correlation Matrix")
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        try:
            corr_matrix = numeric_df.corr()
            fig_corr = plot_correlation_heatmap(corr_matrix)
            st.plotly_chart(fig_corr, use_container_width=True, key="analysis_tab_correlation")
            
            # Show correlation insights
            strong_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corrs.append({
                            'Pair': f"{corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}",
                            'Correlation': f"{corr_val:.3f}",
                            'Strength': 'Strong Positive' if corr_val > 0.7 else 'Strong Negative'
                        })
            
            if strong_corrs:
                st.subheader("Strong Correlations")
                safe_dataframe(pd.DataFrame(strong_corrs), use_container_width=True)
            else:
                st.info("No strong correlations (|r| > 0.7) found")
                
        except Exception as e:
            st.warning(f"Could not render correlation analysis: {str(e)}")
    else:
        st.info("Need at least 2 numeric columns for correlation analysis")


def render_raw_analysis_results(analysis_results: Dict) -> None:
    """Render raw analysis results in JSON format with GPT analysis."""
    st.subheader("Raw Analysis Data")
    
    # GPT Analysis Section
    st.subheader("🤖 AI-Powered Insights")
    
    # Initialize GPT response in session state
    if 'gpt_analysis_response' not in st.session_state:
        st.session_state.gpt_analysis_response = None
    
    if st.button("Get GPT-4 Analysis", key="raw_analysis_gpt"):
        with st.spinner("Analyzing data with GPT-4..."):
            try:
                gpt_response = send_json_to_gpt(analysis_results)
                st.session_state.gpt_analysis_response = gpt_response
            except Exception as e:
                st.session_state.gpt_analysis_response = f"Error: {str(e)}"
    
    # Display stored GPT response if available
    if st.session_state.gpt_analysis_response:
        st.markdown("**GPT-4 Analysis:**")
        if st.session_state.gpt_analysis_response.startswith("Error:"):
            st.error(st.session_state.gpt_analysis_response)
            st.info("Make sure OpenAI API key is configured correctly.")
        else:
            st.write(st.session_state.gpt_analysis_response)
        
        # Add button to clear the analysis
        if st.button("Clear Analysis", key="clear_gpt_analysis"):
            st.session_state.gpt_analysis_response = None
    
    # Raw JSON data
    if st.session_state.advanced_mode:
        with st.expander("Show raw analysis JSON"):
            st.json(analysis_results)
    else:
        with st.expander("Show raw analysis data"):
            st.json(analysis_results)
