import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import io
import json
import openai
import os
import json, math, datetime


# Import custom modules
from src.data_analyzer import create_analysis_report, calculate_usability_score
from src.data_cleaner import apply_cleaning_pipeline, preview_cleaning_operation
from src.visualizations import (
    plot_missing_values_heatmap, 
    plot_correlation_heatmap,
    create_data_quality_dashboard,
    plot_usability_gauge
)
from src.utils import (
    load_csv_file, 
    save_csv_file, 
    validate_dataframe,
    create_download_link,
    generate_cleaning_report,
    format_bytes,
    calculate_memory_usage
)
from src.config import (
    UPLOAD_FILE_TYPES, 
    MAX_FILE_SIZE,
    MISSING_VALUE_THRESHOLD,
    MISSING_NUMERIC_METHODS,
    MISSING_CATEGORICAL_METHODS,
    TEXT_CLEANING_OPERATIONS
)

def json_safe(obj, *, max_rows=200):
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

# ---- GPT call
def send_json_to_gpt(json_data: dict) -> str:
    """
    Send analyzed JSON data to GPT and return the response.
    Works with openai==0.x style.
    """
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
        # old SDK returns dict-like object
        return resp.choices[0].message["content"]
    except Exception as e:
        return f"Error communicating with GPT: {e}"

def main() -> None:
    setup_page_config()

    # ---- init session state ----
    for key, default in [
        ('uploaded_file', None),
        ('original_df', None),
        ('current_df', None),
        ('analysis_results', None),
        ('cleaning_pipeline', []),
        ('cleaning_history', []),
        ('usability_score', None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    sidebar_data = render_sidebar()
    uploaded_df = handle_file_upload(sidebar_data['uploaded_file'])

    if uploaded_df is not None:
        tab1, tab2, tab3, tab4 = st.tabs(["Data Analysis", "Data Cleaning", "Export", "QR Code"])

        with tab1:
            render_analysis_tab(uploaded_df)

        with tab2:
            render_cleaning_tab(uploaded_df)

        with tab3:
            render_export_tab(uploaded_df)

        with tab4:
            st.title("📲 Access Byan via QR Code")
            st.image("Byan-qr.png", use_container_width=True)

    else:
        st.info("Please upload a CSV file to begin analysis")


def setup_page_config() -> None:
    st.set_page_config(
        page_title="Byan Data Analysis & Cleaning Tool",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; text-align: center; margin: 0.5rem 0 1.5rem; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="main-header">Byan Data Analysis & Cleaning Tool</div>', unsafe_allow_html=True)


def render_sidebar() -> Dict[str, Any]:
    st.sidebar.image("Byan.png", use_container_width=True)
    st.sidebar.header("📁 File Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=UPLOAD_FILE_TYPES,
        help=f"Maximum file size: {format_bytes(MAX_FILE_SIZE)}"
    )
    st.sidebar.markdown("---")

    st.sidebar.header("⚙️ Settings")
    show_advanced = st.sidebar.checkbox("Show Advanced Options", value=False)
    auto_analyze = st.sidebar.checkbox("Auto-analyze on upload", value=True)
    max_missing_threshold = st.sidebar.slider(
        "Missing Value Threshold (%)",
        min_value=0,
        max_value=100,
        value=int(MISSING_VALUE_THRESHOLD * 100),
        help="Columns with missing values above this threshold will be flagged"
    )
    st.sidebar.markdown("---")
    st.sidebar.header("❓ Help")
    st.sidebar.info("Upload a CSV, then explore analysis and cleaning suggestions.")

    return {
        'uploaded_file': uploaded_file,
        'show_advanced': show_advanced,
        'auto_analyze': auto_analyze,
        'max_missing_threshold': max_missing_threshold / 100
    }


def render_analysis_tab(df: pd.DataFrame) -> None:
    st.header("📊 Data Analysis Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Rows", f"{df.shape[0]:,}")
    with col2: st.metric("Columns", f"{df.shape[1]:,}")
    with col3:
        memory_info = calculate_memory_usage(df)
        st.metric("Memory Usage", memory_info['total'])
    with col4:
        missing_pct = (df.isnull().sum().sum() / max(df.size, 1)) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")

    if st.session_state.analysis_results is None:
        with st.spinner("Analyzing data..."):
            st.session_state.analysis_results = create_analysis_report(df)

    analysis_results = st.session_state.analysis_results

    if st.session_state.usability_score is None:
        st.session_state.usability_score = calculate_usability_score(analysis_results)

    st.subheader("🎯 Data Usability Score")
    c1, c2 = st.columns([1, 2])
    with c1:
        score = st.session_state.usability_score['overall_score']
        st.metric("Overall Score", f"{score}/100")
        if score >= 80:
            st.success("Excellent data quality!")
        elif score >= 60:
            st.warning("Good data quality with room for improvement")
        else:
            st.error("Poor data quality - cleaning recommended")
    with c2:
        fig_gauge = plot_usability_gauge(score)
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.subheader("📋 Detailed Analysis")

    with st.expander("Missing Values Analysis", expanded=False):
        if analysis_results['missing_values']['total_missing'] > 0:
            fig_missing = plot_missing_values_heatmap(df)
            st.plotly_chart(fig_missing, use_container_width=True)
            st.dataframe(analysis_results['missing_values']['per_column'], use_container_width=True)
        else:
            st.success("No missing values found!")

    # 🔹 Standard Deviation Section (right after Missing Values Analysis)
    with st.expander("Standard Deviation (Numeric Columns)", expanded=False):
        # Pull from the analysis report
        std_info = analysis_results.get("standard_deviation", {})
        per_col = std_info.get("per_column", {})

        if per_col:
            std_df = pd.DataFrame(
                {"Column": list(per_col.keys()), "Standard Deviation": list(per_col.values())}
            ).sort_values("Standard Deviation", ascending=False, ignore_index=True)

            st.dataframe(std_df, use_container_width=True)

            # Optional bar chart (requires plotly installed)
            try:
                import plotly.express as px
                fig_std = px.bar(
                    std_df, x="Column", y="Standard Deviation",
                    title="Standard Deviation per Numeric Column", text="Standard Deviation"
                )
                st.plotly_chart(fig_std, use_container_width=True)
            except Exception:
                st.info("Plotly not installed — showing table only.")
        else:
            st.info("No numeric columns available for standard deviation analysis.")      
 
    with st.expander("Data Types and Statistics", expanded=False):
        st.dataframe(analysis_results['data_types'], use_container_width=True)
        colstats = analysis_results['column_statistics'].copy()
        if 'example_values' in colstats.columns:
            colstats['example_values'] = colstats['example_values'].apply(
                lambda v: ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
            )

        for c in ("min", "max"):
            if c in colstats.columns:
                colstats[c] = colstats[c].apply(
                    lambda x: x.isoformat() if hasattr(x, "isoformat") else x
                )

        st.dataframe(colstats, use_container_width=True)
        st.dataframe(analysis_results['column_statistics'], use_container_width=True)

    with st.expander("Correlation Analysis",expanded=False):
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr(numeric_only=True)
            fig_corr = plot_correlation_heatmap(corr_matrix)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation analysis")

    with st.expander("Full Analysis JSON",expanded=False):
        st.json(analysis_results)

    if st.button("Analyse JSON to GPT"):
        st.info("Sending data to GPT...")
        gpt_response = send_json_to_gpt(analysis_results)
        st.subheader("🤖 GPT Response")
        st.write(gpt_response)

def render_cleaning_tab(df: pd.DataFrame) -> None:
    st.header("Data Cleaning Workshop")
    st.subheader("Available Cleaning Operations")

    with st.expander("Handle Missing Values", expanded=True):
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            st.write(f"Columns with missing values: {', '.join(missing_cols)}")
            selected_col = st.selectbox("Select column to clean", missing_cols)
            if selected_col:
                if pd.api.types.is_numeric_dtype(df[selected_col]):
                    method = st.selectbox("Cleaning method", MISSING_NUMERIC_METHODS)
                else:
                    method = st.selectbox("Cleaning method", MISSING_CATEGORICAL_METHODS)
                if st.button(f"Preview cleaning for {selected_col}"):
                    preview = preview_cleaning_operation(df, {"type": "missing", "col": selected_col, "method": method})
                    st.dataframe(preview.head(100), use_container_width=True)
                if st.button(f"Apply cleaning for {selected_col}"):
                    new_df, op = apply_cleaning_pipeline(df, [{"type": "missing", "col": selected_col, "method": method}])
                    st.session_state.current_df = new_df
                    st.session_state.cleaning_pipeline.append(op)
                    st.success("Applied!")
        else:
            st.success("No missing values found!")

    with st.expander("Handle Outliers"):
        st.info("Basic outlier handling can be added here (IQR/Z-Score).")

    with st.expander("Handle Duplicates"):
        dup_count = df.duplicated().sum()
        st.write(f"Found {dup_count} duplicate rows")
        if dup_count > 0 and st.button("Remove Duplicates"):
            new_df, op = apply_cleaning_pipeline(df, [{"type": "duplicates", "keep": "first"}])
            st.session_state.current_df = new_df
            st.session_state.cleaning_pipeline.append(op)
            st.success("Duplicates removed")

    with st.expander("Optimize Data Types"):
        if st.button("Optimize dtypes"):
            new_df, op = apply_cleaning_pipeline(df, [{"type": "optimize_dtypes"}])
            st.session_state.current_df = new_df
            st.session_state.cleaning_pipeline.append(op)
            st.success("Optimized dtypes")

    st.subheader("📝 Cleaning Pipeline")
    if st.session_state.cleaning_pipeline:
        for i, op in enumerate(st.session_state.cleaning_pipeline, 1):
            st.write(f"{i}. {op}")
    else:
        st.info("No cleaning operations applied yet")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear Pipeline"):
            st.session_state.cleaning_pipeline = []
            st.success("Pipeline cleared!")
    with c2:
        if st.button("Apply All Changes"):
            new_df, _ = apply_cleaning_pipeline(st.session_state.original_df, st.session_state.cleaning_pipeline)
            st.session_state.current_df = new_df
            st.success("All changes applied to the original dataset")


def render_export_tab(df: pd.DataFrame) -> None:
    st.header("Export Options")
    st.subheader("Download Cleaned Data")

    c1, c2 = st.columns(2)
    with c1:
        filename = st.text_input("Filename", value="cleaned_data.csv")
        st.markdown(create_download_link(df, filename), unsafe_allow_html=True)
        st.info(f"File will contain {df.shape[0]:,} rows and {df.shape[1]:,} columns")
    with c2:
        st.subheader("📋 Export Options")
        st.checkbox("Include row index", value=False, key="export_include_index")
        st.selectbox("Date format", ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"], key="export_date_fmt")
        st.selectbox("File encoding", ["utf-8", "latin-1", "cp1252"], key="export_encoding")

    st.subheader("Download Reports")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Analysis Report"):
            st.info("You can download the full analysis JSON above.")
    with c2:
        if st.button("🧹 Cleaning Report"):
            if st.session_state.cleaning_pipeline:
                _ = generate_cleaning_report(
                    st.session_state.original_df, df, st.session_state.cleaning_pipeline
                )
                st.success("Cleaning report generated!")
            else:
                st.warning("No cleaning operations to report")
    with c3:
        if st.button("Pipeline Config"):
            st.info("Export of pipeline config (JSON) can be added.")

    st.subheader("Export Summary")
    if st.session_state.original_df is not None:
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Original Rows", f"{st.session_state.original_df.shape[0]:,}")
        with c2: st.metric("Cleaned Rows", f"{df.shape[0]:,}")
        with c3:
            rows_removed = st.session_state.original_df.shape[0] - df.shape[0]
            st.metric("Rows Removed", f"{rows_removed:,}")


def handle_file_upload(uploaded_file) -> Optional[pd.DataFrame]:
    """Read and validate CSV; cache in session_state."""
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"File too large. Maximum size allowed: {format_bytes(MAX_FILE_SIZE)}")
            return None

        df = load_csv_file(uploaded_file)  # tries common encodings
        if df is None:
            st.error("Could not read the CSV file. Please check the file format.")
            return None

        if not validate_dataframe(df):
            st.error("Invalid dataframe structure")
            return None

        if st.session_state.original_df is None:
            st.session_state.original_df = df.copy()
        st.session_state.current_df = df.copy()
        st.session_state.analysis_results = None
        st.session_state.usability_score = None

        st.success("File uploaded successfully!")
        c1, c2, c3 = st.columns(3)
        with c1: st.info(f"**Rows:** {df.shape[0]:,}")
        with c2: st.info(f"**Columns:** {df.shape[1]:,}")
        with c3:
            memory_info = calculate_memory_usage(df)
            st.info(f"**Size:** {memory_info['total']}")
        return df

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

if __name__ == "__main__":
    main()


