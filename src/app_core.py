"""
App Core Module

Core application logic separated from main.py following Single Responsibility Principle.
Handles application initialization, session state, configuration, and file operations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import warnings
from typing import Optional

# Import custom modules
from .utils import (
    load_csv_file, 
    validate_dataframe,
    format_bytes,
    calculate_memory_usage
)
from .config import MAX_FILE_SIZE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


def initialize_session_state() -> None:
    """Initialize all session state variables with defaults."""
    defaults = {
        'uploaded_file': None,
        'original_df': None,
        'current_df': None,
        'analysis_results': None,
        'cleaning_pipeline': [],
        'cleaning_history': [],
        'usability_score': None,
        'selected_columns': [],
        'preview_mode': False,
        'error_state': False,
        'last_operation': None,
        'operation_history': [],
        'auto_save_enabled': True,
        'advanced_mode': False,
        # Tab state tracking
        'active_main_tab': 0,  # Track main tab (Analysis=0, Cleaning=1, Visualizations=2, Export=3)
        'active_cleaning_subtab': 0  # Track cleaning sub-tab (Missing Values=0, Outliers=1, etc.)
    }
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def setup_page_config() -> None:
    """Configure Streamlit page settings and custom CSS."""
    st.set_page_config(
        page_title="Byan Data Analysis & Cleaning Tool",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo/help',
            'Report a bug': 'https://github.com/your-repo/issues',
            'About': """# Byan Data Analysis & Cleaning Tool
            
            A production-ready tool for comprehensive CSV data analysis and cleaning with AI insights.
            
            **Features:**
            - Intelligent data quality scoring
            - Interactive cleaning operations
            - Advanced visualizations
            - GPT-4 powered insights
            - Export capabilities
                        """
        }
    )
    
    # Enhanced CSS styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin: 0.5rem 0 1.5rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .warning-card {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .error-card {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 0.3rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header with enhanced styling
    st.markdown('<div class="main-header">Byan Data Analysis & Cleaning Tool</div>', unsafe_allow_html=True)
    
    # Add version and info
    st.markdown(
        '<div style="text-align: center; color: #666; margin-bottom: 2rem;">'
        'v2.0.0 | Advanced Analytics'
        '</div>', 
        unsafe_allow_html=True
    )


def handle_file_upload(uploaded_file) -> Optional[pd.DataFrame]:
    """Enhanced file upload handling with comprehensive validation and error recovery."""
    if uploaded_file is None:
        return None
        
    try:
        # File size validation
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(
                f"File too large: {format_bytes(uploaded_file.size)} > {format_bytes(MAX_FILE_SIZE)}\n\n"
                f"**Suggestions:**\n"
                f"• Split the file into smaller chunks\n"
                f"• Use a more efficient format (Parquet, HDF5)\n"
                f"• Remove unnecessary columns before upload"
            )
            return None

        # Show upload progress
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Try to load the CSV with multiple fallbacks
            df = load_csv_file(uploaded_file)
            
        if df is None:
            st.error(
                "Could not read the CSV file.\n\n"
                "**Common issues:**\n"
                "• File encoding (try UTF-8, UTF-8-BOM, or Latin-1)\n"
                "• Wrong delimiter (comma, semicolon, tab)\n"
                "• Corrupted file\n"
                "• Not a valid CSV format\n\n"
                "**Try:**\n"
                "• Opening the file in a text editor to check format\n"
                "• Re-exporting from Excel as CSV (UTF-8)\n"
                "• Using a different file"
            )
            return None

        # Validate dataframe structure
        if not validate_dataframe(df):
            st.error(
                "Invalid dataframe structure detected.\n\n"
                f"**File details:**\n"
                f"• Rows: {df.shape[0] if hasattr(df, 'shape') else 'Unknown'}\n"
                f"• Columns: {df.shape[1] if hasattr(df, 'shape') else 'Unknown'}\n\n"
                "**Possible issues:**\n"
                "• No data in file\n"
                "• Invalid column names\n"
                "• Unsupported data types"
            )
            return None

        # Store in session state - only reset if it's a new file
        is_new_file = (st.session_state.original_df is None or 
                      uploaded_file != st.session_state.uploaded_file or
                      st.session_state.get('uploaded_file_name') != uploaded_file.name)
        
        if is_new_file:
            st.session_state.original_df = df.copy()
            st.session_state.uploaded_file = uploaded_file
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.current_df = df.copy()
            
            # Only clear pipeline and history for truly new files
            st.session_state.analysis_results = None
            st.session_state.usability_score = None
            st.session_state.cleaning_pipeline = []
            st.session_state.operation_history = []
        else:
            # Same file re-uploaded - preserve current dataframe if it exists
            if st.session_state.current_df is not None:
                # Keep the existing current_df (which may have cleaning applied)
                pass
            else:
                st.session_state.current_df = df.copy()

        # Success message with file details
        st.success(f"{uploaded_file.name} uploaded successfully!")
        
        # File information display
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        
        with info_col1:
            st.info(f"**Rows:** {df.shape[0]:,}")
        with info_col2:
            st.info(f"**Columns:** {df.shape[1]:,}")
        with info_col3:
            memory_info = calculate_memory_usage(df)
            st.info(f"**Size:** {memory_info['total']}")
        with info_col4:
            file_size = format_bytes(uploaded_file.size)
            st.info(f"**File:** {file_size}")
            
        # Show potential memory optimization
        optimization_potential = memory_info.get('optimization_potential', {})
        if optimization_potential.get('percentage', 0) > 10:
            st.info(
                f"**Tip:** Your data could be optimized to save "
                f"{optimization_potential['formatted']} ({optimization_potential['percentage']:.1f}%) in memory. "
                f"Use the 'Optimize Data Types' feature in the cleaning tab."
            )
            
        return df

    except Exception as e:
        error_msg = str(e)
        st.error(
            f"Error processing file: {error_msg}\n\n"
            f"**Troubleshooting:**\n"
            f"• Check if the file is a valid CSV\n"
            f"• Ensure the file is not corrupted\n"
            f"• Try a smaller file first\n"
            f"• Contact support if the issue persists"
        )
        logger.error(f"File upload error for {uploaded_file.name if uploaded_file else 'unknown'}: {error_msg}")
        
        if st.session_state.advanced_mode:
            st.exception(e)
            
        return None


def reset_application_state() -> None:
    """Reset all session state variables to defaults."""
    keys_to_reset = [
        'uploaded_file', 'original_df', 'current_df', 'analysis_results',
        'cleaning_pipeline', 'cleaning_history', 'usability_score',
        'selected_columns', 'preview_mode', 'error_state', 'last_operation',
        'operation_history', 'auto_save_enabled', 'advanced_mode'
    ]
    
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success("Application state reset successfully!")


def display_error_recovery() -> None:
    """Display error recovery options."""
    st.error("The application encountered an error. Please choose a recovery option:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Reset Application"):
            reset_application_state()
            st.session_state.error_state = False
            st.rerun()
    
    with col2:
        if st.button("Try Different File"):
            st.session_state.uploaded_file = None
            st.session_state.error_state = False
            st.rerun()
    
    with col3:
        if st.button("Advanced Mode"):
            st.session_state.advanced_mode = True
            st.session_state.error_state = False
            st.rerun()


def handle_critical_error(error: Exception) -> None:
    """Handle critical application errors with recovery options."""
    logger.error(f"Critical error: {str(error)}")
    
    st.error(
        f"**Critical Error Detected**\n\n"
        f"Error: {str(error)}\n\n"
        f"The application will attempt to recover. If issues persist, "
        f"please refresh the page or contact support."
    )
    
    st.session_state.error_state = True
    
    if st.session_state.advanced_mode:
        st.exception(error)


def load_sample_data() -> None:
    """Load sample data for demonstration."""
    try:
        # Create sample data
        np.random.seed(42)
        n_rows = 1000
        
        sample_data = pd.DataFrame({
            'customer_id': range(1, n_rows + 1),
            'name': [f"Customer {i}" for i in range(1, n_rows + 1)],
            'age': np.random.normal(35, 12, n_rows).astype(int),
            'income': np.random.normal(50000, 15000, n_rows),
            'city': np.random.choice(['New York', 'London', 'Tokyo', 'Sydney'], n_rows),
            'signup_date': pd.date_range('2020-01-01', periods=n_rows, freq='D'),
            'is_premium': np.random.choice([True, False], n_rows, p=[0.3, 0.7]),
            'satisfaction_score': np.random.uniform(1, 5, n_rows)
        })
        
        # Add some missing values
        missing_indices = np.random.choice(n_rows, size=int(n_rows * 0.05), replace=False)
        sample_data.loc[missing_indices, 'income'] = np.nan
        
        missing_indices = np.random.choice(n_rows, size=int(n_rows * 0.02), replace=False)
        sample_data.loc[missing_indices, 'satisfaction_score'] = np.nan
        
        # Add some duplicates
        sample_data = pd.concat([sample_data, sample_data.sample(20)], ignore_index=True)
        
        # Store in session state
        st.session_state.original_df = sample_data.copy()
        st.session_state.current_df = sample_data.copy()
        st.session_state.analysis_results = None
        st.session_state.usability_score = None
        st.session_state.cleaning_pipeline = []
        
        st.success("Sample data loaded! Explore the analysis and cleaning features.")
        st.rerun()
        
    except Exception as e:
        st.error(f"Failed to load sample data: {str(e)}")