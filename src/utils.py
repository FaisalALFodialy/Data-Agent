
import io, csv
import pandas as pd
import json
import os
from typing import Dict, Any, List
import re

from .config import (
    VALIDATION_RULES
)

#CSV loader 
def load_csv_file(uploaded_file) -> pd.DataFrame | None:
    """
    Robust CSV reader:
    - Tries multiple encodings (UTF-8, UTF-8-SIG, Windows-1256, CP1252, Latin-1)
    - Sniffs delimiter (comma/semicolon/tab/pipe) using csv.Sniffer
    - Falls back to pandas inference (sep=None, engine='python')
    - Final fallback: try reading as Excel if it's actually an Excel file
    """
    if uploaded_file is None:
        return None

    # read bytes once
    uploaded_file.seek(0)
    content = uploaded_file.read()
    if not content:
        return None

    # encodings to try (Arabic-friendly first)
    encodings = ["utf-8", "utf-8-sig", "windows-1256", "cp1256", "cp1252", "latin-1"]

    # helper: try with a specific encoding and optional delimiter
    def try_read(encoding: str, sep=None):
        buf = io.BytesIO(content)
        try:
            if sep is None:
                # let pandas infer sep using the python engine (more tolerant)
                return pd.read_csv(buf, encoding=encoding, sep=None, engine="python")
            else:
                return pd.read_csv(buf, encoding=encoding, sep=sep)
        except Exception:
            return None

    # 1) Try to sniff delimiter with csv.Sniffer on a text sample
    for enc in encodings:
        try:
            sample_text = content[:50000].decode(enc, errors="strict")
        except Exception:
            continue

        # Normalize line endings
        sample_text = sample_text.replace("\r\n", "\n").replace("\r", "\n")
        try:
            dialect = csv.Sniffer().sniff(sample_text, delimiters=[",", ";", "\t", "|"])
            sep = dialect.delimiter
            df = try_read(enc, sep=sep)
            if df is not None and df.shape[1] > 0:
                return df
        except Exception:
            # Sniffer failed; we'll try inference paths below
            pass

        # 2) Let pandas infer separator
        df = try_read(enc, sep=None)
        if df is not None and df.shape[1] > 0:
            return df

        # 3) Try common explicit separators as a fallback
        for sep in [",", ";", "\t", "|"]:
            df = try_read(enc, sep=sep)
            if df is not None and df.shape[1] > 0:
                return df

    # 4) Final fallback: maybe it's actually an Excel file with .csv extension
    try:
        buf = io.BytesIO(content)
        df_xls = pd.read_excel(buf)
        if isinstance(df_xls, pd.DataFrame) and df_xls.shape[1] > 0:
            return df_xls
    except Exception:
        pass

    return None

def save_csv_file(df: pd.DataFrame, path: str, include_index: bool = False, encoding: str = "utf-8"):
    df.to_csv(path, index=include_index, encoding=encoding)

def validate_dataframe(df: pd.DataFrame) -> bool:
    return isinstance(df, pd.DataFrame) and df.shape[1] > 0

def create_download_link(df: pd.DataFrame, filename: str) -> str:
    """Create download link using modern Streamlit download button data format."""
    csv_data = df.to_csv(index=False)
    return csv_data

# Removed duplicate function definition - full implementation is below

def format_bytes(num_bytes: int) -> str:
    for unit in ['B','KB','MB','GB','TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"

def calculate_memory_usage(df: pd.DataFrame) -> dict:
    """
    Calculate comprehensive memory usage statistics for dataframe.
    
    Provides detailed memory usage breakdown including total usage,
    per-column usage, and optimization recommendations.
    """
    bytes_total = int(df.memory_usage(deep=True).sum())
    
    # Calculate per-column memory usage
    per_column = {}
    for col in df.columns:
        col_memory = df[col].memory_usage(deep=True)
        per_column[col] = {
            'bytes': int(col_memory),
            'formatted': format_bytes(col_memory),
            'percentage': round((col_memory / bytes_total * 100), 2) if bytes_total > 0 else 0
        }
    
    # Calculate by data type
    by_dtype = {}
    for dtype in df.dtypes.unique():
        dtype_cols = df.select_dtypes(include=[dtype]).columns
        dtype_memory = sum(df[col].memory_usage(deep=True) for col in dtype_cols)
        by_dtype[str(dtype)] = {
            'bytes': int(dtype_memory),
            'formatted': format_bytes(dtype_memory),
            'percentage': round((dtype_memory / bytes_total * 100), 2) if bytes_total > 0 else 0,
            'columns_count': len(dtype_cols)
        }
    
    # Estimate optimization potential
    optimization_potential = 0
    for col in df.columns:
        if df[col].dtype == 'int64':
            # Could potentially use smaller int types
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= -128 and col_max <= 127:
                optimization_potential += df[col].memory_usage(deep=True) * 0.875  # int8 vs int64
            elif col_min >= -32768 and col_max <= 32767:
                optimization_potential += df[col].memory_usage(deep=True) * 0.75   # int16 vs int64
        elif df[col].dtype == 'object':
            # Could potentially use categorical
            if df[col].nunique() / len(df) < 0.05:  # Low cardinality
                optimization_potential += df[col].memory_usage(deep=True) * 0.5
    
    return {
        "bytes": bytes_total, 
        "total": format_bytes(bytes_total),
        "per_column": per_column,
        "by_dtype": by_dtype,
        "optimization_potential": {
            'bytes': int(optimization_potential),
            'formatted': format_bytes(optimization_potential),
            'percentage': round((optimization_potential / bytes_total * 100), 2) if bytes_total > 0 else 0
        }
    }


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of numeric column names from dataframe.
    
    Identifies columns containing numeric data types including
    integers, floats, and numeric objects that can be converted.
    """
    numeric_columns = []
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)
        elif df[col].dtype == 'object':
            # Check if object column contains numeric strings
            try:
                # Try to convert a sample to numeric
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    pd.to_numeric(sample, errors='raise')
                    numeric_columns.append(col)
            except (ValueError, TypeError):
                pass
    
    return numeric_columns


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of categorical column names from dataframe.
    
    Identifies columns containing categorical data including
    strings, objects, and explicitly categorical dtypes.
    """
    categorical_columns = []
    
    for col in df.columns:
        if df[col].dtype == 'category':
            categorical_columns.append(col)
        elif df[col].dtype == 'object':
            # Check if it's not a numeric string or datetime
            if not _is_numeric_like(df[col]) and not _is_datetime_like(df[col]):
                categorical_columns.append(col)
        elif df[col].dtype == 'bool':
            categorical_columns.append(col)
    
    return categorical_columns


def get_datetime_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of datetime column names from dataframe.
    
    Identifies columns containing datetime data including
    explicit datetime types and parseable date strings.
    """
    datetime_columns = []
    
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_columns.append(col)
        elif df[col].dtype == 'object':
            if _is_datetime_like(df[col]):
                datetime_columns.append(col)
    
    return datetime_columns


def _is_numeric_like(series: pd.Series) -> bool:
    """Helper function to check if object series contains numeric data."""
    try:
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
        pd.to_numeric(sample, errors='raise')
        return True
    except (ValueError, TypeError):
        return False


def _is_datetime_like(series: pd.Series) -> bool:
    """Helper function to check if object series contains datetime data."""
    try:
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
        
        # Try to parse as datetime
        pd.to_datetime(sample, errors='raise')
        return True
    except (ValueError, TypeError):
        # Try with common datetime patterns
        datetime_patterns = [
            r'^\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'^\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
        ]
        
        matches = 0
        for value in sample:
            if isinstance(value, str):
                for pattern in datetime_patterns:
                    if re.match(pattern, value.strip()):
                        matches += 1
                        break
        
        return matches / len(sample) > 0.8


def save_pipeline_config(pipeline: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save cleaning pipeline configuration to JSON file.
    
    Exports pipeline configuration for reuse on similar datasets
    or for documenting cleaning procedures.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Add metadata
    config_data = {
        'metadata': {
            'created_at': pd.Timestamp.now().isoformat(),
            'version': '1.0.0',
            'pipeline_length': len(pipeline)
        },
        'pipeline': pipeline
    }
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise IOError(f"Failed to save pipeline configuration: {str(e)}")


def load_pipeline_config(file_path: str) -> List[Dict[str, Any]]:
    """
    Load cleaning pipeline configuration from JSON file.
    
    Imports previously saved pipeline configuration for
    reuse on new datasets or workflow automation.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pipeline configuration file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Validate structure
        if not isinstance(config_data, dict):
            raise ValueError("Invalid configuration file format")
        
        if 'pipeline' not in config_data:
            raise ValueError("Configuration file missing 'pipeline' key")
        
        pipeline = config_data['pipeline']
        
        # Validate pipeline structure
        if not isinstance(pipeline, list):
            raise ValueError("Pipeline must be a list of operations")
        
        for i, operation in enumerate(pipeline):
            if not isinstance(operation, dict):
                raise ValueError(f"Operation {i} must be a dictionary")
            if 'operation' not in operation:
                raise ValueError(f"Operation {i} missing 'operation' key")
        
        return pipeline
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {str(e)}")
    except Exception as e:
        raise IOError(f"Failed to load pipeline configuration: {str(e)}")


def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate dataframe structure and content for processing.
    
    Performs comprehensive validation checks to ensure dataframe
    is suitable for analysis and cleaning operations.
    """
    if not isinstance(df, pd.DataFrame):
        return False
    
    # Check minimum/maximum dimensions
    min_rows = VALIDATION_RULES.get('min_rows', 1)
    min_cols = VALIDATION_RULES.get('min_columns', 1)
    max_cols = VALIDATION_RULES.get('max_columns', 1000)
    
    if df.shape[0] < min_rows:
        return False
    
    if df.shape[1] < min_cols or df.shape[1] > max_cols:
        return False
    
    # Check for valid column names
    for col in df.columns:
        if not isinstance(col, (str, int, float)):
            return False
        if isinstance(col, str) and len(col.strip()) == 0:
            return False
    
    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        return False
    
    # Check data types are supported
    allowed_dtypes = VALIDATION_RULES.get('allowed_dtypes', [])
    if allowed_dtypes:
        for dtype in df.dtypes:
            if str(dtype) not in allowed_dtypes and dtype.name not in allowed_dtypes:
                # Allow common variations
                if not any(str(dtype).startswith(allowed) for allowed in allowed_dtypes):
                    return False
    
    return True


def safe_dataframe(data, **kwargs):
    """
    Safe wrapper for st.dataframe() that automatically sanitizes data for PyArrow compatibility.
    
    Args:
        data: DataFrame or data to display
        **kwargs: Additional arguments passed to st.dataframe()
    
    Returns:
        Result of st.dataframe() call
    """
    import streamlit as st
    
    # If it's a DataFrame, sanitize it
    if isinstance(data, pd.DataFrame):
        sanitized_data = sanitize_dataframe_for_display(data)
        return st.dataframe(sanitized_data, **kwargs)
    else:
        # For non-DataFrame data, pass through directly
        return st.dataframe(data, **kwargs)


def sanitize_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize dataframe for PyArrow-compatible display in Streamlit.
    
    Converts mixed-type object columns to strings and handles numpy data types
    to prevent PyArrow serialization errors when displaying dataframes.
    """
    if df is None or df.empty:
        return df
    
    # Create a copy to avoid modifying original
    display_df = df.copy()
    
    try:
        for col in display_df.columns:
            col_data = display_df[col]
            
            # Handle object columns with mixed types
            if col_data.dtype == 'object':
                try:
                    # Convert to string and replace null representations
                    display_df[col] = col_data.apply(lambda x: 
                        '' if (x is None or pd.isna(x) or str(x).lower() in ['nan', 'none', 'null', '<na>']) 
                        else str(x)
                    )
                except Exception:
                    # Fallback: convert each value individually
                    display_df[col] = col_data.apply(
                        lambda x: '' if (x is None or pd.isna(x) or str(x).lower() in ['nan', 'none', 'null', '<na>']) 
                        else str(x)
                    )
            
            # Convert numpy data types to native Python types
            elif col_data.dtype.name.startswith(('int', 'uint')):
                try:
                    # Convert numpy integers to Python int
                    display_df[col] = col_data.astype('Int64')  # Nullable integer
                except Exception:
                    display_df[col] = col_data.astype(str)
            
            elif col_data.dtype.name.startswith('float'):
                try:
                    # Convert numpy floats to Python float
                    display_df[col] = col_data.astype('Float64')  # Nullable float
                except Exception:
                    display_df[col] = col_data.astype(str)
            
            elif col_data.dtype.name.startswith('bool'):
                try:
                    # Convert numpy bool to Python bool
                    display_df[col] = col_data.astype('boolean')  # Nullable boolean
                except Exception:
                    display_df[col] = col_data.astype(str)
            
            # Handle datetime columns
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                try:
                    # Ensure proper datetime format
                    display_df[col] = pd.to_datetime(col_data, errors='coerce')
                except Exception:
                    display_df[col] = col_data.astype(str)
            
            # Handle categorical columns
            elif col_data.dtype.name == 'category':
                try:
                    # Convert categories to string
                    display_df[col] = col_data.astype(str)
                except Exception:
                    display_df[col] = col_data.apply(lambda x: str(x) if x is not None else '')
        
        # Replace any remaining null values across all columns
        display_df = display_df.fillna('')
        
        # Replace specific null string representations
        for col in display_df.columns:
            if display_df[col].dtype == 'object':
                display_df[col] = display_df[col].replace(['nan', 'NaN', 'None', 'null', 'NULL', '<NA>', 'NaT'], '')
        
        # Final safety check - convert any remaining problematic columns to string
        for col in display_df.columns:
            try:
                # Test if the column can be serialized by trying to convert to dict
                _ = display_df[col].head(1).to_dict()
            except Exception:
                # If serialization fails, convert to string
                display_df[col] = display_df[col].astype(str)
    
    except Exception:
        # Ultimate fallback - convert entire dataframe to strings
        try:
            for col in display_df.columns:
                display_df[col] = display_df[col].astype(str)
        except Exception:
            # If even string conversion fails, return empty dataframe with same structure
            return pd.DataFrame(columns=df.columns)
    
    return display_df


def get_current_dataframe() -> pd.DataFrame:
    """
    Get current dataframe from session state with validation.
    
    Returns the current working dataframe, ensuring it exists and is valid.
    Provides a centralized access point for consistent state management.
    """
    import streamlit as st
    
    if 'current_df' not in st.session_state or st.session_state.current_df is None:
        raise ValueError("No dataframe loaded in session state")
    
    df = st.session_state.current_df
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Invalid dataframe in session state")
    
    return df


def update_current_dataframe(df: pd.DataFrame, operation_name: str = "Unknown") -> None:
    """
    Update current dataframe in session state with validation and history tracking.
    
    Args:
        df: New dataframe to set as current
        operation_name: Name of operation for history tracking
    """
    import streamlit as st
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Invalid dataframe provided for update")
    
    if df.empty:
        raise ValueError("Cannot update with empty dataframe")
    
    # Store the current dataframe as backup before updating
    if 'current_df' in st.session_state and st.session_state.current_df is not None:
        backup_df = st.session_state.current_df.copy()
        
        # Store in operation history if not already there
        if 'operation_history' not in st.session_state:
            st.session_state.operation_history = []
        
        # Only store if this isn't already the most recent operation
        if not st.session_state.operation_history or \
           st.session_state.operation_history[-1].get('operation') != operation_name:
            operation_history = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'operation': operation_name,
                'before_df': backup_df,
                'after_df': df.copy()
            }
            st.session_state.operation_history.append(operation_history)
    
    # Update current dataframe
    st.session_state.current_df = df.copy()
    
    # Invalidate analysis results since data changed
    st.session_state.analysis_results = None
    st.session_state.usability_score = None


def validate_session_state() -> bool:
    """
    Validate session state integrity for data operations.
    
    Returns True if session state is valid, False otherwise.
    """
    import streamlit as st
    
    try:
        # Check if current_df exists and is valid
        if 'current_df' not in st.session_state:
            return False
        
        if st.session_state.current_df is None:
            return False
        
        df = st.session_state.current_df
        if not isinstance(df, pd.DataFrame) or df.empty:
            return False
        
        # Check if required session state keys exist
        required_keys = ['operation_history', 'cleaning_pipeline']
        for key in required_keys:
            if key not in st.session_state:
                st.session_state[key] = []
        
        return True
        
    except Exception:
        return False


def ensure_session_state_integrity() -> None:
    """
    Ensure session state has all required keys and valid data.
    
    Initializes missing keys and validates existing data.
    """
    import streamlit as st
    
    # Initialize required session state keys if missing
    defaults = {
        'operation_history': [],
        'cleaning_pipeline': [],
        'analysis_results': None,
        'usability_score': None,
        'advanced_mode': False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Validate operation_history structure
    if not isinstance(st.session_state.operation_history, list):
        st.session_state.operation_history = []
    
    # Validate cleaning_pipeline structure  
    if not isinstance(st.session_state.cleaning_pipeline, list):
        st.session_state.cleaning_pipeline = []


def generate_cleaning_report(original_df: pd.DataFrame, cleaned_df: pd.DataFrame, operations: List[Dict]) -> str:
    """
    Generate comprehensive report of cleaning operations performed.
    
    Creates detailed report comparing original and cleaned datasets
    with statistics on changes made and operations applied.
    """
    report_lines = []
    report_lines.append("# Data Cleaning Report")
    report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Dataset comparison
    report_lines.append("## Dataset Comparison")
    report_lines.append(f"Original dataset: {original_df.shape[0]:,} rows × {original_df.shape[1]:,} columns")
    report_lines.append(f"Cleaned dataset: {cleaned_df.shape[0]:,} rows × {cleaned_df.shape[1]:,} columns")
    
    rows_removed = original_df.shape[0] - cleaned_df.shape[0]
    cols_removed = original_df.shape[1] - cleaned_df.shape[1]
    
    if rows_removed > 0:
        report_lines.append(f"Rows removed: {rows_removed:,} ({rows_removed/original_df.shape[0]*100:.2f}%)")
    if cols_removed > 0:
        report_lines.append(f"Columns removed: {cols_removed:,}")
    
    report_lines.append("")
    
    # Memory usage comparison
    original_memory = original_df.memory_usage(deep=True).sum()
    cleaned_memory = cleaned_df.memory_usage(deep=True).sum()
    memory_saved = original_memory - cleaned_memory
    
    report_lines.append("## Memory Usage")
    report_lines.append(f"Original: {format_bytes(original_memory)}")
    report_lines.append(f"Cleaned: {format_bytes(cleaned_memory)}")
    if memory_saved > 0:
        report_lines.append(f"Memory saved: {format_bytes(memory_saved)} ({memory_saved/original_memory*100:.2f}%)")
    report_lines.append("")
    
    # Operations performed
    report_lines.append("## Operations Performed")
    if operations:
        for i, op in enumerate(operations, 1):
            op_name = op.get('operation', 'Unknown')
            column = op.get('column', 'N/A')
            parameters = op.get('parameters', {})
            
            report_lines.append(f"{i}. **{op_name}**")
            if column != 'N/A':
                report_lines.append(f"   - Column: {column}")
            if parameters:
                report_lines.append(f"   - Parameters: {parameters}")
            report_lines.append("")
    else:
        report_lines.append("No operations performed.")
        report_lines.append("")
    
    # Data quality improvement
    report_lines.append("## Data Quality Improvement")
    
    # Missing values comparison
    original_missing = original_df.isnull().sum().sum()
    cleaned_missing = cleaned_df.isnull().sum().sum()
    
    report_lines.append(f"Missing values - Original: {original_missing:,}, Cleaned: {cleaned_missing:,}")
    
    if original_missing > cleaned_missing:
        missing_fixed = original_missing - cleaned_missing
        report_lines.append(f"Missing values handled: {missing_fixed:,}")
    
    # Duplicates comparison (if same columns)
    if set(original_df.columns) == set(cleaned_df.columns):
        original_duplicates = original_df.duplicated().sum()
        cleaned_duplicates = cleaned_df.duplicated().sum()
        
        report_lines.append(f"Duplicates - Original: {original_duplicates:,}, Cleaned: {cleaned_duplicates:,}")
        
        if original_duplicates > cleaned_duplicates:
            duplicates_removed = original_duplicates - cleaned_duplicates
            report_lines.append(f"Duplicates removed: {duplicates_removed:,}")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("Report generated by CSV Data Analysis & Cleaning Tool")
    
    return "\n".join(report_lines)
