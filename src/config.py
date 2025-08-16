"""
Configuration Constants and Settings

This module contains all configuration variables, constants, and default
settings for the CSV Data Analysis and Cleaning Tool.

"""

from typing import List, Dict, Any
import os

# FILE HANDLING CONFIGURATION

# Supported file types for upload
UPLOAD_FILE_TYPES: List[str] = ['csv']

# Maximum file size (200MB)
MAX_FILE_SIZE: int = 200 * 1024 * 1024

# Default encoding options to try when reading CSV files
DEFAULT_ENCODINGS: List[str] = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

# CSV reading parameters
CSV_READ_PARAMS: Dict[str, Any] = {
    'low_memory': False,
    'na_values': ['', 'N/A', 'NA', 'NULL', 'null', 'NaN', 'nan', '#N/A', '#VALUE!', '#NULL!'],
    'keep_default_na': True,
    'skip_blank_lines': True
}

# DATA QUALITY THRESHOLDS

# Missing value threshold (50% - columns with more missing values are flagged)
MISSING_VALUE_THRESHOLD: float = 0.5

# High cardinality threshold (90% - columns with >90% unique values)
HIGH_CARDINALITY_THRESHOLD: float = 0.9

# Low cardinality threshold (5% - columns with <5% unique values for categorical optimization)
LOW_CARDINALITY_THRESHOLD: float = 0.05

# Outlier detection thresholds
OUTLIER_IQR_MULTIPLIER: float = 1.5
OUTLIER_ZSCORE_THRESHOLD: float = 3.0

# Duplicate detection threshold
DUPLICATE_THRESHOLD: float = 0.01  # 1% duplicates triggers warning

# DATA CLEANING OPTIONS

# Available outlier detection methods
OUTLIER_METHODS: List[str] = ['iqr', 'zscore', 'isolation_forest']

# Missing value treatment options for numeric columns
MISSING_NUMERIC_METHODS: List[str] = [
    'mean',
    'median', 
    'mode',
    'forward_fill',
    'backward_fill',
    'interpolate',
    'constant',
    'drop'
]

# Missing value treatment options for categorical columns
MISSING_CATEGORICAL_METHODS: List[str] = [
    'mode',
    'constant',
    'forward_fill',
    'backward_fill',
    'new_category',
    'drop'
]

# Text cleaning operations
TEXT_CLEANING_OPERATIONS: List[str] = [
    'lowercase',
    'uppercase', 
    'title_case',
    'trim_whitespace',
    'remove_special_chars',
    'remove_extra_spaces',
    'standardize_quotes'
]

# Data type optimization options
DTYPE_OPTIMIZATION_OPTIONS: Dict[str, List[str]] = {
    'numeric': ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64'],
    'categorical': ['category'],
    'datetime': ['datetime64[ns]'],
    'string': ['string', 'object']
}

# USABILITY SCORING WEIGHTS

# Weights for calculating overall usability score (must sum to 1.0)
USABILITY_SCORE_WEIGHTS: Dict[str, float] = {
    'completeness': 0.30,    # Based on missing values
    'consistency': 0.25,     # Based on data type consistency and format
    'validity': 0.25,        # Based on outliers and invalid values
    'uniqueness': 0.20       # Based on duplicate detection
}

# Score grade boundaries
USABILITY_SCORE_GRADES: Dict[str, Dict[str, float]] = {
    'A': {'min': 90, 'max': 100, 'description': 'Excellent - Ready for analysis'},
    'B': {'min': 80, 'max': 89, 'description': 'Good - Minor cleaning recommended'},
    'C': {'min': 70, 'max': 79, 'description': 'Fair - Moderate cleaning needed'},
    'D': {'min': 60, 'max': 69, 'description': 'Poor - Significant cleaning required'},
    'F': {'min': 0, 'max': 59, 'description': 'Failing - Extensive cleaning needed'}
}

# VISUALIZATION SETTINGS

# Default color palettes for visualizations
COLOR_PALETTES: Dict[str, List[str]] = {
    'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'sequential': ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef'],
    'diverging': ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#92c5de'],
    'qualitative': ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']
}

# Plot configuration defaults
PLOT_CONFIG: Dict[str, Any] = {
    'width': 800,
    'height': 500,
    'template': 'plotly_white',
    'font_size': 12,
    'title_font_size': 16,
    'show_toolbar': True
}

# Missing values heatmap settings
HEATMAP_CONFIG: Dict[str, Any] = {
    'colorscale': 'Reds',
    'show_colorbar': True,
    'aspect_ratio': 'auto'
}

# STREAMLIT SESSION STATE KEYS

# Session state variable names
SESSION_STATE_KEYS: Dict[str, str] = {
    'uploaded_file': 'uploaded_file',
    'original_df': 'original_df',
    'current_df': 'current_df',
    'analysis_results': 'analysis_results',
    'cleaning_pipeline': 'cleaning_pipeline',
    'cleaning_history': 'cleaning_history',
    'usability_score': 'usability_score',
    'selected_columns': 'selected_columns',
    'preview_df': 'preview_df',
    'operation_history': 'operation_history'
}

# ANALYSIS PARAMETERS

# Statistical analysis parameters
STATS_CONFIG: Dict[str, Any] = {
    'describe_percentiles': [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
    'correlation_threshold': 0.5,
    'skewness_threshold': 1.0,
    'kurtosis_threshold': 3.0
}

# Memory usage calculation parameters
MEMORY_CONFIG: Dict[str, Any] = {
    'deep_analysis': True,
    'sample_size': 10000,  # Sample size for large datasets
    'precision': 2         # Decimal places for memory reporting
}

# EXPORT SETTINGS

# Default export parameters
EXPORT_CONFIG: Dict[str, Any] = {
    'csv_separator': ',',
    'csv_encoding': 'utf-8',
    'include_index': False,
    'date_format': '%Y-%m-%d',
    'float_format': None
}

# Report generation settings
REPORT_CONFIG: Dict[str, Any] = {
    'include_visualizations': True,
    'include_statistics': True,
    'include_recommendations': True,
    'format': 'html'  # Options: 'html', 'pdf', 'markdown'
}

# PERFORMANCE SETTINGS

# Performance optimization thresholds
PERFORMANCE_CONFIG: Dict[str, Any] = {
    'large_dataset_threshold': 100000,    # Rows
    'memory_warning_threshold': 500,      # MB
    'processing_chunk_size': 10000,       # Rows per chunk
    'max_correlation_features': 100,      # Max features for correlation matrix
    'sampling_threshold': 50000           # Sample large datasets for preview
}

# Caching configuration
CACHE_CONFIG: Dict[str, Any] = {
    'ttl': 3600,  # Cache time-to-live in seconds (1 hour)
    'max_entries': 100,
    'persist': False
}

# LOGGING AND DEBUGGING

# Logging configuration
LOGGING_CONFIG: Dict[str, Any] = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': ['console'],  # Options: 'console', 'file'
    'log_file': 'data_analysis_tool.log'
}

# Debug settings
DEBUG_CONFIG: Dict[str, Any] = {
    'enable_profiling': False,
    'show_memory_usage': False,
    'verbose_operations': False
}

# VALIDATION RULES

# Data validation rules
VALIDATION_RULES: Dict[str, Any] = {
    'min_rows': 1,
    'min_columns': 1,
    'max_columns': 1000,
    'allowed_dtypes': ['int64', 'float64', 'object', 'datetime64[ns]', 'bool', 'category'],
    'check_encoding': True,
    'check_delimiters': True
}

# Column name validation
COLUMN_NAME_RULES: Dict[str, Any] = {
    'max_length': 100,
    'allowed_chars': r'^[a-zA-Z0-9_\-\s]+$',
    'no_duplicates': True,
    'standardize_names': True
}

# HELPER FUNCTIONS

def get_config(section: str, key: str = None) -> Any:
    """
    Get configuration value by section and optionally by key.
    
    Args:
        section (str): Configuration section name
        key (str, optional): Specific key within section
        
    Returns:
        Any: Configuration value or entire section
        
    Example:
        >>> max_size = get_config('file', 'max_size')
        >>> all_thresholds = get_config('thresholds')
    """
    config_map = {
        'file': {
            'types': UPLOAD_FILE_TYPES,
            'max_size': MAX_FILE_SIZE,
            'encodings': DEFAULT_ENCODINGS
        },
        'thresholds': {
            'missing_values': MISSING_VALUE_THRESHOLD,
            'high_cardinality': HIGH_CARDINALITY_THRESHOLD,
            'low_cardinality': LOW_CARDINALITY_THRESHOLD
        },
        'cleaning': {
            'numeric_methods': MISSING_NUMERIC_METHODS,
            'categorical_methods': MISSING_CATEGORICAL_METHODS,
            'text_operations': TEXT_CLEANING_OPERATIONS
        },
        'scoring': USABILITY_SCORE_WEIGHTS,
        'visualization': {
            'colors': COLOR_PALETTES,
            'plot': PLOT_CONFIG
        },
        'performance': PERFORMANCE_CONFIG
    }
    
    if section in config_map:
        if key is None:
            return config_map[section]
        return config_map[section].get(key)
    
    return None


def validate_config() -> bool:
    """
    Validate configuration consistency and values.
    
    Returns:
        bool: True if configuration is valid, False otherwise
        
    TODO: Implement configuration validation:
          - Check that weights sum to 1.0
          - Validate threshold ranges (0-1)
          - Ensure required keys exist
          - Check file paths and permissions
    """
    # Validate usability score weights sum to 1.0
    weights_sum = sum(USABILITY_SCORE_WEIGHTS.values())
    if abs(weights_sum - 1.0) > 0.001:
        return False
    
    # Validate threshold ranges
    thresholds = [MISSING_VALUE_THRESHOLD, HIGH_CARDINALITY_THRESHOLD, LOW_CARDINALITY_THRESHOLD]
    if not all(0.0 <= t <= 1.0 for t in thresholds):
        return False
    
    return True


# Configuration validation on import
if not validate_config():
    raise ValueError("Invalid configuration detected. Please check config.py settings.")