"""
Data Cleaning Module

This module provides comprehensive data cleaning and transformation functions
for the CSV Data Analysis and Cleaning Tool. Includes missing value treatment,
outlier handling, duplicate removal, and data optimization operations.

"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import re
import warnings

from .config import (
    MISSING_NUMERIC_METHODS,
    MISSING_CATEGORICAL_METHODS,
    TEXT_CLEANING_OPERATIONS,
    OUTLIER_IQR_MULTIPLIER,
    OUTLIER_ZSCORE_THRESHOLD,
    DTYPE_OPTIMIZATION_OPTIONS
)

warnings.filterwarnings('ignore', category=RuntimeWarning)


def handle_missing_numeric(df: pd.DataFrame, column: str, method: str, custom_value: Optional[float] = None) -> pd.DataFrame:
    """
    Handle missing values in numeric columns using various imputation methods.

    Provides multiple strategies for filling missing numeric values,
    from simple statistical measures to advanced interpolation techniques.

    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Name of the numeric column to process
        method (str): Imputation method from MISSING_NUMERIC_METHODS
                     ('mean', 'median', 'mode', 'forward_fill', 'backward_fill',
                      'interpolate', 'constant', 'drop')
        custom_value (Optional[float]): Value to use for 'constant' method

    Returns:
        pd.DataFrame: Dataframe with missing values handled in the specified column

    Raises:
        ValueError: If method is invalid or column is not numeric
        KeyError: If column doesn't exist

    Example:
        >>> df = pd.DataFrame({'A': [1, 2, np.nan, 4]})
        >>> cleaned_df = handle_missing_numeric(df, 'A', 'mean')
        >>> print(cleaned_df['A'].isna().sum())  # 0
    """
    # Validate if column exists, if not raise an error
    if column not in df.columns:
        raise KeyError(f"Column '{column}' does not exist in the dataframe")

    # Validate column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric")

    # Validate method
    if method not in MISSING_NUMERIC_METHODS:
        raise ValueError(f"Invalid method '{method}'. Must be one of {MISSING_NUMERIC_METHODS}")

    # Create a copy to avoid modifying original dataframe (Good practice as Rakan, Mohammed suggested)
    df_copy = df.copy()

    # Check if all values are missing
    if df_copy[column].isna().all():
        if method in ['mean', 'median', 'mode', 'interpolate']:
            warnings.warn(f"All values in column '{column}' are missing. Cannot compute {method}.")
            return df_copy
        elif method == 'constant' and custom_value is None:
            raise ValueError("custom_value is required when method is 'constant' and all values are missing")

    # Handle different imputation methods
    if method == 'mean':
        fill_value = df_copy[column].mean()
        df_copy[column] = df_copy[column].fillna(fill_value)

    elif method == 'median':
        fill_value = df_copy[column].median()
        df_copy[column] = df_copy[column].fillna(fill_value)

    elif method == 'mode':
        mode_values = df_copy[column].mode()
        if len(mode_values) > 0:
            fill_value = mode_values.iloc[0]
            df_copy[column] = df_copy[column].fillna(fill_value)
        else:
            warnings.warn(f"No mode found for column '{column}'. Values remain unchanged.")

    elif method == 'forward_fill':
        df_copy[column] = df_copy[column].fillna(method='ffill')

    elif method == 'backward_fill':
        df_copy[column] = df_copy[column].fillna(method='bfill')

    elif method == 'interpolate':
        # Only interpolate if there are at least 2 non-null values
        if df_copy[column].notna().sum() >= 2:
            df_copy[column] = df_copy[column].interpolate()
        else:
            warnings.warn(f"Not enough non-null values in column '{column}' for interpolation. Values remain unchanged.")

    elif method == 'constant':
        if custom_value is None:
            raise ValueError("custom_value is required when method is 'constant'")
        df_copy[column] = df_copy[column].fillna(custom_value)

    elif method == 'drop':
        df_copy = df_copy.dropna(subset=[column])

    # Preserve original data type
    original_dtype = df[column].dtype
    if method != 'drop':  # Don't need to preserve dtype if dropping rows
        try:
            df_copy[column] = df_copy[column].astype(original_dtype)
        except (ValueError, TypeError):
            # If conversion fails, keep the current dtype
            warnings.warn(f"Could not preserve original dtype {original_dtype} for column '{column}'")

    return df_copy


def handle_missing_categorical(
        df: pd.DataFrame,
        column: str,
        method: str,
        custom_value: Optional[str] = None
) -> pd.DataFrame:
    """
    Handle missing values in categorical columns using appropriate methods.

    Methods (must be in MISSING_CATEGORICAL_METHODS):
      'mode'          : Fill with most frequent category (ties resolved by first in .mode()).
      'constant'      : Fill with custom_value or default 'Unknown'.
      'forward_fill'  : Fill with last valid category (ffill).
      'backward_fill' : Fill with next valid category (bfill).
      'new_category'  : Fill with literal 'Missing' (added to categories if needed).
      'drop'          : Drop rows where the column is missing.

    Notes:
      - Preserves pandas Categorical dtype by adding categories before filling.
      - For all-missing columns, 'mode'/'forward_fill'/'backward_fill' are no-ops with a warning.
      - For 'constant' and 'new_category', filling proceeds even if all values are missing.
    """
    # validations
    if column not in df.columns:
        raise KeyError(f"Column '{column}' does not exist in the dataframe")

    if method not in MISSING_CATEGORICAL_METHODS:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of {MISSING_CATEGORICAL_METHODS}"
        )

    df_copy = df.copy()

    # helpers to ensure category dtype
    is_cat = pd.api.types.is_categorical_dtype(df_copy[column])

    def _ensure_category(col: pd.Series, value: Optional[str]) -> pd.Series:
        if is_cat and value is not None and value not in col.cat.categories:
            return col.cat.add_categories([value])
        return col

    # all-missing short-circuits for methods that cannot compute from data
    if df_copy[column].isna().all():
        if method in ("mode", "forward_fill", "backward_fill"):
            warnings.warn(
                f"All values in column '{column}' are missing. Cannot compute {method}."
            )
            return df_copy
        # 'constant', 'new_category', and 'drop' proceed

    # methods
    if method == "mode":
        mode_values = df_copy[column].mode(dropna=True)
        if len(mode_values) > 0:
            fill_value = mode_values.iloc[0]
            df_copy[column] = _ensure_category(df_copy[column], fill_value)
            df_copy[column] = df_copy[column].fillna(fill_value)
        else:
            warnings.warn(f"No mode found for column '{column}'. Values remain unchanged.")

    elif method == "constant":
        fill_value = "Unknown" if custom_value is None else custom_value
        df_copy[column] = _ensure_category(df_copy[column], fill_value)
        df_copy[column] = df_copy[column].fillna(fill_value)

    elif method == "forward_fill":
        df_copy[column] = df_copy[column].ffill()

    elif method == "backward_fill":
        df_copy[column] = df_copy[column].bfill()

    elif method == "new_category":
        fill_value = "Missing"
        df_copy[column] = _ensure_category(df_copy[column], fill_value)
        df_copy[column] = df_copy[column].fillna(fill_value)

    elif method == "drop":
        df_copy = df_copy.dropna(subset=[column])

    # edge notification for ffill/bfill when leading/trailing NaNs remain
    if method in ("forward_fill", "backward_fill") and df_copy[column].isna().any():
        warnings.warn(
            f"Some missing values in column '{column}' could not be filled using {method}."
        )

    return df_copy



def remove_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from a numeric column using the IQR method.
    Refer to https://medium.com/@pp1222001/outlier-detection-and-removal-using-the-iqr-method-6fab2954315d
    Notes:
      - Rows with NaN in `column` are retained (they are not treated as outliers).
      - If IQR is 0 or undefined (e.g., constant/empty after dropna), the input is returned unchanged.
    """

    # validations
    if column not in df.columns:
        raise KeyError(f"Column '{column}' does not exist in the dataframe")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")

    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("`multiplier` must be a positive, finite number")

    s = df[column] # Series to process
    non_null = s.dropna()
    if non_null.empty:
        warnings.warn(f"Column '{column}' contains only missing values; nothing to remove.")
        return df.copy()

    # IQR computation
    q1, q3 = non_null.quantile([0.25, 0.75])
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        warnings.warn(f"IQR is zero/undefined for column '{column}'; no outliers detected.")
        return df.copy()

    lo = q1 - multiplier * iqr
    hi = q3 + multiplier * iqr

    # Keep values within [lo, hi]; keep NaNs as-is
    in_range_or_nan = s.isna() | ((s >= lo) & (s <= hi))  # NaNs are always kept, s <= lo and s >= hi are outliers.
    result = df.loc[in_range_or_nan].copy()

    # Edge-case notifications (non-fatal)
    removed = len(df) - len(result)
    if removed == 0:
        # optional: no warning needed, but helpful in pipelines..
        warnings.warn(f"No outliers removed from column '{column}' (multiplier={multiplier}).")
    elif len(result) == 0:
        warnings.warn(f"All rows considered outliers in column '{column}'. Returning empty DataFrame.")

    return result

def cap_outliers(
        df: pd.DataFrame,
        column: str,
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99
) -> pd.DataFrame:
    """
    Cap extreme values in a numeric column at specified percentile boundaries.

    Keeps all rows while reducing the influence of extreme values by clipping
    them to the chosen lower / upper percentile cutoffs.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Numeric column to process.
        lower_percentile (float): Lower percentile in (0,1). Default 0.01.
        upper_percentile (float): Upper percentile in (0,1). Default 0.99.

    Returns:
        pd.DataFrame: Copy of DataFrame with capped column.

    Raises:
        KeyError: If column not found.
        ValueError: On invalid dtype or percentile parameters.
    """
    # Column existence
    if column not in df.columns:
        raise KeyError(f"Column '{column}' does not exist in the dataframe")

    # Numeric dtype check
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")

    # Percentile parameter validation
    for name, p in (("lower_percentile", lower_percentile), ("upper_percentile", upper_percentile)):
        if not isinstance(p, (int, float)) or not np.isfinite(p):
            raise ValueError(f"{name} must be a finite number")
        if p <= 0 or p >= 1:
            raise ValueError(f"{name} must be within the open interval (0,1)")

    if lower_percentile >= upper_percentile:
        raise ValueError("lower_percentile must be < upper_percentile")

    s = df[column]
    non_null = s.dropna()

    # All missing -> nothing to do
    if non_null.empty:
        warnings.warn(f"Column '{column}' contains only missing values; no capping applied.")
        return df.copy()

    # Compute bounds
    lower_bound = non_null.quantile(lower_percentile)
    upper_bound = non_null.quantile(upper_percentile)

    # Constant / degenerate
    if pd.isna(lower_bound) or pd.isna(upper_bound):
        warnings.warn(f"Percentile bounds undefined for column '{column}'; no capping applied.")
        return df.copy()
    if lower_bound == upper_bound:
        warnings.warn(f"Column '{column}' appears constant at value {lower_bound}; no capping needed.")
        return df.copy()

    df_copy = df.copy()
    col = df_copy[column]

    # Identify values to cap (exclude NaNs)
    mask_low = col < lower_bound
    mask_high = col > upper_bound
    n_low = int(mask_low.sum())
    n_high = int(mask_high.sum())
    total_capped = n_low + n_high

    if total_capped == 0:
        # No change but still return a copy for consistency
        return df_copy

    # Apply clipping (preserves NaNs)
    df_copy[column] = col.clip(lower=lower_bound, upper=upper_bound)

    # Attempt to preserve original dtype
    orig_dtype = df[column].dtype
    try:
        # If original was an integer dtype but NaNs exist, casting will fail; ignore gracefully
        df_copy[column] = df_copy[column].astype(orig_dtype)
    except (ValueError, TypeError):
        # Keep clipped dtype (likely float) if safe cast not possible
        pass

    warnings.warn(
        f"Capped {total_capped} value(s) in column '{column}' "
        f"(lower: {n_low}, upper: {n_high}) at "
        f"[{lower_bound:.6g}, {upper_bound:.6g}] "
        f"(p={lower_percentile}, {upper_percentile})."
    )

    return df_copy


def remove_duplicates(
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: Union[str, bool] = "first"
) -> pd.DataFrame:
    """
    Remove duplicate rows.
    Args:
        df (pd.DataFrame): Input DataFrame.
        subset (Optional[List[str]]): Columns to consider for identifying duplicates.
                                      If None, all columns are used.
        keep (str | bool): Which duplicates to keep:
            'first' (default): Keep first occurrence.
            'last' : Keep last occurrence.
            False  : Drop all duplicates.

    Returns:
        pd.DataFrame: Copy without (selected) duplicate rows.

    Raises:
        KeyError: If any column in subset does not exist.
        ValueError: If keep is invalid.
        TypeError: If subset is not a sequence of strings.

    Notes:
        - Original DataFrame is never mutated.
        - If no duplicates are found, a copy is still returned.
    """
    # Validate keep
    if keep not in ("first", "last", False):
        raise ValueError("keep must be one of {'first','last', False}")

    # Validate subset
    if subset is not None:
        if not isinstance(subset, (list, tuple)):
            raise TypeError("subset must be a list or tuple of column names")
        missing = [c for c in subset if c not in df.columns]
        if missing:
            raise KeyError(f"Columns not found in DataFrame: {missing}")

    df_copy = df.copy()

    before = len(df_copy)
    result = df_copy.drop_duplicates(subset=subset, keep=keep)
    removed = before - len(result)

    if removed == 0:
        warnings.warn("No duplicate rows removed.")
    else:
        cols_repr = subset if subset is not None else "all columns"
        warnings.warn(f"Removed {removed} duplicate row(s) based on {cols_repr} (keep={keep}).")

    return result


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize dataframe memory usage by converting to more efficient data types.
    
    Analyzes each column and converts to the most memory-efficient
    data type while preserving data integrity and precision.
    
    Args:
        df (pd.DataFrame): Input dataframe to optimize
        
    Returns:
        pd.DataFrame: Dataframe with optimized data types
        
    Example:
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [1.0, 2.0, 3.0]})
        >>> optimized_df = optimize_dtypes(df)
        >>> print(optimized_df.dtypes)  # Shows optimized types
        
    """
    df_optimized = df.copy()
    original_memory = df.memory_usage(deep=True).sum()
    
    for column in df_optimized.columns:
        col_data = df_optimized[column]
        
        # Skip if column is all null
        if col_data.isna().all():
            continue
            
        # Handle numeric columns
        if pd.api.types.is_numeric_dtype(col_data):
            # Integer optimization
            if pd.api.types.is_integer_dtype(col_data):
                col_min, col_max = col_data.min(), col_data.max()
                
                # Try int8 (-128 to 127)
                if col_min >= -128 and col_max <= 127:
                    df_optimized[column] = col_data.astype('int8')
                # Try int16 (-32,768 to 32,767)
                elif col_min >= -32768 and col_max <= 32767:
                    df_optimized[column] = col_data.astype('int16')
                # Try int32 (-2,147,483,648 to 2,147,483,647)
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df_optimized[column] = col_data.astype('int32')
                # Keep as int64
                else:
                    df_optimized[column] = col_data.astype('int64')
                    
            # Float optimization
            elif pd.api.types.is_float_dtype(col_data):
                # Try float32 if precision allows
                try:
                    col_float32 = col_data.astype('float32')
                    # Check if conversion is lossless (within reasonable tolerance)
                    if np.allclose(col_data.dropna(), col_float32.dropna(), rtol=1e-6, atol=1e-6, equal_nan=True):
                        df_optimized[column] = col_float32
                    else:
                        df_optimized[column] = col_data.astype('float64')
                except (ValueError, OverflowError):
                    df_optimized[column] = col_data.astype('float64')
        
        # Handle object columns
        elif col_data.dtype == 'object':
            # Check for datetime patterns
            if _is_datetime_like(col_data):
                try:
                    df_optimized[column] = pd.to_datetime(col_data, errors='coerce')
                    continue
                except (ValueError, TypeError):
                    pass
            
            # Check for numeric strings
            if _is_numeric_string(col_data):
                try:
                    # Try integer first
                    numeric_converted = pd.to_numeric(col_data, errors='coerce')
                    if numeric_converted.notna().all() or col_data.isna().sum() == numeric_converted.isna().sum():
                        if (numeric_converted == numeric_converted.astype('int64')).all():
                            df_optimized[column] = numeric_converted.astype('int64')
                        else:
                            df_optimized[column] = numeric_converted
                        continue
                except (ValueError, TypeError):
                    pass
            
            # Check for categorical optimization (low cardinality)
            unique_count = col_data.nunique()
            total_count = len(col_data)
            cardinality_ratio = unique_count / total_count if total_count > 0 else 0
            
            # Convert to categorical if cardinality is low (< 5% unique values or < 50 unique values)
            if cardinality_ratio < 0.05 or unique_count < 50:
                df_optimized[column] = col_data.astype('category')
    
    optimized_memory = df_optimized.memory_usage(deep=True).sum()
    memory_saved = original_memory - optimized_memory
    savings_percentage = (memory_saved / original_memory) * 100 if original_memory > 0 else 0
    
    # Add optimization metadata
    if hasattr(df_optimized, 'attrs'):
        df_optimized.attrs['optimization_info'] = {
            'original_memory': original_memory,
            'optimized_memory': optimized_memory,
            'memory_saved': memory_saved,
            'savings_percentage': savings_percentage
        }
    
    return df_optimized


def _is_numeric_string(series: pd.Series) -> bool:
    """Helper function to check if object series contains numeric strings."""
    try:
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
        pd.to_numeric(sample, errors='raise')
        return True
    except (ValueError, TypeError):
        return False


def _is_datetime_like(series: pd.Series) -> bool:
    """Helper function to check if object series contains datetime strings."""
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


def _is_datetime_like(series: pd.Series) -> bool:
    """Check if a series contains datetime-like strings."""
    if series.dtype != 'object':
        return False
    
    # Sample non-null values
    sample = series.dropna().head(100)
    if len(sample) == 0:
        return False
    
    # Common datetime patterns
    datetime_patterns = [
        r'^\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'^\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
        r'^\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
        r'^\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
    ]
    
    matches = 0
    for value in sample:
        if isinstance(value, str):
            for pattern in datetime_patterns:
                if re.match(pattern, value.strip()):
                    matches += 1
                    break
    
    # If more than 80% match datetime patterns
    return matches / len(sample) > 0.8


def _is_numeric_string(series: pd.Series) -> bool:
    """Check if a series contains numeric strings."""
    if series.dtype != 'object':
        return False
    
    # Sample non-null values
    sample = series.dropna().head(100)
    if len(sample) == 0:
        return False
    
    # Try to convert to numeric
    try:
        numeric_converted = pd.to_numeric(sample, errors='coerce')
        valid_conversions = numeric_converted.notna().sum()
        # If more than 90% can be converted to numeric
        return valid_conversions / len(sample) > 0.9
    except:
        return False


def clean_text_column(df: pd.DataFrame, column: str, operations: List[str]) -> pd.DataFrame:
    """
    Apply text cleaning operations to a string column.
    
    Performs various text standardization and cleaning operations
    to improve data consistency and quality.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Name of the text column to clean
        operations (List[str]): List of operations from TEXT_CLEANING_OPERATIONS
                               ('lowercase', 'uppercase', 'title_case', 'trim_whitespace',
                                'remove_special_chars', 'remove_extra_spaces', 'standardize_quotes')
        
    Returns:
        pd.DataFrame: Dataframe with text column cleaned
        
    Raises:
        ValueError: If operations list contains invalid operation
        KeyError: If column doesn't exist
        
    Example:
        >>> df = pd.DataFrame({'text': ['  Hello World  ', 'GOODBYE']})
        >>> cleaned_df = clean_text_column(df, 'text', ['lowercase', 'trim_whitespace'])
        >>> print(cleaned_df['text'].tolist())  # ['hello world', 'goodbye']
        
    """
    # Validate inputs
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataframe")
    
    # Validate operations
    valid_operations = set(TEXT_CLEANING_OPERATIONS)
    invalid_ops = set(operations) - valid_operations
    if invalid_ops:
        raise ValueError(f"Invalid operations: {invalid_ops}. Valid operations: {valid_operations}")
    
    df_cleaned = df.copy()
    
    # Apply operations in the specified order
    for operation in operations:
        # Skip null values in all operations
        mask = df_cleaned[column].notna()
        
        if operation == 'lowercase':
            df_cleaned.loc[mask, column] = df_cleaned.loc[mask, column].str.lower()
            
        elif operation == 'uppercase':
            df_cleaned.loc[mask, column] = df_cleaned.loc[mask, column].str.upper()
            
        elif operation == 'title_case':
            df_cleaned.loc[mask, column] = df_cleaned.loc[mask, column].str.title()
            
        elif operation == 'trim_whitespace':
            df_cleaned.loc[mask, column] = df_cleaned.loc[mask, column].str.strip()
            
        elif operation == 'remove_special_chars':
            # Keep only alphanumeric characters and spaces
            df_cleaned.loc[mask, column] = df_cleaned.loc[mask, column].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
            
        elif operation == 'remove_extra_spaces':
            # Replace multiple consecutive spaces with single space
            df_cleaned.loc[mask, column] = df_cleaned.loc[mask, column].str.replace(r'\s+', ' ', regex=True)
            
        elif operation == 'standardize_quotes':
            # Replace various quote types with standard double quotes
            df_cleaned.loc[mask, column] = df_cleaned.loc[mask, column].str.replace(r'[""''`´]', '"', regex=True)
    
    return df_cleaned


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names for consistency and usability.
    
    Applies consistent naming conventions to improve data accessibility
    and prevent issues with special characters or spaces in column names.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with standardized column names
        
    Example:
        >>> df = pd.DataFrame({'  Column Name!  ': [1, 2], 'another-col': [3, 4]})
        >>> standardized_df = standardize_column_names(df)
        >>> print(standardized_df.columns.tolist())  # ['column_name', 'another_col']
        
    """
    df_standardized = df.copy()
    new_columns = []
    
    for original_name in df.columns:
        # Convert to string if not already
        name = str(original_name)
        
        # Trim whitespace
        name = name.strip()
        
        # Convert to lowercase
        name = name.lower()
        
        # Replace spaces and other separators with underscores
        name = re.sub(r'[\s\-\.]+', '_', name)
        
        # Remove special characters (keep only alphanumeric and underscore)
        name = re.sub(r'[^a-z0-9_]', '', name)
        
        # Remove multiple consecutive underscores
        name = re.sub(r'_+', '_', name)
        
        # Remove leading/trailing underscores
        name = name.strip('_')
        
        # Ensure name starts with letter or underscore (not number)
        if name and name[0].isdigit():
            name = 'col_' + name
            
        # Handle empty names
        if not name:
            name = f'unnamed_column_{len(new_columns)}'
            
        new_columns.append(name)
    
    # Handle duplicate names
    final_columns = []
    name_counts = {}
    
    for name in new_columns:
        if name in name_counts:
            name_counts[name] += 1
            final_name = f"{name}_{name_counts[name]}"
        else:
            name_counts[name] = 0
            final_name = name
        final_columns.append(final_name)
    
    # Apply new column names
    df_standardized.columns = final_columns
    
    return df_standardized


def apply_cleaning_pipeline(df: pd.DataFrame, pipeline: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Apply a sequence of cleaning operations defined in a pipeline.
    
    Executes multiple cleaning operations in order, allowing for
    complex data cleaning workflows with proper error handling.
    
    Args:
        df (pd.DataFrame): Input dataframe to clean
        pipeline (List[Dict[str, Any]]): List of operation dictionaries, each containing:
                                        - 'operation': Function name to call
                                        - 'column': Target column (if applicable)
                                        - 'parameters': Dict of operation parameters
        
    Returns:
        pd.DataFrame: Dataframe after applying all pipeline operations
        
    Raises:
        ValueError: If pipeline contains invalid operations
        
    Example:
        >>> pipeline = [
        ...     {'operation': 'handle_missing_numeric', 'column': 'age', 'parameters': {'method': 'mean'}},
        ...     {'operation': 'remove_duplicates', 'parameters': {'keep': 'first'}}
        ... ]
        >>> cleaned_df = apply_cleaning_pipeline(df, pipeline)
        
    """
    if not pipeline:
        return df.copy()
    
    # Define available operations mapping
    operation_map = {
        'handle_missing_numeric': handle_missing_numeric,
        'handle_missing_categorical': handle_missing_categorical,
        'remove_outliers_iqr': remove_outliers_iqr,
        'cap_outliers': cap_outliers,
        'remove_duplicates': remove_duplicates,
        'optimize_dtypes': optimize_dtypes,
        'clean_text_column': clean_text_column,
        'standardize_column_names': standardize_column_names,
    }
    
    current_df = df.copy()
    applied_operations = []
    
    try:
        for i, operation_config in enumerate(pipeline):
            # Validate operation structure
            if not isinstance(operation_config, dict):
                raise ValueError(f"Operation {i} must be a dictionary")
            
            if 'operation' not in operation_config:
                raise ValueError(f"Operation {i} missing 'operation' key")
            
            operation_name = operation_config['operation']
            
            # Check if operation is supported
            if operation_name not in operation_map:
                raise ValueError(f"Unsupported operation: {operation_name}. Available: {list(operation_map.keys())}")
            
            operation_func = operation_map[operation_name]
            
            # Prepare parameters
            parameters = operation_config.get('parameters', {})
            column = operation_config.get('column')
            
            # Execute operation based on its signature
            try:
                if operation_name in ['handle_missing_numeric', 'handle_missing_categorical', 
                                    'remove_outliers_iqr', 'cap_outliers', 'clean_text_column']:
                    # Operations that require a column
                    if not column:
                        raise ValueError(f"Operation {operation_name} requires a 'column' parameter")
                    current_df = operation_func(current_df, column, **parameters)
                    
                elif operation_name == 'remove_duplicates':
                    # Handle subset parameter specially
                    subset = parameters.get('subset')
                    keep = parameters.get('keep', 'first')
                    current_df = operation_func(current_df, subset, keep)
                    
                else:
                    # Operations that work on the entire dataframe
                    current_df = operation_func(current_df, **parameters)
                
                # Record successful operation
                applied_operations.append({
                    'operation': operation_name,
                    'column': column,
                    'parameters': parameters,
                    'status': 'success'
                })
                
            except Exception as op_error:
                # Record failed operation
                applied_operations.append({
                    'operation': operation_name,
                    'column': column,
                    'parameters': parameters,
                    'status': 'failed',
                    'error': str(op_error)
                })
                raise ValueError(f"Operation {operation_name} failed: {str(op_error)}")
    
    except Exception as pipeline_error:
        # If any operation fails, you could implement rollback here
        # For now, we'll raise the error with context
        raise ValueError(f"Pipeline execution failed at operation {len(applied_operations)}: {str(pipeline_error)}")
    
    return current_df


def preview_cleaning_operation(df: pd.DataFrame, operation: Dict[str, Any]) -> pd.DataFrame:
    """
    Preview the effect of a cleaning operation without modifying the original data.
    
    Applies a single cleaning operation to a copy of the dataframe
    to allow users to see the impact before committing changes.
    
    Args:
        df (pd.DataFrame): Input dataframe
        operation (Dict[str, Any]): Operation dictionary containing:
                                   - 'operation': Function name to call
                                   - 'column': Target column (if applicable)
                                   - 'parameters': Dict of operation parameters
        
    Returns:
        pd.DataFrame: Preview dataframe showing the effect of the operation
        
    Raises:
        ValueError: If operation is invalid or parameters are incorrect
        
    Example:
        >>> operation = {
        ...     'operation': 'handle_missing_numeric',
        ...     'column': 'age',
        ...     'parameters': {'method': 'mean'}
        ... }
        >>> preview_df = preview_cleaning_operation(df, operation)
        >>> print(f"Before: {df['age'].isna().sum()}, After: {preview_df['age'].isna().sum()}")
        
    """
    # Use the pipeline execution with a single operation
    try:
        # Validate operation structure
        if not isinstance(operation, dict):
            raise ValueError("Operation must be a dictionary")
        
        if 'operation' not in operation:
            raise ValueError("Operation missing 'operation' key")
        
        # Create a single-operation pipeline
        preview_pipeline = [operation]
        
        # Apply the pipeline to a copy of the dataframe
        preview_df = apply_cleaning_pipeline(df, preview_pipeline)
        
        return preview_df
        
    except Exception as e:
        raise ValueError(f"Preview operation failed: {str(e)}")
