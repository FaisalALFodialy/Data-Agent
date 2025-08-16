
# Automated CSV Data Analysis and Cleaning Tool

A comprehensive Python-based Streamlit application that provides Kaggle-style usability scoring and automated data cleaning capabilities for CSV files. This tool empowers data scientists and analysts to quickly assess data quality and apply intelligent cleaning operations with an intuitive web interface.

![Data Analysis Tool](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## Project Overview

### Purpose
Transform raw CSV data into analysis-ready datasets through automated quality assessment and interactive cleaning workflows. The tool provides:

- **Kaggle-style Usability Scoring**: Comprehensive 0-100 data quality score
- **Interactive Data Cleaning**: Step-by-step cleaning with preview functionality  
- **Rich Visualizations**: Interactive charts and dashboards using Plotly
- **Export Capabilities**: Download cleaned data and detailed reports
- **Pipeline Management**: Save and reuse cleaning workflows

### Target Users
- Data Scientists and Analysts
- Junior Developers learning data cleaning
- Business Analysts working with CSV data
- Students studying data preprocessing

##  Features

###  Comprehensive Data Analysis
- **Dataset Overview**: Dimensions, memory usage, data types distribution
- **Missing Value Analysis**: Patterns, correlations, heatmaps
- **Statistical Profiling**: Descriptive statistics, distribution analysis
- **Quality Metrics**: Duplicate detection, outlier identification
- **Usability Scoring**: Composite 0-100 score based on data quality dimensions

### Interactive Data Cleaning
- **Missing Value Treatment**:
  - Numeric: mean, median, mode, interpolation, forward/backward fill
  - Categorical: mode, constant values, forward/backward fill
- **Outlier Handling**: IQR and Z-score detection with removal or capping
- **Duplicate Management**: Remove exact or partial duplicates
- **Data Type Optimization**: Memory-efficient type conversion
- **Text Cleaning**: Standardization, case conversion, special character handling

### Rich Visualizations
- Interactive missing value heatmaps
- Distribution plots (histogram, box, violin)
- Correlation matrices and heatmaps
- Before/after comparison charts
- Data quality dashboard with gauges

### Export and Pipeline Management
- Download cleaned CSV files
- Generate comprehensive cleaning reports
- Save/load cleaning pipelines for reuse
- Export analysis summaries

## Architecture

### Project Structure
```
data-analysis-tool/
├── main.py                 # Streamlit application entry point
├── src/
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration constants and settings
│   ├── data_analyzer.py    # Data profiling and analysis functions (13 functions)
│   ├── data_cleaner.py     # Data cleaning operations (10 functions)  
│   ├── visualizations.py  # Interactive plotting functions (10 functions)
│   └── utils.py           # Helper and utility functions (12 functions)
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── .gitignore            # Git ignore patterns
```

### Core Modules

#### `main.py` - Streamlit Application (7 Functions)
- `main()` - Application entry point with page configuration
- `setup_page_config()` - Streamlit page setup and styling
- `render_sidebar()` - File upload and navigation interface
- `render_analysis_tab()` - Data analysis dashboard
- `render_cleaning_tab()` - Interactive cleaning interface
- `render_export_tab()` - Export and download functionality
- `handle_file_upload()` - CSV processing with encoding detection

#### `src/data_analyzer.py` - Analysis Engine (13 Functions)
- `calculate_basic_stats()` - Dataset overview statistics
- `analyze_missing_values()` - Comprehensive missing data analysis
- `detect_data_types()` - Intelligent type detection and optimization
- `calculate_cardinality()` - Unique value analysis per column
- `detect_outliers_iqr()` - Interquartile range outlier detection
- `detect_outliers_zscore()` - Z-score outlier detection  
- `calculate_skewness_kurtosis()` - Distribution shape analysis
- `detect_duplicates()` - Duplicate row identification
- `calculate_correlation_matrix()` - Numeric variable correlations
- `generate_column_statistics()` - Detailed per-column analysis
- `calculate_usability_score()` - Composite quality scoring (0-100)
- `create_analysis_report()` - Comprehensive analysis orchestration

#### `src/data_cleaner.py` - Cleaning Engine (10 Functions)
- `handle_missing_numeric()` - Numeric missing value imputation
- `handle_missing_categorical()` - Categorical missing value handling
- `remove_outliers_iqr()` - IQR-based outlier removal
- `cap_outliers()` - Outlier capping at percentile boundaries
- `remove_duplicates()` - Duplicate row removal with options
- `optimize_dtypes()` - Memory-efficient data type conversion
- `clean_text_column()` - Text standardization operations
- `standardize_column_names()` - Column name normalization
- `apply_cleaning_pipeline()` - Sequential operation execution
- `preview_cleaning_operation()` - Change preview before applying

#### `src/visualizations.py` - Visualization Suite (10 Functions)
- `plot_missing_values_heatmap()` - Interactive missing data heatmap
- `plot_missing_values_bar()` - Missing value counts by column
- `plot_distribution()` - Distribution plots (histogram, box, violin)
- `plot_correlation_heatmap()` - Correlation matrix visualization
- `plot_outliers_boxplot()` - Multi-column outlier visualization
- `plot_data_types_pie()` - Data type distribution chart
- `plot_cardinality_bar()` - Unique value counts by column
- `plot_usability_gauge()` - Quality score gauge chart
- `plot_before_after_comparison()` - Cleaning impact visualization
- `create_data_quality_dashboard()` - Multi-metric dashboard

#### `src/utils.py` - Utility Functions (12 Functions)  
- `load_csv_file()` - Robust CSV loading with encoding detection
- `save_csv_file()` - Optimized CSV export
- `get_numeric_columns()` - Numeric column identification
- `get_categorical_columns()` - Categorical column identification
- `get_datetime_columns()` - DateTime column identification
- `format_bytes()` - Human-readable byte formatting
- `calculate_memory_usage()` - Memory usage analysis
- `validate_dataframe()` - Structure and content validation
- `create_download_link()` - Streamlit download link generation
- `generate_cleaning_report()` - Comprehensive cleaning documentation
- `save_pipeline_config()` - Pipeline configuration export (JSON)
- `load_pipeline_config()` - Pipeline configuration import

#### `src/config.py` - Configuration Management
- File handling settings (size limits, encodings)
- Data quality thresholds and parameters
- Cleaning method options and parameters
- Visualization settings and color palettes
- Streamlit session state management
- Performance and caching configuration

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run main.py
   ```

5. **Open in browser**:
   The application will automatically open at `http://localhost:8501`

## Usage Guide

### Getting Started
1. **Upload CSV File**: Use the sidebar file uploader (max 200MB)
2. **Automatic Analysis**: Tool performs comprehensive data profiling
3. **Review Quality Score**: Check the 0-100 usability score and grade
4. **Interactive Cleaning**: Apply cleaning operations with preview
5. **Export Results**: Download cleaned data and reports

### Data Analysis Workflow
1. **Data Overview**: Review dataset dimensions and memory usage
2. **Missing Values**: Examine patterns and percentages
3. **Quality Metrics**: Check duplicates, outliers, and inconsistencies
4. **Usability Score**: Understand overall data quality (A-F grade)
5. **Column Profiling**: Detailed statistics per column

### Data Cleaning Workflow
1. **Missing Value Treatment**: Choose appropriate imputation methods
2. **Outlier Handling**: Remove or cap extreme values
3. **Duplicate Removal**: Clean exact or partial duplicates
4. **Data Type Optimization**: Improve memory efficiency
5. **Pipeline Management**: Save workflows for reuse

### Export Options
- **Cleaned CSV**: Download processed dataset
- **Analysis Report**: Comprehensive quality assessment
- **Cleaning Report**: Documentation of operations performed
- **Pipeline Config**: JSON file for workflow reuse

## 🔧 Configuration

### Key Configuration Files

#### `src/config.py` - Main Configuration
```python
# File handling
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
UPLOAD_FILE_TYPES = ['csv']

# Quality thresholds  
MISSING_VALUE_THRESHOLD = 0.5      # 50%
HIGH_CARDINALITY_THRESHOLD = 0.9   # 90%
LOW_CARDINALITY_THRESHOLD = 0.05   # 5%

# Usability scoring weights
USABILITY_SCORE_WEIGHTS = {
    'completeness': 0.30,   # Missing values
    'consistency': 0.25,    # Data type consistency  
    'validity': 0.25,       # Outliers and invalid data
    'uniqueness': 0.20      # Duplicates
}
```

### Customization Options
- **Quality Thresholds**: Adjust missing value and cardinality thresholds
- **Scoring Weights**: Modify usability score calculation
- **Visualization Colors**: Customize chart color palettes
- **Performance Settings**: Configure memory and processing limits

## Development

### Code Quality Standards
- **PEP 8 Compliance**: All code follows Python style guidelines
- **Type Hints**: Comprehensive type annotations throughout
- **Documentation**: Detailed docstrings with examples and TODO comments
- **Error Handling**: Robust exception handling and validation
- **Testing Ready**: Structure prepared for unit test implementation

### Function Documentation Standard
Each function includes:
```python
def function_name(param: type) -> return_type:
    """
    Brief description of function purpose.
    
    Detailed explanation of what the function does and how it contributes
    to the overall system.
    
    Args:
        param (type): Description of parameter
        
    Returns:
        return_type: Description of return value
        
    Raises:
        ExceptionType: When this exception occurs
        
    Example:
        >>> result = function_name(example_input)
        >>> print(result)  # Expected output
        
    TODO: Implementation details:
          - Specific requirement 1
          - Specific requirement 2  
          - Edge case handling
    """
    pass
```

### Adding New Features
1. **Analysis Functions**: Add to `src/data_analyzer.py`
2. **Cleaning Operations**: Add to `src/data_cleaner.py`
3. **Visualizations**: Add to `src/visualizations.py`
4. **Utilities**: Add to `src/utils.py`
5. **Configuration**: Update `src/config.py`

## Data Quality Scoring

### Usability Score Calculation (0-100)
The tool calculates a composite quality score based on four dimensions:

**Completeness (30% weight)**
- Based on missing value percentages
- 100% = no missing values
- Decreases proportionally with missing data

**Consistency (25% weight)**  
- Data type uniformity and format consistency
- Column naming conventions
- Value format standardization

**Validity (25% weight)**
- Outlier detection and impact
- Data range validation
- Logical constraint compliance

**Uniqueness (20% weight)**
- Duplicate row detection
- Primary key uniqueness
- Data redundancy assessment

### Grade Assignment
- **A (90-100)**: Excellent - Ready for analysis
- **B (80-89)**: Good - Minor cleaning recommended  
- **C (70-79)**: Fair - Moderate cleaning needed
- **D (60-69)**: Poor - Significant cleaning required
- **F (0-59)**: Failing - Extensive cleaning needed

## Contributing

### For Team Members
1. **Function Implementation**: Choose functions from TODO lists
2. **Testing**: Write unit tests for implemented functions
3. **Documentation**: Update docstrings and examples
4. **Code Review**: Follow pull request process


### Development Workflow
1. **Choose a Function**: Pick from TODO comments in source files
2. **Implement**: Follow existing patterns and documentation standards
3. **Test**: Verify function works with sample data
4. **Document**: Update docstrings with implementation details
5. **Submit**: Create pull request with clear description

##  Implementation Status

### Phase 1: Skeleton Structure 
- [x] Project structure and organization
- [x] All 47 function stubs with comprehensive documentation
- [x] Configuration management system
- [x] Streamlit application framework
- [x] Import structure and dependencies

### Phase 2: Core Implementation (Next Steps)
- [ ] Data analysis functions (13 functions)
- [ ] Data cleaning operations (10 functions)
- [ ] Visualization components (10 functions)
- [ ] Utility functions (12 functions)
- [ ] Integration testing and bug fixes

### Phase 3: Advanced Features (Future)
- [ ] Machine learning-based outlier detection
- [ ] Advanced missing value imputation
- [ ] Custom visualization themes
- [ ] Automated report generation
- [ ] API endpoints for programmatic access

## Dependencies

### Core Dependencies
- **pandas** (≥2.0.3): Data manipulation and analysis
- **numpy** (≥1.24.3): Numerical computing
- **streamlit** (≥1.28.0): Web application framework
- **plotly** (≥5.17.0): Interactive visualizations
- **scikit-learn** (≥1.3.1): Machine learning utilities
- **scipy** (≥1.11.3): Scientific computing

### Development Dependencies  
- **pytest** (≥7.4.0): Unit testing framework
- **black** (≥23.7.0): Code formatting
- **flake8** (≥6.0.0): Linting
- **mypy** (≥1.5.0): Type checking

### Optional Dependencies
- **jupyter** (≥1.0.0): Development notebooks
- **great-expectations** (≥0.17.12): Advanced data validation
- **nltk** (≥3.8.1): Natural language processing
- **numba** (≥0.57.1): Performance optimization


##  Support

### Getting Help
- **Documentation**: Check function docstrings and TODO comments
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas

### Common Issues
- **Import Errors**: Ensure all dependencies are installed (`pip install -r requirements.txt`)
- **File Upload Issues**: Check file size (<200MB) and format (CSV only)
- **Performance**: For large files, consider sampling or chunking options

---


*This tool is designed to make data cleaning accessible, efficient, and comprehensive for data professionals at all levels.*
