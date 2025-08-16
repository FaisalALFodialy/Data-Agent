"""
Automated CSV Data Analysis and Cleaning Tool - Source Package

This package contains all the core modules for the CSV data analysis and cleaning tool:

- data_analyzer: Comprehensive data profiling and analysis functions
- data_cleaner: Interactive data cleaning and transformation operations
- visualizations: Interactive plotting and dashboard components
- utils: Helper functions and utilities
- config: Configuration constants and settings

Author: Data Science Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Data Science Team"

# Package metadata
__all__ = [
    "data_analyzer",
    "data_cleaner", 
    "visualizations",
    "utils",
    "config"
]

# Version information
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'stable'
}

def get_version() -> str:
    """
    Get the current package version.
    
    Returns:
        str: Version string in format 'major.minor.patch'
    """
    return f"{VERSION_INFO['major']}.{VERSION_INFO['minor']}.{VERSION_INFO['patch']}"

def get_package_info() -> dict:
    """
    Get comprehensive package information.
    
    Returns:
        dict: Package information including version, author, and modules
    """
    return {
        'name': 'CSV Data Analysis Tool',
        'version': get_version(),
        'author': __author__,
        'modules': __all__,
        'release_type': VERSION_INFO['release']
    }