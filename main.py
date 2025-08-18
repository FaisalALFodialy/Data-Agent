"""
Automated CSV Data Analysis and Cleaning Tool

Minimal entry point following Single Responsibility Principle.
Main application logic has been refactored into focused modules.

"""

import streamlit as st
import warnings
import logging

# Import refactored modules
from src.app_core import (
    initialize_session_state,
    setup_page_config,
    handle_file_upload,
    handle_critical_error,
    display_error_recovery
)
from src.ui_handlers import (
    render_sidebar,
    render_analysis_tab,
    render_cleaning_tab,
    render_export_tab,
    render_welcome_screen,
    render_data_preview
)
from src.ui_components import render_visualization_tab

# Configure logging and warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


def main() -> None:
    """Main application entry point with enhanced error handling."""
    try:
        # Initialize application
        setup_page_config()
        initialize_session_state()
        
        # Render sidebar and get settings
        sidebar_config = render_sidebar()
        
        # Handle file upload
        if sidebar_config['uploaded_file'] is not None:
            handle_file_upload(sidebar_config['uploaded_file'])

        # Main application interface - All tabs access session state directly
        if st.session_state.current_df is not None:
            # Quick data preview
            render_data_preview()
            
            # Main tabs with state persistence
            tab_names = ["📊 Analysis", "🧹 Cleaning", "📈 Visualizations", "📤 Export"]
            
            # Use session state to track active tab
            selected_tab = st.selectbox(
                "Select Section:", 
                options=tab_names,
                index=st.session_state.active_main_tab,
                key="main_tab_selector"
            )
            
            # Update session state when tab changes
            if selected_tab != tab_names[st.session_state.active_main_tab]:
                st.session_state.active_main_tab = tab_names.index(selected_tab)
            
            # Add some styling separation
            st.markdown("---")
            
            # Render content based on selected tab
            if selected_tab == "📊 Analysis":
                render_analysis_tab()
            elif selected_tab == "🧹 Cleaning":
                render_cleaning_tab()
            elif selected_tab == "📈 Visualizations":
                render_visualization_tab()
            elif selected_tab == "📤 Export":
                render_export_tab()
                
        else:
            render_welcome_screen()
            
        # Error recovery
        if st.session_state.get('error_state', False):
            display_error_recovery()
            
    except Exception as e:
        handle_critical_error(e)


if __name__ == "__main__":
    main()
