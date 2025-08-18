# ui_components.py
"""
UI Components Module

Reusable UI components for the Streamlit application following DRY and SOLID principles.
All UI rendering functions separated from business logic for maintainability.
"""

import streamlit as st
import pandas as pd

# Import modules

# Import business logic modules
from .data_cleaner import (
    preview_cleaning_operation,
    handle_missing_numeric,
    handle_missing_categorical,
    remove_outliers_iqr,
    cap_outliers,
    remove_duplicates,
    optimize_dtypes,
    clean_text_column
)
from .visualizations import (
    plot_missing_values_heatmap,
    plot_missing_values_bar,
    plot_distribution,
    plot_outliers_boxplot,
    plot_data_types_pie,
    plot_cardinality_bar,
    plot_before_after_comparison
)
from .utils import (
    calculate_memory_usage,
    get_numeric_columns,
    get_categorical_columns,
    format_bytes,
    generate_cleaning_report,
    safe_dataframe,
    get_current_dataframe,
    update_current_dataframe,
    validate_session_state,
    ensure_session_state_integrity
)
from .config import (
    MISSING_NUMERIC_METHODS,
    MISSING_CATEGORICAL_METHODS,
    TEXT_CLEANING_OPERATIONS,
    OUTLIER_IQR_MULTIPLIER
)


# ============================================================================
# VISUALIZATION TAB COMPONENT
# ============================================================================

def render_visualization_tab() -> None:
    """Render comprehensive visualization tab with interactive charts using session state."""
    try:
        # Ensure session state integrity before operations
        ensure_session_state_integrity()
        
        # Validate session state and get current dataframe
        if not validate_session_state():
            st.warning("No data available for visualization")
            return
            
        st.header("📊 Data Visualizations")
        
        # Visualization type selector
        viz_type = st.selectbox(
            "Select visualization type",
            [
                "Distribution Analysis",
                "Missing Values Analysis", 
                "Outlier Detection",
                "Data Types Overview",
                "Cardinality Analysis",
                "Before/After Comparison"
            ],
            key="viz_type_selector"
        )
        
        if viz_type == "Distribution Analysis":
            render_distribution_visualizations()
            
        elif viz_type == "Missing Values Analysis":
            render_missing_values_visualizations()
            
        elif viz_type == "Outlier Detection":
            render_outlier_visualizations()
            
        elif viz_type == "Data Types Overview":
            render_data_types_visualizations()
            
        elif viz_type == "Cardinality Analysis":
            render_cardinality_visualizations()
            
        elif viz_type == "Before/After Comparison":
            render_comparison_visualizations()
            
    except Exception as e:
        st.error(f"Error in visualization tab: {str(e)}")
        if st.session_state.get('advanced_mode', False):
            st.exception(e)


def render_distribution_visualizations() -> None:
    """Render distribution analysis visualizations."""
    st.subheader("📈 Distribution Analysis")
    
    try:
        df = get_current_dataframe()
    except ValueError:
        st.warning("No data available for distribution analysis")
        return
    
    # Column selector
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    all_cols = list(df.columns)
    
    selected_col = st.selectbox("Select column to analyze", all_cols, key="viz_column_selector")
    
    if selected_col:
        col1, col2 = st.columns(2)
        
        with col1:
            plot_type = st.selectbox(
                "Plot type",
                ["histogram", "box", "violin"] if selected_col in numeric_cols else ["histogram"],
                key="viz_plot_type_selector"
            )
        
        with col2:
            if plot_type == "histogram":
                bins = st.slider("Number of bins", 10, 100, 30, key="viz_histogram_bins")
        
        try:
            fig = plot_distribution(df, selected_col, plot_type)
            st.plotly_chart(fig, use_container_width=True, key="viz_tab_distribution_chart")
            
            # Show summary statistics
            if selected_col in numeric_cols:
                st.subheader("📊 Summary Statistics")
                stats = df[selected_col].describe()
                safe_dataframe(stats.to_frame().T, use_container_width=True)
            else:
                st.subheader("📊 Value Counts")
                value_counts = df[selected_col].value_counts().head(10)
                safe_dataframe(value_counts.to_frame(), use_container_width=True)
                
        except Exception as e:
            st.error(f"Could not render distribution plot: {str(e)}")


def render_missing_values_visualizations() -> None:
    """Render missing values visualizations."""
    st.subheader("🔍 Missing Values Analysis")
    
    try:
        df = get_current_dataframe()
    except ValueError:
        st.warning("No data available for missing values analysis")
        return
    
    missing_count = df.isnull().sum().sum()
    
    if missing_count > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                fig_heatmap = plot_missing_values_heatmap(df)
                st.plotly_chart(fig_heatmap, use_container_width=True, key="viz_tab_missing_heatmap")
            except Exception as e:
                st.warning(f"Could not render missing values heatmap: {str(e)}")
        
        with col2:
            try:
                fig_bar = plot_missing_values_bar(df)
                st.plotly_chart(fig_bar, use_container_width=True, key="viz_tab_missing_bar")
            except Exception as e:
                st.warning(f"Could not render missing values bar chart: {str(e)}")
        
        # Missing values summary table
        st.subheader("📋 Missing Values Summary")
        missing_summary = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
        }).query('`Missing Count` > 0').sort_values('Missing Count', ascending=False)
        
        safe_dataframe(missing_summary, use_container_width=True)
    else:
        st.success("✅ No missing values found in the dataset!")


def render_outlier_visualizations() -> None:
    """Render outlier detection visualizations."""
    st.subheader("🎯 Outlier Detection")
    
    try:
        df = get_current_dataframe()
    except ValueError:
        st.warning("No data available for outlier analysis")
        return
    
    numeric_cols = get_numeric_columns(df)
    
    if numeric_cols:
        selected_cols = st.multiselect(
            "Select columns for outlier analysis",
            numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))],
            key="viz_outlier_columns"
        )
        
        if selected_cols:
            try:
                fig = plot_outliers_boxplot(df, selected_cols)
                st.plotly_chart(fig, use_container_width=True, key="viz_tab_outliers_chart")
                
                # Outlier statistics
                st.subheader("📊 Outlier Statistics (IQR Method)")
                outlier_stats = []
                
                for col in selected_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    outlier_count = len(outliers)
                    outlier_pct = (outlier_count / len(df)) * 100
                    
                    outlier_stats.append({
                        'Column': col,
                        'Outlier Count': outlier_count,
                        'Outlier %': f"{outlier_pct:.2f}%",
                        'Lower Bound': f"{lower_bound:.2f}",
                        'Upper Bound': f"{upper_bound:.2f}"
                    })
                
                safe_dataframe(pd.DataFrame(outlier_stats), use_container_width=True)
                
            except Exception as e:
                st.error(f"Could not render outlier analysis: {str(e)}")
        else:
            st.info("Select columns to analyze outliers")
    else:
        st.info("No numeric columns available for outlier analysis")


def render_data_types_visualizations() -> None:
    """Render data types visualizations."""
    st.subheader("🗂️ Data Types Overview")
    
    try:
        df = get_current_dataframe()
    except ValueError:
        st.warning("No data available for data types analysis")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            fig = plot_data_types_pie(df)
            st.plotly_chart(fig, use_container_width=True, key="viz_tab_datatype_chart")
        except Exception as e:
            st.warning(f"Could not render data types chart: {str(e)}")
    
    with col2:
        # Data types summary table with safe conversion
        try:
            dtype_summary = pd.DataFrame({
                'Column': df.columns,
                'Data Type': [str(dt) for dt in df.dtypes],
                'Non-Null Count': df.count(),
                'Memory Usage': [format_bytes(df[col].memory_usage(deep=True)) for col in df.columns]
            })
            safe_dataframe(dtype_summary, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display data types summary: {str(e)}")
            # Fallback simple display
            simple_summary = pd.DataFrame({
                'Column': df.columns,
                'Data Type': [str(dt) for dt in df.dtypes]
            })
            safe_dataframe(simple_summary, use_container_width=True)


def render_cardinality_visualizations() -> None:
    """Render cardinality analysis visualizations."""
    st.subheader("🔢 Cardinality Analysis")
    
    try:
        df = get_current_dataframe()
    except ValueError:
        st.warning("No data available for cardinality analysis")
        return
    
    # Calculate cardinality for each column
    cardinality_data = {col: df[col].nunique() for col in df.columns}
    
    try:
        fig = plot_cardinality_bar(cardinality_data)
        st.plotly_chart(fig, use_container_width=True, key="viz_tab_cardinality_chart")
        
        # Cardinality categories
        st.subheader("📊 Cardinality Categories")
        cardinality_df = pd.DataFrame({
            'Column': list(cardinality_data.keys()),
            'Unique Values': list(cardinality_data.values()),
            'Cardinality %': [(v / len(df)) * 100 for v in cardinality_data.values()]
        })
        
        # Add category
        cardinality_df['Category'] = cardinality_df['Unique Values'].apply(
            lambda x: 'Low (≤10)' if x <= 10 else 'Medium (11-50)' if x <= 50 else 'High (>50)'
        )
        
        safe_dataframe(cardinality_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Could not render cardinality analysis: {str(e)}")


def render_comparison_visualizations() -> None:
    """Render before/after comparison visualizations."""
    st.subheader("⚖️ Before/After Comparison")
    
    try:
        df = get_current_dataframe()
    except ValueError:
        st.warning("No data available for comparison")
        return
    
    if st.session_state.get('original_df') is not None:
        original_df = st.session_state.original_df
        
        # Column selector
        common_cols = list(set(original_df.columns) & set(df.columns))
        
        if common_cols:
            selected_col = st.selectbox("Select column to compare", common_cols, key="viz_comparison_column")
            
            if selected_col:
                try:
                    fig = plot_before_after_comparison(original_df, df, selected_col)
                    st.plotly_chart(fig, use_container_width=True, key="viz_tab_comparison_chart")
                    
                    # Comparison statistics
                    st.subheader("📊 Comparison Statistics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Original Missing",
                            int(original_df[selected_col].isnull().sum()),
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            "Current Missing", 
                            int(df[selected_col].isnull().sum()),
                            delta=int(df[selected_col].isnull().sum() - original_df[selected_col].isnull().sum())
                        )
                    
                    with col3:
                        if pd.api.types.is_numeric_dtype(original_df[selected_col]):
                            orig_mean = float(original_df[selected_col].mean())
                            curr_mean = float(df[selected_col].mean())
                            st.metric(
                                "Mean Change",
                                f"{curr_mean:.2f}",
                                delta=f"{curr_mean - orig_mean:.2f}"
                            )
                        else:
                            orig_unique = int(original_df[selected_col].nunique())
                            curr_unique = int(df[selected_col].nunique())
                            st.metric(
                                "Unique Values",
                                curr_unique,
                                delta=curr_unique - orig_unique
                            )
                            
                except Exception as e:
                    st.error(f"Could not render comparison: {str(e)}")
            else:
                st.info("Select a column to compare")
        else:
            st.warning("No common columns found between original and current data")
    else:
        st.info("Upload original data to enable before/after comparison")


# ============================================================================
# CLEANING TAB COMPONENTS
# ============================================================================

def render_cleaning_recommendations() -> None:
    """Render intelligent cleaning recommendations."""
    try:
        # Ensure session state integrity before operations
        ensure_session_state_integrity()
        
        # Validate session state before proceeding
        if not validate_session_state():
            st.warning("No data available for recommendations")
            return
            
        df = get_current_dataframe()
    except ValueError:
        st.warning("No data available for recommendations")
        return
        
    st.subheader("🧠 Intelligent Cleaning Recommendations")
    
    recommendations = []
    
    # Missing values recommendations
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        for col in missing_cols[:3]:  # Top 3 columns with missing values
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 50:
                recommendations.append(f"🔴 High missing data in '{col}' ({missing_pct:.1f}%) - Consider dropping column")
            elif missing_pct > 20:
                recommendations.append(f"🟡 Moderate missing data in '{col}' ({missing_pct:.1f}%) - Consider imputation")
            else:
                recommendations.append(f"🟢 Low missing data in '{col}' ({missing_pct:.1f}%) - Safe for imputation")
    
    # Duplicates recommendations
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        dup_pct = (duplicates / len(df)) * 100
        recommendations.append(f"🔄 Found {duplicates} duplicate rows ({dup_pct:.1f}%) - Recommend removal")
    
    # Data types recommendations
    memory_info = calculate_memory_usage(df)
    if memory_info.get('optimization_potential', {}).get('percentage', 0) > 10:
        recommendations.append("💾 Data types can be optimized to save memory - Use 'Optimize Data Types'")
    
    # Outliers recommendations
    numeric_cols = get_numeric_columns(df)
    if numeric_cols:
        for col in numeric_cols[:2]:  # Check first 2 numeric columns
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            if len(outliers) > len(df) * 0.05:  # More than 5% outliers
                recommendations.append(f"📊 High outliers in '{col}' - Consider capping or removal")
    
    # Display recommendations
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    else:
        st.success("✅ Your data looks clean! No major issues detected.")








def render_text_cleaning() -> None:
    """Render text cleaning interface."""
    st.subheader("🔤 Text Data Cleaning")
    
    try:
        # Ensure session state integrity before operations
        ensure_session_state_integrity()
        
        # Validate session state before proceeding
        if not validate_session_state():
            st.warning("No data available for text cleaning")
            return
            
        # Always use the current dataframe from session state for operations
        df = get_current_dataframe()
    except ValueError:
        st.warning("No data available for text cleaning")
        return
    
    # Get text columns
    text_cols = [col for col in df.columns if df[col].dtype == 'object']
    
    if text_cols:
        selected_col = st.selectbox("Select text column to clean", text_cols, key="text_cleaning_column_selector")
        
        if selected_col:
            col1, col2 = st.columns(2)
            
            with col1:
                # Text cleaning operations
                st.markdown("**Select cleaning operations:**")
                operations = []
                
                for operation in TEXT_CLEANING_OPERATIONS:
                    if st.checkbox(operation.replace('_', ' ').title(), key=f"text_op_{operation}"):
                        operations.append(operation)
                
                # Store operations in session state for persistence
                st.session_state.text_operations = operations
                st.session_state.selected_text_col = selected_col
                
                st.info(f"Selected {len(operations)} operations")
            
            with col2:
                # Show sample data
                st.markdown("**Sample data (before cleaning):**")
                sample_data = df[selected_col].dropna().head(5)
                for i, text in enumerate(sample_data):
                    st.text(f"{i+1}. {str(text)[:100]}...")
            
            if operations:
                # Action buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("👁️ Preview Cleaning", key="preview_text_cleaning"):
                        try:
                            # Work directly with session state dataframe
                            current_df = get_current_dataframe()
                            preview_df = clean_text_column(current_df, selected_col, operations)
                            
                            st.markdown("**After cleaning preview:**")
                            sample_cleaned = preview_df[selected_col].dropna().head(5)
                            for i, text in enumerate(sample_cleaned):
                                st.text(f"{i+1}. {str(text)[:100]}...")
                            
                        except Exception as e:
                            st.error(f"Preview failed: {str(e)}")
                
                with col2:
                    if st.button("✅ Apply Cleaning", key="apply_text_cleaning"):
                        try:
                            # Get values from session state for reliability
                            current_df = get_current_dataframe()
                            selected_col = st.session_state.get('selected_text_col')
                            operations = st.session_state.get('text_operations', [])
                            
                            # Enhanced validation
                            if not selected_col:
                                st.error("❌ No column selected. Please select a text column to clean.")
                                return
                            if not operations:
                                st.error("❌ No operations selected. Please select at least one cleaning operation.")
                                return
                            if selected_col not in current_df.columns:
                                st.error(f"❌ Column '{selected_col}' not found in dataframe.")
                                return
                            
                            # Show progress indicator
                            with st.spinner(f"Applying {len(operations)} text operations to column '{selected_col}'..."):
                                cleaned_df = clean_text_column(current_df, selected_col, operations)
                            
                            # Validate that cleaning actually worked
                            if cleaned_df is None:
                                st.error("❌ Text cleaning operation returned no data")
                                return
                            
                            if cleaned_df.shape[0] == 0:
                                st.error("❌ Text cleaning operation resulted in empty dataframe")
                                return
                            
                            # Update session state with cleaned dataframe
                            operation_name = f"Clean text in '{selected_col}': {', '.join(operations)}"
                            update_current_dataframe(cleaned_df, operation_name)
                            
                            # Add to pipeline history
                            if 'cleaning_pipeline' not in st.session_state:
                                st.session_state.cleaning_pipeline = []
                            
                            operation = {
                                'operation': f"Clean text in '{selected_col}'",
                                'operations': operations,
                                'column': selected_col,
                                'timestamp': pd.Timestamp.now().strftime('%H:%M:%S'),
                                'operations_count': len(operations)
                            }
                            st.session_state.cleaning_pipeline.append(operation)
                            
                            # Show immediate success feedback
                            st.success(f"✅ Successfully applied {len(operations)} text operations to column '{selected_col}'!")
                            
                            # Show immediate results
                            with st.container():
                                st.markdown("**📊 Cleaning Results:**")
                                result_col1, result_col2 = st.columns(2)
                                
                                with result_col1:
                                    st.metric("Operations Applied", len(operations))
                                with result_col2:
                                    st.metric("Column", selected_col)
                                
                                # Show sample of cleaned data
                                st.markdown("**🔍 Sample of cleaned text:**")
                                sample_after = cleaned_df[selected_col].dropna().head(5)
                                for i, text in enumerate(sample_after):
                                    st.text(f"{i+1}. {str(text)[:100]}...")
                            
                        except Exception as e:
                            st.error(f"❌ Text cleaning failed: {str(e)}")
                            # Show detailed error in advanced mode
                            if st.session_state.get('advanced_mode', False):
                                st.exception(e)
                                st.markdown("**Debug Info:**")
                                st.json({
                                    'selected_col': st.session_state.get('selected_text_col'),
                                    'operations': st.session_state.get('text_operations', []),
                                    'df_shape': current_df.shape if 'current_df' in locals() else 'Unknown'
                                })
            else:
                st.info("Select at least one cleaning operation")
    else:
        st.info("No text columns found for cleaning")


def render_data_types_cleaning() -> None:
    """Render data types optimization interface."""
    st.subheader("🗂️ Data Types Optimization")
    
    try:
        # Ensure session state integrity before operations
        ensure_session_state_integrity()
        
        # Validate session state before proceeding
        if not validate_session_state():
            st.warning("No data available for data types cleaning")
            return
            
        # Always use the current dataframe from session state for operations
        df = get_current_dataframe()
    except ValueError:
        st.warning("No data available for data types cleaning")
        return
    
    # Show current memory usage
    memory_info = calculate_memory_usage(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Current Memory Usage:**")
        st.metric("Total Memory", str(memory_info['total']))
        
        optimization_potential = memory_info.get('optimization_potential', {})
        if optimization_potential.get('percentage', 0) > 0:
            st.metric(
                "Potential Savings", 
                str(optimization_potential['formatted']),
                delta=f"-{float(optimization_potential['percentage']):.1f}%"
            )
        else:
            st.info("Data types are already optimized")
    
    with col2:
        st.markdown("**Memory by Data Type:**")
        by_dtype = memory_info.get('by_dtype', {})
        if by_dtype:
            dtype_df = pd.DataFrame([
                {'Type': dtype, 'Memory': info['formatted'], 'Columns': info['columns_count']}
                for dtype, info in by_dtype.items()
            ])
            safe_dataframe(dtype_df, use_container_width=True)
    
    # Show detailed column information
    st.markdown("**Column Details:**")
    per_column = memory_info.get('per_column', {})
    if per_column:
        try:
            column_df = pd.DataFrame([
                {'Column': col, 'Current Type': str(df[col].dtype), 'Memory': info['formatted'], '%': info['percentage']}
                for col, info in per_column.items()
            ]).sort_values('Memory', ascending=False)
            safe_dataframe(column_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display column details: {str(e)}")
            # Simple fallback
            simple_df = pd.DataFrame([
                {'Column': col, 'Memory': info['formatted']}
                for col, info in per_column.items()
            ])
            safe_dataframe(simple_df, use_container_width=True)
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("👁️ Preview Optimization", key="preview_dtype_optimization"):
            try:
                # Work directly with session state dataframe
                current_df = get_current_dataframe()
                optimized_df = optimize_dtypes(current_df)
                new_memory_info = calculate_memory_usage(optimized_df)
                
                memory_saved = memory_info['bytes'] - new_memory_info['bytes']
                savings_pct = (memory_saved / memory_info['bytes']) * 100
                
                st.info(f"Memory would change: {memory_info['total']} → {new_memory_info['total']}")
                st.success(f"Savings: {format_bytes(memory_saved)} ({savings_pct:.1f}%)")
                
                # Show type changes
                type_changes = []
                for col in df.columns:
                    old_type = df[col].dtype
                    new_type = optimized_df[col].dtype
                    if old_type != new_type:
                        type_changes.append({
                            'Column': col,
                            'From': str(old_type),
                            'To': str(new_type)
                        })
                
                if type_changes:
                    st.markdown("**Type Changes:**")
                    safe_dataframe(pd.DataFrame(type_changes), use_container_width=True)
                else:
                    st.info("No type changes needed")
                
            except Exception as e:
                st.error(f"Preview failed: {str(e)}")
    
    with col2:
        if st.button("✅ Apply Optimization", key="apply_dtype_optimization"):
            try:
                # Work directly with session state dataframe
                current_df = get_current_dataframe()
                original_memory = memory_info['bytes']
                optimized_df = optimize_dtypes(current_df)
                new_memory = calculate_memory_usage(optimized_df)['bytes']
                
                update_current_dataframe(optimized_df, "Data types optimization")
                
                # Add to pipeline history
                if 'cleaning_pipeline' not in st.session_state:
                    st.session_state.cleaning_pipeline = []
                
                memory_saved = original_memory - new_memory
                savings_pct = (memory_saved / original_memory) * 100
                
                operation = {
                    'operation': "Optimize data types",
                    'memory_saved': format_bytes(memory_saved),
                    'savings_pct': f"{savings_pct:.1f}%",
                    'timestamp': pd.Timestamp.now().strftime('%H:%M:%S')
                }
                st.session_state.cleaning_pipeline.append(operation)
                
                st.success(f"✅ Optimized data types! Saved {format_bytes(memory_saved)} ({savings_pct:.1f}%)")
                
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")


def render_cleaning_pipeline_management() -> None:
    """Render cleaning pipeline management interface."""
    st.subheader("⚙️ Cleaning Pipeline Management")
    
    # Show current pipeline
    pipeline = st.session_state.get('cleaning_pipeline', [])
    
    if pipeline:
        st.markdown("**Applied Operations:**")
        for i, operation in enumerate(pipeline, 1):
            with st.expander(f"{i}. {operation.get('operation', 'Unknown Operation')} - {operation.get('timestamp', '')}"):
                operation_details = {k: v for k, v in operation.items() if k not in ['operation', 'timestamp']}
                if operation_details:
                    st.json(operation_details)
                else:
                    st.info("No additional details")
        
        # Pipeline actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Pipeline"):
                try:
                    current_df = get_current_dataframe()
                    pipeline_config = {
                        'metadata': {
                            'created_at': pd.Timestamp.now().isoformat(),
                            'operations_count': len(pipeline),
                            'original_shape': st.session_state.get('original_df', current_df).shape,
                            'final_shape': current_df.shape
                        },
                        'pipeline': pipeline
                    }
                    
                    import json
                    config_str = json.dumps(pipeline_config, indent=2, default=str)
                    st.download_button(
                        label="📥 Download Pipeline Config",
                        data=config_str,
                        file_name=f"cleaning_pipeline_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                except Exception as e:
                    st.error(f"Failed to save pipeline: {str(e)}")
        
        with col2:
            # Split reset section into two columns for undo and reset
            reset_col1, reset_col2 = st.columns(2)
            
            with reset_col1:
                if st.button("🔙 Undo Last", use_container_width=True):
                    if st.session_state.get('operation_history') and len(st.session_state.operation_history) > 0:
                        # Get the last operation
                        last_operation = st.session_state.operation_history[-1]
                        
                        # Restore the dataframe to before the last operation
                        if 'before_df' in last_operation:
                            st.session_state.current_df = last_operation['before_df'].copy()
                            
                            # Remove the last operation from history
                            st.session_state.operation_history.pop()
                            
                            # Also remove from cleaning pipeline if exists
                            if st.session_state.get('cleaning_pipeline'):
                                st.session_state.cleaning_pipeline.pop()
                            
                            # Invalidate analysis results
                            st.session_state.analysis_results = None
                            st.session_state.usability_score = None
                            
                            st.success(f"✅ Undone: {last_operation.get('operation', 'Last operation')}")
                        else:
                            st.error("Cannot undo: No backup data found for last operation")
                    else:
                        st.error("No operations to undo")
            
            with reset_col2:
                if st.button("🔄 Reset to Original", use_container_width=True):
                    if st.session_state.get('original_df') is not None:
                        st.session_state.current_df = st.session_state.original_df.copy()
                        st.session_state.cleaning_pipeline = []
                        st.session_state.operation_history = []
                        # Invalidate analysis results since data changed
                        st.session_state.analysis_results = None
                        st.session_state.usability_score = None
                        st.success("✅ Reset to original data!")
                    else:
                        st.error("Original data not available")
        
        with col3:
            if st.button("🗑️ Clear Pipeline"):
                st.session_state.cleaning_pipeline = []
                st.success("✅ Pipeline cleared!")
        
        # Cleaning report generation
        st.markdown("---")
        st.subheader("📋 Generate Cleaning Report")
        
        if st.button("📊 Generate Report"):
            try:
                if st.session_state.get('original_df') is not None:
                    current_df = get_current_dataframe()
                    report = generate_cleaning_report(
                        st.session_state.original_df, 
                        current_df, 
                        pipeline
                    )
                    
                    st.markdown("**Cleaning Report:**")
                    st.text(report)
                    
                    # Download button for report
                    st.download_button(
                        label="📥 Download Report",
                        data=report,
                        file_name=f"cleaning_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                else:
                    st.error("Original data not available for comparison")
                    
            except Exception as e:
                st.error(f"Failed to generate report: {str(e)}")
    else:
        st.info("No cleaning operations have been applied yet. Start cleaning your data using the tabs above!")
        
        # Load pipeline option
        st.markdown("---")
        st.subheader("📂 Load Existing Pipeline")
        
        uploaded_config = st.file_uploader(
            "Upload pipeline configuration",
            type=['json'],
            help="Upload a previously saved pipeline configuration file",
            key="pipeline_config_uploader"
        )
        
        if uploaded_config is not None:
            try:
                import json
                config_data = json.load(uploaded_config)
                
                if 'pipeline' in config_data:
                    st.session_state.cleaning_pipeline = config_data['pipeline']
                    # Invalidate analysis results since pipeline changed
                    st.session_state.analysis_results = None
                    st.session_state.usability_score = None
                    st.success("✅ Pipeline configuration loaded!")
                else:
                    st.error("Invalid pipeline configuration file")
                    
            except Exception as e:
                st.error(f"Failed to load pipeline: {str(e)}")


# ============================================================================
# EXPORT TAB COMPONENTS  
# ============================================================================

def render_analysis_report_download() -> None:
    """Render analysis report download functionality."""
    try:
        if st.session_state.get('analysis_results'):
            analysis_results = st.session_state.analysis_results
            
            # Convert analysis results to readable format
            current_df = get_current_dataframe()
            report_content = f"""# Data Analysis Report
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- **Shape**: {current_df.shape[0]:,} rows × {current_df.shape[1]:,} columns
- **Memory Usage**: {calculate_memory_usage(current_df)['total']}
- **Data Quality Score**: {st.session_state.get('usability_score', {}).get('overall_score', 'N/A')}/100

## Analysis Results
"""
            
            # Add basic statistics
            if 'basic_stats' in analysis_results:
                report_content += "\n### Basic Statistics\n"
                basic_stats = analysis_results['basic_stats']
                for key, value in basic_stats.items():
                    report_content += f"- **{key}**: {value}\n"
            
            # Add missing values info
            if 'missing_values' in analysis_results:
                missing_info = analysis_results['missing_values']
                report_content += f"\n### Missing Values\n"
                report_content += f"- **Total Missing**: {missing_info.get('total_missing', 0):,}\n"
                report_content += f"- **Missing Percentage**: {missing_info.get('missing_percentage', 0):.2f}%\n"
            
            # Add data types info
            if 'data_types' in analysis_results:
                report_content += "\n### Data Types\n"
                dtypes_info = analysis_results['data_types']
                if hasattr(dtypes_info, 'to_dict'):
                    for dtype, count in dtypes_info.value_counts().items():
                        report_content += f"- **{dtype}**: {count} columns\n"
            
            st.download_button(
                label="📥 Download Analysis Report",
                data=report_content,
                file_name=f"analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
            
            st.success("✅ Analysis report ready for download!")
        else:
            st.error("No analysis results available. Please run analysis first.")
            
    except Exception as e:
        st.error(f"Failed to generate analysis report: {str(e)}")


def render_cleaning_report_download() -> None:
    """Render cleaning report download functionality."""
    try:
        if st.session_state.get('original_df') is not None and st.session_state.get('cleaning_pipeline'):
            original_df = st.session_state.original_df
            current_df = get_current_dataframe()
            pipeline = st.session_state.cleaning_pipeline
            
            report = generate_cleaning_report(original_df, current_df, pipeline)
            
            st.download_button(
                label="📥 Download Cleaning Report",
                data=report,
                file_name=f"cleaning_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
            
            st.success("✅ Cleaning report ready for download!")
        else:
            st.error("No cleaning operations have been performed or original data is not available.")
            
    except Exception as e:
        st.error(f"Failed to generate cleaning report: {str(e)}")


def render_pipeline_config_download() -> None:
    """Render pipeline configuration download functionality."""
    try:
        pipeline = st.session_state.get('cleaning_pipeline', [])
        
        if pipeline:
            pipeline_config = {
                'metadata': {
                    'created_at': pd.Timestamp.now().isoformat(),
                    'version': '1.0.0',
                    'operations_count': len(pipeline),
                    'original_shape': getattr(st.session_state.get('original_df'), 'shape', None),
                    'final_shape': getattr(st.session_state.get('current_df'), 'shape', None)
                },
                'pipeline': pipeline
            }
            
            import json
            config_str = json.dumps(pipeline_config, indent=2, default=str)
            
            st.download_button(
                label="📥 Download Pipeline Config",
                data=config_str,
                file_name=f"cleaning_pipeline_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            st.success("✅ Pipeline configuration ready for download!")
        else:
            st.error("No cleaning pipeline to export. Please perform some cleaning operations first.")
            
    except Exception as e:
        st.error(f"Failed to generate pipeline config: {str(e)}")


def render_complete_package_export() -> None:
    """Render complete package export functionality."""
    try:
        import zipfile
        import io
        import json
        
        # Get current dataframe
        df = get_current_dataframe()
        
        # Create a BytesIO object to hold the zip file
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add cleaned data
            csv_data = df.to_csv(index=False)
            zip_file.writestr('cleaned_data.csv', csv_data)
            
            # Add analysis report if available
            if st.session_state.get('analysis_results'):
                analysis_data = json.dumps(st.session_state.analysis_results, indent=2, default=str)
                zip_file.writestr('analysis_results.json', analysis_data)
            
            # Add cleaning report if available
            if st.session_state.get('original_df') is not None and st.session_state.get('cleaning_pipeline'):
                cleaning_report = generate_cleaning_report(
                    st.session_state.original_df, 
                    df, 
                    st.session_state.cleaning_pipeline
                )
                zip_file.writestr('cleaning_report.md', cleaning_report)
            
            # Add pipeline config if available
            pipeline = st.session_state.get('cleaning_pipeline', [])
            if pipeline:
                pipeline_config = {
                    'metadata': {
                        'created_at': pd.Timestamp.now().isoformat(),
                        'version': '1.0.0',
                        'operations_count': len(pipeline)
                    },
                    'pipeline': pipeline
                }
                config_str = json.dumps(pipeline_config, indent=2, default=str)
                zip_file.writestr('cleaning_pipeline.json', config_str)
            
            # Add summary file
            summary = f"""# Export Summary
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Files Included:
- `cleaned_data.csv`: The cleaned dataset
- `analysis_results.json`: Complete analysis results (if available)
- `cleaning_report.md`: Human-readable cleaning report (if available)
- `cleaning_pipeline.json`: Pipeline configuration for reproducibility (if available)

## Dataset Information:
- Rows: {df.shape[0]:,}
- Columns: {df.shape[1]:,}
- Memory Usage: {calculate_memory_usage(df)['total']}
- Quality Score: {st.session_state.get('usability_score', {}).get('overall_score', 'N/A')}/100

Generated by Byan Data Analysis & Cleaning Tool
"""
            zip_file.writestr('README.md', summary)
        
        zip_buffer.seek(0)
        
        st.download_button(
            label="📦 Download Complete Package",
            data=zip_buffer.getvalue(),
            file_name=f"data_analysis_package_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip"
        )
        
        st.success("✅ Complete package ready for download!")
        
    except Exception as e:
        st.error(f"Failed to create export package: {str(e)}")


# ============================================================================
# REUSABLE UI BUILDER FUNCTIONS (Phase 1: Additive Only)
# ============================================================================

def render_cleaning_config_section(config_col1_content, config_col2_content):
    """
    Render configuration section with two columns for cleaning operations.
    
    Args:
        config_col1_content: Function to render content for left column
        config_col2_content: Function to render content for right column
    """
    st.markdown("**📋 Configuration**")
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        config_col1_content()
    
    with config_col2:
        config_col2_content()


def render_current_data_section(data_sample, title="📊 Current Data", subtitle=None):
    """
    Render current data preview section.
    
    Args:
        data_sample: Data sample to display (pandas DataFrame or Series)
        title: Section title
        subtitle: Optional subtitle/description
    """
    st.markdown(f"**{title}**")
    if subtitle:
        st.markdown(f"*{subtitle}*")
    safe_dataframe(data_sample.to_frame() if hasattr(data_sample, 'to_frame') else data_sample, 
                   use_container_width=True)


def render_action_buttons_section(preview_btn_config, apply_btn_config):
    """
    Render action buttons section with preview and apply buttons side by side.
    
    Args:
        preview_btn_config: Dict with 'text', 'key', 'callback' for preview button
        apply_btn_config: Dict with 'text', 'key', 'callback' for apply button
    """
    st.markdown("**🎯 Actions**")
    
    button_col1, button_col2 = st.columns(2)
    
    with button_col1:
        if st.button(preview_btn_config['text'], key=preview_btn_config['key'], use_container_width=True):
            preview_btn_config['callback']()
    
    with button_col2:
        if st.button(apply_btn_config['text'], key=apply_btn_config['key'], use_container_width=True):
            apply_btn_config['callback']()


def render_preview_results_section(preview_data, metrics_data, success_flag_key):
    """
    Render preview results section with data and metrics.
    
    Args:
        preview_data: Data to display in preview
        metrics_data: Dict containing metrics to display
        success_flag_key: Session state key for preview success flag
    """
    if (success_flag_key in st.session_state and 
        st.session_state[success_flag_key] and 
        st.session_state[success_flag_key].get('preview_success', False)):
        
        st.markdown("---")
        st.markdown("**🔮 Preview Results**")
        
        # Display preview data
        st.markdown("*✨ After cleaning preview (first 5 rows):*")
        display_data = preview_data.head(5) if hasattr(preview_data, 'head') else preview_data
        safe_dataframe(display_data.to_frame() if hasattr(display_data, 'to_frame') else display_data, 
                       use_container_width=True)
        
        # Display preview metrics
        st.markdown("**📊 Preview Impact:**")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric(metrics_data.get('metric1_label', 'Before'), 
                     metrics_data.get('metric1_value', 'N/A'))
        with metric_col2:
            st.metric(metrics_data.get('metric2_label', 'After'), 
                     metrics_data.get('metric2_value', 'N/A'), 
                     delta=metrics_data.get('metric2_delta'))
        with metric_col3:
            st.metric(metrics_data.get('metric3_label', 'Impact'), 
                     metrics_data.get('metric3_value', 'N/A'), 
                     delta_color=metrics_data.get('metric3_color', 'normal'))


def render_applied_changes_section(before_data, after_data, labels_dict):
    """
    Render applied changes section with side-by-side before/after comparison.
    
    Args:
        before_data: Data to display in before column (pandas DataFrame or Series)
        after_data: Data to display in after column (pandas DataFrame or Series) 
        labels_dict: Dict containing 'before_label' and 'after_label' for column headers
    """
    st.markdown("**🎯 Applied Changes:**")
    
    # Break out of any container constraints using full width container pattern
    with st.container():
        # Force full width by creating single-column layout first
        full_width = st.columns([1])
        with full_width[0]:
            # Then create side-by-side columns within the full width
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"**{labels_dict.get('before_label', 'Before')}:**")
                if hasattr(before_data, 'to_frame'):
                    st.dataframe(before_data.to_frame(), use_container_width=True)
                else:
                    st.dataframe(before_data, use_container_width=True)
            with col2:
                st.markdown(f"**{labels_dict.get('after_label', 'After')}:**")
                if hasattr(after_data, 'to_frame'):
                    st.dataframe(after_data.to_frame(), use_container_width=True)
                else:
                    st.dataframe(after_data, use_container_width=True)


def render_column_info_line(column_name, info_dict):
    """
    Render a formatted information line about a column.
    
    Args:
        column_name: Name of the column
        info_dict: Dictionary containing column information to display
    """
    info_parts = [f"**Column:** {column_name}"]
    
    # Add each piece of info if available
    for key, value in info_dict.items():
        if value is not None:
            info_parts.append(f"**{key}:** {value}")
    
    st.markdown(" | ".join(info_parts))


def clear_preview_on_change(preview_key, current_value, session_key):
    """
    Clear preview results when a configuration value changes.
    
    Args:
        preview_key: Session state key for preview results
        current_value: Current value of the configuration
        session_key: Session state key to track the value
    """
    if current_value != st.session_state.get(session_key):
        if preview_key in st.session_state:
            del st.session_state[preview_key]
        st.session_state[session_key] = current_value


def handle_operation_error(error, operation_name, show_debug=False):
    """
    Handle and display operation errors with consistent formatting.
    
    Args:
        error: The exception that occurred
        operation_name: Name of the operation that failed
        show_debug: Whether to show debug information
    """
    st.error(f"❌ {operation_name} failed: {str(error)}")
    
    if show_debug and st.session_state.get('advanced_mode', False):
        import traceback
        st.exception(error)
        st.code(traceback.format_exc())


# ============================================================================
# PHASE 2: PARALLEL IMPLEMENTATION TESTING
# ============================================================================

def render_missing_values_cleaning() -> None:
    """
    Missing values cleaning using modular UI builders.
    """
    st.subheader("🔍 Missing Values Treatment (New Version)")
    
    try:
        # Ensure session state integrity before operations
        ensure_session_state_integrity()
        
        # Validate session state before proceeding
        if not validate_session_state():
            st.warning("No data available for missing values cleaning")
            return
            
        # Work directly with session state
        current_df = get_current_dataframe()
        missing_cols = current_df.columns[current_df.isnull().any()].tolist()
    except ValueError:
        st.warning("No data available for missing values cleaning")
        return
    
    if missing_cols:
        # ROW 1: Configuration using new builder
        def config_col1():
            selected_col = st.selectbox("Select column to clean", missing_cols, key="missing_values_column_selector_v2")
            # Clear preview results if column changes
            clear_preview_on_change('preview_results_v2', selected_col, 'selected_missing_col_v2')
            return selected_col
        
        def config_col2():
            selected_col = st.session_state.get('selected_missing_col_v2')
            if selected_col:
                # Method selection with session state persistence
                if pd.api.types.is_numeric_dtype(current_df[selected_col]):
                    method = st.selectbox("Cleaning method", MISSING_NUMERIC_METHODS, key="missing_numeric_method_v2")
                else:
                    method = st.selectbox("Cleaning method", MISSING_CATEGORICAL_METHODS, key="missing_categorical_method_v2")
                
                # Clear preview results if method changes
                clear_preview_on_change('preview_results_v2', method, 'missing_method_v2')
                return method
            return None
        
        # Use the new builder for configuration section
        render_cleaning_config_section(config_col1, config_col2)
        
        selected_col = st.session_state.get('selected_missing_col_v2')
        method = st.session_state.get('missing_method_v2')
        
        if selected_col and method:
            # Custom value input if needed
            custom_value = None
            if method == 'constant':
                if pd.api.types.is_numeric_dtype(current_df[selected_col]):
                    custom_value = st.number_input("Custom value", value=0.0, key="missing_numeric_custom_value_v2")
                else:
                    custom_value = st.text_input("Custom value", value="Unknown", key="missing_categorical_custom_value_v2")
                st.session_state.missing_custom_value_v2 = custom_value
            else:
                st.session_state.missing_custom_value_v2 = None
            
            # Column information using new builder
            missing_count = current_df[selected_col].isnull().sum()
            missing_pct = (missing_count / len(current_df)) * 100
            info_dict = {
                "Missing Count": f"{missing_count:,}",
                "Missing %": f"{missing_pct:.2f}%", 
                "Data Type": str(current_df[selected_col].dtype)
            }
            render_column_info_line(selected_col, info_dict)
            
            st.markdown("---")
            
            # ROW 2: Current Data using new builder
            sample_data = current_df[selected_col].head(5)
            render_current_data_section(sample_data, 
                                      title="📊 Current Data",
                                      subtitle="🔍 Sample from selected column (first 5 rows):")
            
            st.markdown("---")
            
            # ROW 3: Action Buttons using new builder
            def preview_callback():
                try:
                    # Get values from session state for reliability
                    current_df = get_current_dataframe()
                    selected_col = st.session_state.get('selected_missing_col_v2')
                    method = st.session_state.get('missing_method_v2')
                    custom_value = st.session_state.get('missing_custom_value_v2')
                    
                    if not selected_col or not method:
                        st.error("Please select a column and method before previewing")
                        return
                    
                    # Clear any existing preview results to prevent stale data
                    if 'preview_results_v2' in st.session_state:
                        del st.session_state.preview_results_v2
                    
                    # Perform preview operation
                    with st.spinner("Generating preview..."):
                        operation = {
                            'operation': 'handle_missing_numeric' if pd.api.types.is_numeric_dtype(current_df[selected_col]) else 'handle_missing_categorical',
                            'column': selected_col,
                            'parameters': {'method': method, 'custom_value': custom_value} if custom_value is not None else {'method': method}
                        }
                        preview_df = preview_cleaning_operation(current_df, operation)
                        
                        # Store preview results in session state
                        st.session_state.preview_results_v2 = {
                            'data': preview_df[selected_col],
                            'before_missing': current_df[selected_col].isnull().sum(),
                            'after_missing': preview_df[selected_col].isnull().sum(),
                            'preview_success': True
                        }
                    
                    # Force rerun to show preview results immediately
                    st.rerun()
                    
                except Exception as e:
                    handle_operation_error(e, "Preview", st.session_state.get('advanced_mode', False))
            
            def apply_callback():
                try:
                    # Get values from session state for reliability
                    current_df = get_current_dataframe()
                    selected_col = st.session_state.get('selected_missing_col_v2')
                    method = st.session_state.get('missing_method_v2')
                    custom_value = st.session_state.get('missing_custom_value_v2')
                    
                    # Enhanced validation
                    if not selected_col:
                        st.error("❌ No column selected. Please select a column to clean.")
                        return
                    if not method:
                        st.error("❌ No method selected. Please select a cleaning method.")
                        return
                    if selected_col not in current_df.columns:
                        st.error(f"❌ Column '{selected_col}' not found in dataframe.")
                        return
                    
                    # Show progress indicator
                    with st.spinner(f"Applying {method} to column '{selected_col}'..."):
                        # Apply cleaning based on data type
                        if pd.api.types.is_numeric_dtype(current_df[selected_col]):
                            if custom_value is not None:
                                cleaned_df = handle_missing_numeric(current_df, selected_col, method, custom_value)
                            else:
                                cleaned_df = handle_missing_numeric(current_df, selected_col, method)
                        else:
                            if custom_value is not None:
                                cleaned_df = handle_missing_categorical(current_df, selected_col, method, custom_value)
                            else:
                                cleaned_df = handle_missing_categorical(current_df, selected_col, method)
                    
                    # Validate that cleaning actually worked
                    if cleaned_df is None:
                        st.error("❌ Cleaning operation returned no data")
                        return
                    
                    if cleaned_df.shape[0] == 0:
                        st.error("❌ Cleaning operation resulted in empty dataframe")
                        return
                    
                    # Calculate before/after statistics 
                    before_missing = int(current_df[selected_col].isnull().sum())
                    after_missing = int(cleaned_df[selected_col].isnull().sum())
                    missing_fixed = before_missing - after_missing
                    
                    # Store before data for comparison
                    before_sample = current_df[selected_col].head(5)
                    after_sample = cleaned_df[selected_col].head(5)
                    
                    # Update session state with cleaned dataframe using centralized function
                    operation_name = f"Missing values: {method} on '{selected_col}'"
                    update_current_dataframe(cleaned_df, operation_name)
                    
                    # Add to pipeline history
                    if 'cleaning_pipeline' not in st.session_state:
                        st.session_state.cleaning_pipeline = []
                    
                    operation = {
                        'operation': f"Handle missing values in '{selected_col}'",
                        'method': method,
                        'column': selected_col,
                        'timestamp': pd.Timestamp.now().strftime('%H:%M:%S'),
                        'before_missing': before_missing,
                        'after_missing': after_missing,
                        'rows_before': len(current_df),
                        'rows_after': len(cleaned_df)
                    }
                    st.session_state.cleaning_pipeline.append(operation)
                    
                    # Show immediate success feedback
                    st.success(f"✅ Successfully applied {method} to column '{selected_col}'!")
                    
                    # Show comprehensive before/after results using new builder
                    metrics = {
                        'metric1_value': f"{before_missing} missing",
                        'metric2_value': f"{after_missing} missing", 
                        'metric2_delta': f"{after_missing - before_missing}",
                        'metric3_value': f"{missing_fixed} values" if missing_fixed > 0 else "No change",
                        'metric3_color': "normal" if missing_fixed > 0 else "off",
                        'shape_info': f"{len(current_df):,} → {len(cleaned_df):,} rows"
                    }
                    
                    labels = {
                        'before_label': "Before cleaning",
                        'after_label': "After cleaning",
                        'metric1_label': "Before",
                        'metric1_help': "Missing values before cleaning",
                        'metric2_label': "After",
                        'metric3_label': "Fixed" if missing_fixed > 0 else "Change"
                    }
                    
                    # Show comprehensive before/after results using modular builder
                    labels = {
                        'before_label': 'Before cleaning',
                        'after_label': 'After cleaning'
                    }
                    render_applied_changes_section(before_sample, after_sample, labels)
                    
                    # Metrics section
                    st.markdown("**📊 Impact Summary:**")
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        st.metric("Before", f"{before_missing} missing", help="Missing values before cleaning")
                    with result_col2:
                        st.metric("After", f"{after_missing} missing", delta=f"{after_missing - before_missing}")
                    with result_col3:
                        if missing_fixed > 0:
                            st.metric("Fixed", f"{missing_fixed} values", delta_color="normal")
                        else:
                            st.metric("Change", "No change", delta_color="off")
                    
                    # Show data shape changes
                    st.markdown(f"**Data shape:** {len(current_df):,} → {len(cleaned_df):,} rows")
                    
                    # Clear preview results since we've applied the changes
                    if 'preview_results_v2' in st.session_state:
                        del st.session_state.preview_results_v2
                    
                except Exception as e:
                    handle_operation_error(e, "Cleaning", st.session_state.get('advanced_mode', False))
            
            preview_btn_config = {
                'text': "👁️ Preview Cleaning",
                'key': "preview_missing_values_v2", 
                'callback': preview_callback
            }
            
            apply_btn_config = {
                'text': "✅ Apply Cleaning",
                'key': "apply_missing_values_v2",
                'callback': apply_callback
            }
            
            render_action_buttons_section(preview_btn_config, apply_btn_config)
            
            # ROW 4: Preview Results using new builder
            if 'preview_results_v2' in st.session_state:
                preview_data = st.session_state.preview_results_v2['data']
                before_missing = int(st.session_state.preview_results_v2['before_missing'])
                after_missing = int(st.session_state.preview_results_v2['after_missing'])
                
                metrics_data = {
                    'metric1_label': "Before",
                    'metric1_value': f"{before_missing} missing",
                    'metric2_label': "After", 
                    'metric2_value': f"{after_missing} missing",
                    'metric2_delta': f"{after_missing - before_missing}",
                    'metric3_label': "Would Fix" if after_missing < before_missing else "Change",
                    'metric3_value': f"{before_missing - after_missing} values" if after_missing < before_missing else "No improvement",
                    'metric3_color': "normal" if after_missing < before_missing else "off"
                }
                
                render_preview_results_section(preview_data, metrics_data, 'preview_results_v2')
    else:
        st.success("✅ No missing values found!")


def render_outliers_cleaning() -> None:
    """
    Outliers cleaning using modular UI builders.
    """
    st.subheader("🎯 Outlier Treatment")
    
    try:
        # Ensure session state integrity before operations
        ensure_session_state_integrity()
        
        # Validate session state before proceeding
        if not validate_session_state():
            st.warning("No data available for outlier cleaning")
            return
            
        # Always use the current dataframe from session state for operations
        current_df = get_current_dataframe()
    except ValueError:
        st.warning("No data available for outlier cleaning")
        return
    
    numeric_cols = get_numeric_columns(current_df)
    
    if numeric_cols:
        # ROW 1: Configuration using new builder
        def config_col1():
            selected_col = st.selectbox("Select column for outlier treatment", numeric_cols, key="outlier_column_selector_v2")
            # Clear preview results if column changes
            clear_preview_on_change('outlier_preview_results_v2', selected_col, 'selected_outlier_col_v2')
            return selected_col
        
        def config_col2():
            selected_col = st.session_state.get('selected_outlier_col_v2')
            if selected_col:
                # Outlier detection method
                method = st.selectbox(
                    "Outlier treatment method",
                    ["Remove (IQR)", "Cap (Percentile)"],
                    key="outlier_treatment_method_v2"
                )
                # Clear preview results if method changes
                clear_preview_on_change('outlier_preview_results_v2', method, 'outlier_method_v2')
                return method
            return None
        
        # Use the new builder for configuration section
        render_cleaning_config_section(config_col1, config_col2)
        
        selected_col = st.session_state.get('selected_outlier_col_v2')
        method = st.session_state.get('outlier_method_v2')
        
        if selected_col and method:
            # Parameter controls (below dropdowns)
            if method == "Remove (IQR)":
                multiplier = st.slider("IQR multiplier", 1.0, 3.0, OUTLIER_IQR_MULTIPLIER, 0.1, key="outlier_iqr_multiplier_v2")
                st.session_state.outlier_multiplier_v2 = multiplier
                st.info(f"Values outside Q1 - {multiplier}×IQR and Q3 + {multiplier}×IQR will be removed")
            else:
                lower_pct = st.slider("Lower percentile", 0.01, 0.1, 0.01, 0.01, key="outlier_lower_percentile_v2")
                upper_pct = st.slider("Upper percentile", 0.9, 0.99, 0.99, 0.01, key="outlier_upper_percentile_v2")
                st.session_state.outlier_lower_pct_v2 = lower_pct
                st.session_state.outlier_upper_pct_v2 = upper_pct
                st.info(f"Values below {lower_pct*100}th and above {upper_pct*100}th percentile will be capped")
            
            # Show current outlier statistics using new builder
            Q1 = current_df[selected_col].quantile(0.25)
            Q3 = current_df[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            
            if method == "Remove (IQR)":
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                outliers = current_df[(current_df[selected_col] < lower_bound) | (current_df[selected_col] > upper_bound)]
                outlier_count = len(outliers)
                outlier_pct = (outlier_count / len(current_df)) * 100
                info_dict = {
                    "Q1": f"{Q1:.2f}",
                    "Q3": f"{Q3:.2f}",
                    "IQR": f"{IQR:.2f}",
                    "Outliers": f"{outlier_count} ({outlier_pct:.1f}%)"
                }
            else:
                lower_bound = current_df[selected_col].quantile(lower_pct)
                upper_bound = current_df[selected_col].quantile(upper_pct)
                outliers = current_df[(current_df[selected_col] < lower_bound) | (current_df[selected_col] > upper_bound)]
                outlier_count = len(outliers)
                outlier_pct = (outlier_count / len(current_df)) * 100
                info_dict = {
                    "Lower Cap": f"{lower_bound:.2f}",
                    "Upper Cap": f"{upper_bound:.2f}",
                    "Values to Cap": f"{outlier_count} ({outlier_pct:.1f}%)"
                }
            
            render_column_info_line(selected_col, info_dict)
            st.markdown("---")
            
            # ROW 2: Current Data using new builder
            sample_data = current_df[selected_col].head(5)
            render_current_data_section(sample_data, 
                                      title="📊 Current Data",
                                      subtitle="🔍 Sample from selected column (first 5 rows):")
            
            st.markdown("---")
            
            # ROW 3: Action Buttons using new builder
            def preview_callback():
                try:
                    # Get values from session state for reliability
                    current_df = get_current_dataframe()
                    selected_col = st.session_state.get('selected_outlier_col_v2')
                    method = st.session_state.get('outlier_method_v2')
                    
                    if not selected_col or not method:
                        st.error("Please select a column and method before previewing")
                        return
                    
                    # Clear any existing preview results to prevent stale data
                    if 'outlier_preview_results_v2' in st.session_state:
                        del st.session_state.outlier_preview_results_v2
                    
                    # Perform preview operation
                    with st.spinner("Generating preview..."):
                        if method == "Remove (IQR)":
                            multiplier = st.session_state.get('outlier_multiplier_v2', OUTLIER_IQR_MULTIPLIER)
                            preview_df = remove_outliers_iqr(current_df, selected_col, multiplier)
                        else:
                            lower_pct = st.session_state.get('outlier_lower_pct_v2', 0.01)
                            upper_pct = st.session_state.get('outlier_upper_pct_v2', 0.99)
                            preview_df = cap_outliers(current_df, selected_col, lower_pct, upper_pct)
                        
                        # Store preview results in session state
                        st.session_state.outlier_preview_results_v2 = {
                            'data': preview_df[selected_col],
                            'before_rows': len(current_df),
                            'after_rows': len(preview_df),
                            'before_stats': current_df[selected_col].describe(),
                            'after_stats': preview_df[selected_col].describe(),
                            'preview_success': True
                        }
                    
                    # Force rerun to show preview results immediately
                    st.rerun()
                    
                except Exception as e:
                    handle_operation_error(e, "Preview", st.session_state.get('advanced_mode', False))
            
            def apply_callback():
                try:
                    # Get values from session state for reliability
                    current_df = get_current_dataframe()
                    selected_col = st.session_state.get('selected_outlier_col_v2')
                    method = st.session_state.get('outlier_method_v2')
                    
                    # Enhanced validation
                    if not selected_col:
                        st.error("❌ No column selected. Please select a column to treat.")
                        return
                    if not method:
                        st.error("❌ No method selected. Please select a treatment method.")
                        return
                    if selected_col not in current_df.columns:
                        st.error(f"❌ Column '{selected_col}' not found in dataframe.")
                        return
                    
                    # Show progress indicator
                    with st.spinner(f"Applying {method} to column '{selected_col}'..."):
                        if method == "Remove (IQR)":
                            multiplier = st.session_state.get('outlier_multiplier_v2', OUTLIER_IQR_MULTIPLIER)
                            cleaned_df = remove_outliers_iqr(current_df, selected_col, multiplier)
                            operation_desc = f"Remove outliers from '{selected_col}' (IQR × {multiplier})"
                        else:
                            lower_pct = st.session_state.get('outlier_lower_pct_v2', 0.01)
                            upper_pct = st.session_state.get('outlier_upper_pct_v2', 0.99)
                            cleaned_df = cap_outliers(current_df, selected_col, lower_pct, upper_pct)
                            operation_desc = f"Cap outliers in '{selected_col}' ({lower_pct*100}%-{upper_pct*100}%)"
                    
                    # Validate that cleaning actually worked
                    if cleaned_df is None:
                        st.error("❌ Treatment operation returned no data")
                        return
                    
                    if cleaned_df.shape[0] == 0:
                        st.error("❌ Treatment operation resulted in empty dataframe")
                        return
                    
                    # Calculate before/after statistics
                    rows_removed = len(current_df) - len(cleaned_df)
                    
                    # Store before data for comparison
                    before_sample = current_df[selected_col].head(5)
                    after_sample = cleaned_df[selected_col].head(5)
                    
                    # Update session state with cleaned dataframe using centralized function
                    update_current_dataframe(cleaned_df, operation_desc)
                    
                    # Add to pipeline history
                    if 'cleaning_pipeline' not in st.session_state:
                        st.session_state.cleaning_pipeline = []
                    
                    operation = {
                        'operation': operation_desc,
                        'method': method,
                        'column': selected_col,
                        'timestamp': pd.Timestamp.now().strftime('%H:%M:%S'),
                        'rows_before': len(current_df),
                        'rows_after': len(cleaned_df)
                    }
                    st.session_state.cleaning_pipeline.append(operation)
                    
                    # Show immediate success feedback
                    st.success(f"✅ Successfully applied {method} to column '{selected_col}'!")
                    
                    # Show comprehensive before/after results using new builder
                    metrics = {
                        'metric1_value': f"{len(current_df):,}",
                        'metric2_value': f"{len(cleaned_df):,}", 
                        'metric2_delta': f"{len(cleaned_df) - len(current_df):,}",
                        'metric3_value': f"{rows_removed:,}" if rows_removed > 0 else "Capped",
                        'metric3_color': "normal",
                        'shape_info': f"{len(current_df):,} → {len(cleaned_df):,} rows"
                    }
                    
                    labels = {
                        'before_label': "Before treatment",
                        'after_label': "After treatment",
                        'metric1_label': "Before Rows",
                        'metric1_help': "Total rows before treatment",
                        'metric2_label': "After Rows",
                        'metric3_label': "Rows Removed" if rows_removed > 0 else "Values Modified"
                    }
                    
                    # Show comprehensive before/after results using modular builder
                    labels = {
                        'before_label': 'Before treatment',
                        'after_label': 'After treatment'
                    }
                    render_applied_changes_section(before_sample, after_sample, labels)
                    
                    # Metrics section
                    st.markdown("**📊 Impact Summary:**")
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        st.metric("Before Rows", f"{len(current_df):,}", help="Total rows before treatment")
                    with result_col2:
                        st.metric("After Rows", f"{len(cleaned_df):,}", delta=f"{len(cleaned_df) - len(current_df):,}")
                    with result_col3:
                        if rows_removed > 0:
                            st.metric("Rows Removed", f"{rows_removed:,}", delta_color="normal")
                        else:
                            st.metric("Values Modified", "Capped", delta_color="normal")
                    
                    # Show data shape changes
                    st.markdown(f"**Data shape:** {len(current_df):,} → {len(cleaned_df):,} rows")
                    
                    # Clear preview results since we've applied the changes
                    if 'outlier_preview_results_v2' in st.session_state:
                        del st.session_state.outlier_preview_results_v2
                    
                except Exception as e:
                    handle_operation_error(e, "Treatment", st.session_state.get('advanced_mode', False))
            
            preview_btn_config = {
                'text': "👁️ Preview Treatment",
                'key': "preview_outliers_v2", 
                'callback': preview_callback
            }
            
            apply_btn_config = {
                'text': "✅ Apply Treatment",
                'key': "apply_outliers_v2",
                'callback': apply_callback
            }
            
            render_action_buttons_section(preview_btn_config, apply_btn_config)
            
            # ROW 4: Preview Results using new builder
            if 'outlier_preview_results_v2' in st.session_state:
                preview_data = st.session_state.outlier_preview_results_v2['data']
                before_rows = int(st.session_state.outlier_preview_results_v2['before_rows'])
                after_rows = int(st.session_state.outlier_preview_results_v2['after_rows'])
                
                metrics_data = {
                    'metric1_label': "Before Rows",
                    'metric1_value': f"{before_rows:,}",
                    'metric2_label': "After Rows", 
                    'metric2_value': f"{after_rows:,}",
                    'metric2_delta': f"{after_rows - before_rows:,}",
                    'metric3_label': "Would Remove" if after_rows < before_rows else "Change",
                    'metric3_value': f"{before_rows - after_rows:,} rows" if after_rows < before_rows else "Values capped",
                    'metric3_color': "normal"
                }
                
                render_preview_results_section(preview_data, metrics_data, 'outlier_preview_results_v2')
    else:
        st.info("No numeric columns available for outlier treatment")


def render_duplicates_cleaning() -> None:
    """
    Duplicates cleaning using modular UI builders.
    """
    st.subheader("🔄 Duplicate Rows Treatment")
    
    try:
        # Ensure session state integrity before operations
        ensure_session_state_integrity()
        
        # Validate session state before proceeding
        if not validate_session_state():
            st.warning("No data available for duplicate cleaning")
            return
            
        # Always use the current dataframe from session state for operations
        current_df = get_current_dataframe()
    except ValueError:
        st.warning("No data available for duplicate cleaning")
        return
    
    duplicates = current_df.duplicated().sum()
    
    if duplicates > 0:
        # ROW 1: Configuration using new builder
        def config_col1():
            # Duplicate handling options
            keep_option = st.selectbox(
                "Keep which occurrence?",
                ["first", "last", "none"],
                format_func=lambda x: {
                    "first": "First occurrence", 
                    "last": "Last occurrence", 
                    "none": "Remove all duplicates"
                }[x],
                key="duplicate_keep_option_v2"
            )
            # Clear preview results if keep option changes
            clear_preview_on_change('duplicate_preview_results_v2', keep_option, 'duplicate_keep_option_v2')
            return keep_option
        
        def config_col2():
            # Subset selection
            subset_cols = st.multiselect(
                "Consider only these columns for duplicates (leave empty for all columns)",
                current_df.columns.tolist(),
                key="duplicate_subset_columns_v2"
            )
            # Clear preview results if subset changes
            clear_preview_on_change('duplicate_preview_results_v2', subset_cols, 'duplicate_subset_cols_v2')
            return subset_cols
        
        # Use the new builder for configuration section
        render_cleaning_config_section(config_col1, config_col2)
        
        keep_option = st.session_state.get('duplicate_keep_option_v2', 'first')
        subset_cols = st.session_state.get('duplicate_subset_cols_v2', [])
        
        # Show duplicate information using new builder
        dup_pct = (duplicates / len(current_df)) * 100
        unique_rows = len(current_df) - duplicates
        info_dict = {
            "Total Rows": f"{len(current_df):,}",
            "Duplicate Rows": f"{duplicates:,} ({dup_pct:.2f}%)",
            "Unique Rows": f"{unique_rows:,}"
        }
        render_column_info_line("Duplicates", info_dict)
        st.markdown("---")
        
        # ROW 2: Current Data using new builder - show duplicate samples
        st.markdown("**📊 Current Data**")
        st.markdown("*🔍 Sample duplicate rows (showing first 5 duplicates):*")
        if len(subset_cols) > 0:
            sample_dups = current_df[current_df.duplicated(subset=subset_cols, keep=False)].head(5)
        else:
            sample_dups = current_df[current_df.duplicated(keep=False)].head(5)
        
        if not sample_dups.empty:
            safe_dataframe(sample_dups, use_container_width=True)
        else:
            st.info("No duplicates found with current settings")
        
        st.markdown("---")
        
        # ROW 3: Action Buttons using new builder
        def preview_callback():
            try:
                # Get values from session state for reliability
                current_df = get_current_dataframe()
                keep_option = st.session_state.get('duplicate_keep_option_v2', 'first')
                subset_cols = st.session_state.get('duplicate_subset_cols_v2', [])
                
                # Clear any existing preview results to prevent stale data
                if 'duplicate_preview_results_v2' in st.session_state:
                    del st.session_state.duplicate_preview_results_v2
                
                # Perform preview operation
                with st.spinner("Generating preview..."):
                    subset = subset_cols if subset_cols else None
                    keep = False if keep_option == "none" else keep_option
                    preview_df = remove_duplicates(current_df, subset, keep)
                    
                    # Store preview results in session state
                    st.session_state.duplicate_preview_results_v2 = {
                        'data': preview_df,
                        'before_rows': len(current_df),
                        'after_rows': len(preview_df),
                        'preview_success': True
                    }
                
                # Force rerun to show preview results immediately
                st.rerun()
                
            except Exception as e:
                handle_operation_error(e, "Preview", st.session_state.get('advanced_mode', False))
        
        def apply_callback():
            try:
                # Get values from session state for reliability
                current_df = get_current_dataframe()
                keep_option = st.session_state.get('duplicate_keep_option_v2', 'first')
                subset_cols = st.session_state.get('duplicate_subset_cols_v2', [])
                
                # Enhanced validation
                if not keep_option:
                    st.error("❌ No keep option selected. Please select how to handle duplicates.")
                    return
                
                # Show progress indicator
                with st.spinner(f"Removing duplicates (keep={keep_option})..."):
                    subset = subset_cols if subset_cols else None
                    keep = False if keep_option == "none" else keep_option
                    cleaned_df = remove_duplicates(current_df, subset, keep)
                
                # Validate that cleaning actually worked
                if cleaned_df is None:
                    st.error("❌ Removal operation returned no data")
                    return
                
                if cleaned_df.shape[0] == 0:
                    st.error("❌ Removal operation resulted in empty dataframe")
                    return
                
                # Calculate before/after statistics
                duplicates_removed = len(current_df) - len(cleaned_df)
                
                # Store before data for comparison
                before_sample = current_df.head(5)
                after_sample = cleaned_df.head(5)
                
                # Update session state with cleaned dataframe using centralized function
                operation_desc = f"Remove duplicates ({duplicates_removed:,} removed)"
                update_current_dataframe(cleaned_df, operation_desc)
                
                # Add to pipeline history
                if 'cleaning_pipeline' not in st.session_state:
                    st.session_state.cleaning_pipeline = []
                
                operation = {
                    'operation': f"Remove duplicates (keep={keep_option})",
                    'method': f"keep={keep_option}",
                    'columns': subset_cols,
                    'timestamp': pd.Timestamp.now().strftime('%H:%M:%S'),
                    'rows_before': len(current_df),
                    'rows_after': len(cleaned_df)
                }
                st.session_state.cleaning_pipeline.append(operation)
                
                # Show immediate success feedback
                st.success(f"✅ Successfully removed {duplicates_removed:,} duplicate rows!")
                
                # Show comprehensive before/after results using new builder
                metrics = {
                    'metric1_value': f"{len(current_df):,}",
                    'metric2_value': f"{len(cleaned_df):,}", 
                    'metric2_delta': f"{len(cleaned_df) - len(current_df):,}",
                    'metric3_value': f"{duplicates_removed:,} duplicates" if duplicates_removed > 0 else "No change",
                    'metric3_color': "normal" if duplicates_removed > 0 else "off",
                    'shape_info': f"{len(current_df):,} → {len(cleaned_df):,} rows"
                }
                
                labels = {
                    'before_label': "Before removal",
                    'after_label': "After removal",
                    'metric1_label': "Before Rows",
                    'metric1_help': "Total rows before removal",
                    'metric2_label': "After Rows",
                    'metric3_label': "Removed" if duplicates_removed > 0 else "Change"
                }
                
                # Show comprehensive before/after results using modular builder
                labels = {
                    'before_label': 'Before removal',
                    'after_label': 'After removal'
                }
                render_applied_changes_section(before_sample, after_sample, labels)
                
                # Metrics section
                st.markdown("**📊 Impact Summary:**")
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    st.metric("Before Rows", f"{len(current_df):,}", help="Total rows before removal")
                with result_col2:
                    st.metric("After Rows", f"{len(cleaned_df):,}", delta=f"{len(cleaned_df) - len(current_df):,}")
                with result_col3:
                    if duplicates_removed > 0:
                        st.metric("Removed", f"{duplicates_removed:,} duplicates", delta_color="normal")
                    else:
                        st.metric("Change", "No change", delta_color="off")
                
                # Show data shape changes
                st.markdown(f"**Data shape:** {len(current_df):,} → {len(cleaned_df):,} rows")
                
                # Clear preview results since we've applied the changes
                if 'duplicate_preview_results_v2' in st.session_state:
                    del st.session_state.duplicate_preview_results_v2
                
            except Exception as e:
                handle_operation_error(e, "Removal", st.session_state.get('advanced_mode', False))
        
        preview_btn_config = {
            'text': "👁️ Preview Removal",
            'key': "preview_duplicates_v2", 
            'callback': preview_callback
        }
        
        apply_btn_config = {
            'text': "✅ Remove Duplicates",
            'key': "apply_duplicates_v2",
            'callback': apply_callback
        }
        
        render_action_buttons_section(preview_btn_config, apply_btn_config)
        
        # ROW 4: Preview Results using new builder
        if 'duplicate_preview_results_v2' in st.session_state:
            preview_data = st.session_state.duplicate_preview_results_v2['data'].head(5)
            before_rows = int(st.session_state.duplicate_preview_results_v2['before_rows'])
            after_rows = int(st.session_state.duplicate_preview_results_v2['after_rows'])
            
            metrics_data = {
                'metric1_label': "Before Rows",
                'metric1_value': f"{before_rows:,}",
                'metric2_label': "After Rows", 
                'metric2_value': f"{after_rows:,}",
                'metric2_delta': f"{after_rows - before_rows:,}",
                'metric3_label': "Would Remove" if after_rows < before_rows else "Change",
                'metric3_value': f"{before_rows - after_rows:,} duplicates" if after_rows < before_rows else "No duplicates found",
                'metric3_color': "normal" if after_rows < before_rows else "off"
            }
            
            render_preview_results_section(preview_data, metrics_data, 'duplicate_preview_results_v2')
    else:
        st.success("✅ No duplicate rows found!")