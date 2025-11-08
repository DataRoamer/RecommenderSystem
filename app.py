import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modules.data_loader import load_data, get_basic_metadata, preview_data, validate_dataframe, get_column_info
from modules.data_quality import comprehensive_data_quality_report
from modules.eda_analysis import comprehensive_eda_report
from modules.target_analysis import analyze_target_variable, auto_detect_target_candidates
from modules.feature_engineering import comprehensive_feature_engineering_report
from modules.model_readiness import calculate_comprehensive_readiness_score
from modules.leakage_detection import comprehensive_leakage_detection, get_leakage_summary_stats
from utils.visualizations import (
    plot_missing_heatmap, plot_missing_bar, plot_data_types,
    plot_correlation_heatmap, plot_outliers_boxplot, format_percentage,
    create_quality_score_gauge, plot_missing_pattern_matrix,
    plot_feature_distributions, plot_correlation_network, plot_pairwise_relationships
)
# AI Module - Local LLM Integration (Privacy-First)
from modules.ai import (
    display_ai_setup_wizard,
    display_ai_status_badge,
    display_model_settings,
    check_ai_prerequisites,
    display_ai_chat,
    display_ai_insights,
    display_nl_query_translator,
    display_data_cleaning,
    display_feature_engineering_ai,
    display_anomaly_explanation
)
import os

# Configure page
st.set_page_config(
    page_title="EDA & Data Quality Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'metadata' not in st.session_state:
        st.session_state.metadata = None
    if 'filename' not in st.session_state:
        st.session_state.filename = None
    if 'quality_report' not in st.session_state:
        st.session_state.quality_report = None
    if 'current_section' not in st.session_state:
        st.session_state.current_section = "upload"
    if 'eda_report' not in st.session_state:
        st.session_state.eda_report = None
    if 'target_analysis' not in st.session_state:
        st.session_state.target_analysis = None
    if 'feature_engineering_report' not in st.session_state:
        st.session_state.feature_engineering_report = None
    if 'model_readiness' not in st.session_state:
        st.session_state.model_readiness = None
    if 'selected_target' not in st.session_state:
        st.session_state.selected_target = None
    if 'file_processed' not in st.session_state:
        st.session_state.file_processed = False

def display_upload_section():
    """Display file upload section"""
    st.markdown('<div class="section-header">📁 Data Upload</div>', unsafe_allow_html=True)

    # Show current status if data is loaded
    if st.session_state.data_loaded:
        st.success(f"✅ Currently loaded: {st.session_state.filename}")

        # Show quick stats
        if st.session_state.df is not None:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{st.session_state.df.shape[0]:,}")
            with col2:
                st.metric("Columns", st.session_state.df.shape[1])
            with col3:
                if st.session_state.quality_report:
                    score = st.session_state.quality_report['quality_score']['overall_score']
                    st.metric("Quality Score", f"{score}/100")
            with col4:
                st.metric("Memory", f"{st.session_state.metadata['memory_usage_mb']:.1f} MB")

        st.info("💡 Use the sidebar to navigate to different analysis sections")
        st.markdown("---")

        # Option to upload a different file
        if st.button("📁 Upload a Different Dataset"):
            st.session_state.data_loaded = False
            st.session_state.file_processed = False
            st.rerun()

        return

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your dataset to begin analysis. Supported formats: CSV, Excel (.xlsx, .xls)",
        key="file_uploader"
    )

    if uploaded_file is not None:
        # Get file ID to track if it's the same upload
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"

        # Check if this exact file was already processed
        if ('last_file_id' in st.session_state and
            st.session_state.last_file_id == file_id and
            st.session_state.file_processed):
            # File already processed, show confirmation
            st.success(f"✅ {uploaded_file.name} is already loaded and ready for analysis")

            # Show quick stats
            if st.session_state.df is not None:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", f"{st.session_state.df.shape[0]:,}")
                with col2:
                    st.metric("Columns", st.session_state.df.shape[1])
                with col3:
                    if st.session_state.quality_report:
                        score = st.session_state.quality_report['quality_score']['overall_score']
                        st.metric("Quality Score", f"{score}/100")
                with col4:
                    st.metric("Memory", f"{st.session_state.metadata['memory_usage_mb']:.1f} MB")

            st.info("💡 Navigate to other sections using the sidebar to explore your data")
            return

        # New or different file - store the file ID and reset processed flag
        st.session_state.last_file_id = file_id
        st.session_state.file_processed = False

        # Reset other states for new dataset
        st.session_state.selected_target = None
        st.session_state.target_analysis = None
        st.session_state.model_readiness = None

        # Determine file type
        file_extension = uploaded_file.name.split('.')[-1].lower()
        file_type = 'csv' if file_extension == 'csv' else 'excel'

        # Load data with progress bar
        with st.spinner(f'Loading {uploaded_file.name}...'):
            try:
                df, metadata = load_data(uploaded_file, file_type)

                # Validate data
                validation = validate_dataframe(df)

                if validation['is_valid']:
                    # Store in session state
                    st.session_state.df = df
                    st.session_state.metadata = metadata
                    st.session_state.filename = uploaded_file.name
                    st.session_state.data_loaded = True

                    # Generate quality report and EDA report
                    with st.spinner('Analyzing data quality...'):
                        st.session_state.quality_report = comprehensive_data_quality_report(df)

                    with st.spinner('Performing exploratory data analysis...'):
                        st.session_state.eda_report = comprehensive_eda_report(df)

                    with st.spinner('Analyzing feature engineering opportunities...'):
                        st.session_state.feature_engineering_report = comprehensive_feature_engineering_report(df)

                    # Mark as processed to prevent reprocessing on reruns
                    st.session_state.file_processed = True

                    # Automatically navigate to overview after successful load
                    st.session_state.current_section = "overview"

                    st.success(f"✅ Successfully loaded {uploaded_file.name}")

                    # Display warnings if any
                    if validation['warnings']:
                        for warning in validation['warnings']:
                            st.warning(f"⚠️ {warning}")

                    # Trigger rerun to show the overview section
                    st.rerun()

                else:
                    for error in validation['errors']:
                        st.error(f"❌ {error}")
                    st.session_state.data_loaded = False

            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.session_state.data_loaded = False

def display_data_overview():
    """Display data overview section"""
    if not st.session_state.data_loaded:
        return

    df = st.session_state.df
    metadata = st.session_state.metadata

    st.markdown('<div class="section-header">📊 Data Overview</div>', unsafe_allow_html=True)

    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{metadata['shape'][0]:,}")
    with col2:
        st.metric("Columns", metadata['shape'][1])
    with col3:
        st.metric("Memory Usage", f"{metadata['memory_usage_mb']:.1f} MB")
    with col4:
        missing_pct = metadata['total_missing_percentage']
        st.metric("Missing Data", f"{missing_pct:.1f}%")

    # File information
    st.subheader("File Information")
    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.write("**Filename:**", st.session_state.filename)
        st.write("**Encoding:**", metadata.get('encoding', 'N/A'))
        st.write("**Separator:**", metadata.get('separator', 'N/A'))

    with info_col2:
        st.write("**Numeric Columns:**", len(metadata['numeric_columns']))
        st.write("**Categorical Columns:**", len(metadata['categorical_columns']))
        st.write("**DateTime Columns:**", len(metadata['datetime_columns']))

    # Column types breakdown
    if metadata['numeric_columns'] or metadata['categorical_columns'] or metadata['datetime_columns']:
        st.subheader("Column Types")

        col_type_data = []
        if metadata['numeric_columns']:
            col_type_data.append({"Type": "Numeric", "Count": len(metadata['numeric_columns']), "Columns": ", ".join(metadata['numeric_columns'][:5]) + ("..." if len(metadata['numeric_columns']) > 5 else "")})
        if metadata['categorical_columns']:
            col_type_data.append({"Type": "Categorical", "Count": len(metadata['categorical_columns']), "Columns": ", ".join(metadata['categorical_columns'][:5]) + ("..." if len(metadata['categorical_columns']) > 5 else "")})
        if metadata['datetime_columns']:
            col_type_data.append({"Type": "DateTime", "Count": len(metadata['datetime_columns']), "Columns": ", ".join(metadata['datetime_columns'][:5]) + ("..." if len(metadata['datetime_columns']) > 5 else "")})

        st.dataframe(pd.DataFrame(col_type_data), use_container_width=True)

def display_data_preview():
    """Display data preview section"""
    if not st.session_state.data_loaded:
        return

    df = st.session_state.df

    st.markdown('<div class="section-header">👀 Data Preview</div>', unsafe_allow_html=True)

    # Preview options
    col1, col2 = st.columns([3, 1])
    with col1:
        preview_option = st.selectbox("Preview Type", ["First 10 rows", "Last 10 rows", "Random 10 rows"])
    with col2:
        n_rows = st.number_input("Number of rows", min_value=5, max_value=50, value=10)

    # Display preview
    if preview_option == "First 10 rows":
        preview_df = df.head(n_rows)
    elif preview_option == "Last 10 rows":
        preview_df = df.tail(n_rows)
    else:  # Random rows
        preview_df = df.sample(n=min(n_rows, len(df)))

    st.dataframe(preview_df, use_container_width=True)

    # Column information
    st.subheader("Detailed Column Information")
    column_info = get_column_info(df)
    st.dataframe(column_info, use_container_width=True)

def display_data_quality_dashboard():
    """Display comprehensive data quality dashboard"""
    if not st.session_state.data_loaded or not st.session_state.quality_report:
        return

    df = st.session_state.df
    report = st.session_state.quality_report

    st.markdown('<div class="section-header">📋 Data Quality Analysis</div>', unsafe_allow_html=True)

    # Quality Score Overview
    st.subheader("📈 Overall Quality Score")
    quality_score = report['quality_score']

    col1, col2 = st.columns([1, 2])
    with col1:
        # Display score with color
        score_value = quality_score['overall_score']
        if score_value >= 90:
            st.success(f"🟢 **{score_value}/100** - {quality_score['interpretation']}")
        elif score_value >= 75:
            st.success(f"🟡 **{score_value}/100** - {quality_score['interpretation']}")
        elif score_value >= 60:
            st.warning(f"🟠 **{score_value}/100** - {quality_score['interpretation']}")
        else:
            st.error(f"🔴 **{score_value}/100** - {quality_score['interpretation']}")

    with col2:
        # Score breakdown
        st.write("**Score Breakdown:**")
        st.write(f"• Missing Data: {quality_score['missing_data_score']}/30")
        st.write(f"• Duplicates: {quality_score['duplicate_score']}/25")
        st.write(f"• Consistency: {quality_score['consistency_score']}/25")
        st.write(f"• Outliers: {quality_score['outlier_score']}/20")

    st.markdown("---")

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "❌ Missing Data", "👥 Duplicates", "🔍 Outliers", "📈 Statistics"
    ])

    with tab1:
        # Overview tab
        col1, col2, col3, col4 = st.columns(4)

        missing_pct = report['missing_values']['overall_missing_percentage']
        duplicate_pct = report['duplicates']['duplicate_percentage']
        constant_features = len(report['constant_features']['constant_features'])
        outliers_total = report['outliers_iqr']['summary']['total_outliers']

        with col1:
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        with col2:
            st.metric("Duplicates", f"{duplicate_pct:.1f}%")
        with col3:
            st.metric("Constant Features", constant_features)
        with col4:
            st.metric("Outliers (IQR)", outliers_total)

        # Data type distribution
        st.subheader("📊 Data Type Distribution")
        try:
            fig_dtype = plot_data_types(df)
            st.pyplot(fig_dtype)
        except Exception as e:
            st.error(f"Error creating data type plot: {str(e)}")

    with tab2:
        # Missing data tab
        missing_data = report['missing_values']

        if missing_data['total_missing'] == 0:
            st.success("🎉 No missing values found in the dataset!")
        else:
            st.subheader("❌ Missing Values Summary")

            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Missing", f"{missing_data['total_missing']:,}")
            with col2:
                st.metric("Overall %", f"{missing_data['overall_missing_percentage']:.1f}%")
            with col3:
                st.metric("Columns Affected", len(missing_data['columns_with_missing']))

            # Missing values by column
            if missing_data['columns_with_missing']:
                st.subheader("Missing Values by Column")

                missing_df = pd.DataFrame([
                    {
                        'Column': col,
                        'Missing Count': info['count'],
                        'Missing %': info['percentage'],
                        'Data Type': info['data_type']
                    }
                    for col, info in missing_data['columns_with_missing'].items()
                ]).sort_values('Missing %', ascending=False)

                st.dataframe(missing_df, use_container_width=True)

                # Visualizations
                st.subheader("Missing Data Visualizations")

                col1, col2 = st.columns(2)

                with col1:
                    try:
                        fig_bar = plot_missing_bar(df)
                        st.pyplot(fig_bar)
                    except Exception as e:
                        st.error(f"Error creating missing values bar chart: {str(e)}")

                with col2:
                    try:
                        fig_heatmap = plot_missing_heatmap(df)
                        st.pyplot(fig_heatmap)
                    except Exception as e:
                        st.error(f"Error creating missing values heatmap: {str(e)}")

                # Critical columns warning
                if missing_data['critical_columns']:
                    st.warning(f"⚠️ **Critical columns with >50% missing data:** {', '.join(missing_data['critical_columns'])}")

    with tab3:
        # Duplicates tab
        duplicate_data = report['duplicates']

        if duplicate_data['duplicate_rows'] == 0:
            st.success("🎉 No duplicate rows found in the dataset!")
        else:
            st.subheader("👥 Duplicate Rows Analysis")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duplicate Rows", f"{duplicate_data['duplicate_rows']:,}")
            with col2:
                st.metric("Duplicate %", f"{duplicate_data['duplicate_percentage']:.1f}%")
            with col3:
                st.metric("Unique Rows", f"{duplicate_data['unique_rows']:,}")

            # Show example duplicates if available
            if 'duplicate_examples' in duplicate_data:
                st.subheader("Example Duplicate Rows")
                st.dataframe(duplicate_data['duplicate_examples'], use_container_width=True)

    with tab4:
        # Outliers tab
        outliers_iqr = report['outliers_iqr']
        outliers_zscore = report['outliers_zscore']

        st.subheader("🔍 Outlier Detection")

        # Method selector
        method = st.selectbox("Detection Method", ["IQR Method", "Z-Score Method"])

        if method == "IQR Method":
            outlier_data = outliers_iqr
        else:
            outlier_data = outliers_zscore

        if outlier_data['summary']['total_outliers'] == 0:
            st.success("🎉 No outliers detected using the selected method!")
        else:
            # Summary
            st.metric("Total Outliers", f"{outlier_data['summary']['total_outliers']:,}")

            # Outliers by column
            if outlier_data['outliers_by_column']:
                outlier_df = pd.DataFrame([
                    {
                        'Column': col,
                        'Outlier Count': info['outlier_count'],
                        'Outlier %': info['outlier_percentage']
                    }
                    for col, info in outlier_data['outliers_by_column'].items()
                    if info['outlier_count'] > 0
                ]).sort_values('Outlier %', ascending=False)

                if not outlier_df.empty:
                    st.dataframe(outlier_df, use_container_width=True)

                    # Box plots for outlier visualization
                    st.subheader("Outlier Visualization")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        try:
                            fig_outliers = plot_outliers_boxplot(df, numeric_cols[:6])  # Limit to first 6 columns
                            st.pyplot(fig_outliers)
                        except Exception as e:
                            st.error(f"Error creating outlier plots: {str(e)}")

    with tab5:
        # Statistics tab
        stats = report['statistics']

        # Numeric statistics
        if stats['numeric_stats']:
            st.subheader("📊 Numeric Features Statistics")
            numeric_stats_df = pd.DataFrame(stats['numeric_stats']).T
            st.dataframe(numeric_stats_df, use_container_width=True)

        # Categorical statistics
        if stats['categorical_stats']:
            st.subheader("📋 Categorical Features Statistics")
            cat_stats_list = []
            for col, stat_info in stats['categorical_stats'].items():
                cat_stats_list.append({
                    'Column': col,
                    'Count': stat_info['count'],
                    'Unique': stat_info['unique'],
                    'Unique %': stat_info['unique_ratio'],
                    'Most Frequent': str(stat_info['most_frequent'])[:50],
                    'Frequency': stat_info['most_frequent_count']
                })

            if cat_stats_list:
                cat_stats_df = pd.DataFrame(cat_stats_list)
                st.dataframe(cat_stats_df, use_container_width=True)

def display_sidebar():
    """Display sidebar navigation"""
    st.sidebar.markdown("# 🛠️ EDA Tool")

    # AI Status Badge
    try:
        display_ai_status_badge()
    except Exception:
        pass  # Silently fail if AI module has issues

    st.sidebar.markdown("---")

    # Navigation
    if st.session_state.data_loaded:
        st.sidebar.success("✅ Data Loaded")
        st.sidebar.markdown(f"**File:** {st.session_state.filename}")
        st.sidebar.markdown(f"**Shape:** {st.session_state.df.shape[0]:,} × {st.session_state.df.shape[1]}")

        # Display quality score in sidebar
        if st.session_state.quality_report:
            score = st.session_state.quality_report['quality_score']['overall_score']
            interpretation = st.session_state.quality_report['quality_score']['interpretation']
            if score >= 75:
                st.sidebar.success(f"Quality Score: {score}/100 ({interpretation})")
            elif score >= 60:
                st.sidebar.warning(f"Quality Score: {score}/100 ({interpretation})")
            else:
                st.sidebar.error(f"Quality Score: {score}/100 ({interpretation})")

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🧭 Navigation")

        # Section navigation
        sections = [
            ("📁 Data Upload", "upload"),
            ("🤖 AI Setup", "ai_setup"),
            ("📊 Data Overview", "overview"),
            ("📋 Data Quality", "quality"),
            ("🔍 EDA", "eda"),
            ("🧹 Data Cleaning", "data_cleaning"),
            ("💬 AI Chat", "ai_chat"),
            ("🧠 AI Insights", "ai_insights"),
            ("🔍 NL Query", "nl_query"),
            ("🤖 AI Feature Eng", "ai_feature_eng"),
            ("🔍 Anomaly Explain", "anomaly_explanation"),
            ("🎯 Target Analysis", "target"),
            ("🛠️ Feature Engineering", "features"),
            ("🚨 Leakage Detection", "leakage"),
            ("📈 Model Readiness", "readiness"),
            ("📄 Reports", "reports")
        ]

        for section_name, section_key in sections:
            if st.sidebar.button(section_name, key=f"nav_{section_key}"):
                st.session_state.current_section = section_key
                st.rerun()

        # Show current section status
        current_section = st.session_state.current_section
        implemented_sections = ["upload", "ai_setup", "ai_chat", "ai_insights", "data_cleaning", "nl_query", "ai_feature_eng", "anomaly_explanation", "overview", "quality", "eda", "target", "features", "leakage", "readiness", "reports"]

        for section_name, section_key in sections:
            if section_key == current_section:
                status = "🔵"
            elif section_key in implemented_sections:
                status = "✅"
            else:
                status = "⏳"
            # st.sidebar.text(f"{status} {section_name}")

    else:
        st.sidebar.info("👆 Upload a file to get started")

    st.sidebar.markdown("---")

    # Settings
    with st.sidebar.expander("⚙️ Settings"):
        st.markdown("**Display Options**")
        st.checkbox("Show detailed warnings", value=True)
        st.checkbox("Auto-refresh on data change", value=True)

        st.markdown("**Analysis Options**")
        st.slider("Missing data threshold (%)", 0, 100, 20)
        st.slider("Correlation threshold", 0.0, 1.0, 0.8, 0.1)

        # AI Settings (if available)
        st.markdown("---")
        st.markdown("**AI Settings**")
        try:
            display_model_settings()
        except Exception:
            st.caption("AI features not configured")

def display_eda_dashboard():
    """Display comprehensive EDA dashboard"""
    if not st.session_state.data_loaded or not st.session_state.eda_report:
        return

    df = st.session_state.df
    eda_report = st.session_state.eda_report

    st.markdown('<div class="section-header">🔍 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    # Create tabs for EDA sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Summary Statistics", "📈 Distributions", "🔗 Correlations", "🎯 Relationships", "📐 Advanced Analysis"
    ])

    with tab1:
        # Summary Statistics Tab
        st.subheader("📊 Dataset Summary")

        summary_stats = eda_report['summary_statistics']

        # Overall insights
        insights = summary_stats['overall_insights']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{insights['total_rows']:,}")
        with col2:
            st.metric("Total Columns", insights['total_columns'])
        with col3:
            st.metric("Memory Usage", f"{insights['memory_usage_mb']:.1f} MB")
        with col4:
            st.metric("Data Sparsity", f"{insights['sparsity']:.1f}%")

        # Numeric features summary
        if summary_stats['numeric_summary']:
            st.subheader("🔢 Numeric Features")
            numeric_df = pd.DataFrame(summary_stats['numeric_summary']).T
            st.dataframe(numeric_df, use_container_width=True)

        # Categorical features summary
        if summary_stats['categorical_summary']:
            st.subheader("📋 Categorical Features")
            cat_list = []
            for col, stats in summary_stats['categorical_summary'].items():
                cat_list.append({
                    'Column': col,
                    'Count': stats['count'],
                    'Missing': stats['missing'],
                    'Unique': stats['unique'],
                    'Unique %': stats['unique_pct'],
                    'Most Frequent': str(stats['most_frequent'])[:30],
                    'Frequency %': stats['most_frequent_pct']
                })

            if cat_list:
                cat_df = pd.DataFrame(cat_list)
                st.dataframe(cat_df, use_container_width=True)

    with tab2:
        # Distributions Tab
        st.subheader("📈 Feature Distributions")

        # Feature selector
        all_columns = df.columns.tolist()
        selected_features = st.multiselect(
            "Select features to visualize (max 6)",
            all_columns,
            default=all_columns[:6] if len(all_columns) >= 6 else all_columns
        )

        if selected_features:
            try:
                fig_dist = plot_feature_distributions(df, selected_features, max_cols=6)
                st.pyplot(fig_dist)
            except Exception as e:
                st.error(f"Error creating distribution plots: {str(e)}")

        # Distribution insights
        if 'distributions' in eda_report and eda_report['distributions']['distribution_insights']:
            st.subheader("📊 Distribution Analysis")

            insights_data = []
            for col, insights in eda_report['distributions']['distribution_insights'].items():
                insights_data.append({
                    'Feature': col,
                    'Skewness': insights['skewness'],
                    'Interpretation': insights['skewness_interpretation'],
                    'Kurtosis': insights['kurtosis'],
                    'Shape': insights['distribution_shape']
                })

            if insights_data:
                insights_df = pd.DataFrame(insights_data)
                st.dataframe(insights_df, use_container_width=True)

        # Transformation suggestions
        if 'distributions' in eda_report and eda_report['distributions']['transformation_suggestions']:
            st.subheader("🔄 Transformation Suggestions")

            for col, suggestions in eda_report['distributions']['transformation_suggestions'].items():
                if suggestions:
                    with st.expander(f"Suggestions for {col}"):
                        for suggestion in suggestions:
                            st.write(f"• {suggestion}")

    with tab3:
        # Correlations Tab
        st.subheader("🔗 Correlation Analysis")

        if 'correlations' in eda_report and 'pearson_correlation' in eda_report['correlations']:
            corr_data = eda_report['correlations']

            # Correlation summary
            if 'correlation_summary' in corr_data:
                summary = corr_data['correlation_summary']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Pairs", summary['total_pairs'])
                with col2:
                    st.metric("High Correlation Pairs", summary['high_correlation_pairs'])
                with col3:
                    st.metric("Max Correlation", f"{summary['max_correlation']:.3f}")

            # Correlation heatmap
            try:
                fig_corr = plot_correlation_heatmap(corr_data['pearson_correlation'])
                st.pyplot(fig_corr)
            except Exception as e:
                st.error(f"Error creating correlation heatmap: {str(e)}")

            # High correlation pairs
            if corr_data['high_correlations']:
                st.subheader("⚠️ High Correlation Pairs")
                high_corr_df = pd.DataFrame(corr_data['high_correlations'])
                st.dataframe(high_corr_df, use_container_width=True)

                # Correlation network
                st.subheader("🕸️ Correlation Network")
                threshold = st.slider("Correlation Threshold", 0.1, 1.0, 0.5, 0.1)
                try:
                    fig_network = plot_correlation_network(corr_data['pearson_correlation'], threshold)
                    st.pyplot(fig_network)
                except Exception as e:
                    st.error(f"Error creating correlation network: {str(e)}")
            else:
                st.info("No high correlations found (threshold ≥ 0.7)")

    with tab4:
        # Relationships Tab
        st.subheader("🎯 Feature Relationships")

        # Pairwise relationship explorer
        st.subheader("📊 Pairwise Relationships")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = df.columns.tolist()

        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Select X-axis feature", numeric_cols)
            with col2:
                y_col = st.selectbox("Select Y-axis feature",
                                   [col for col in numeric_cols if col != x_col])

            # Optional color coding
            hue_col = st.selectbox("Optional: Color by feature",
                                 ["None"] + [col for col in all_cols if col not in [x_col, y_col]])
            hue_col = None if hue_col == "None" else hue_col

            if x_col and y_col:
                try:
                    fig_scatter = plot_pairwise_relationships(df, x_col, y_col, hue_col)
                    st.pyplot(fig_scatter)
                except Exception as e:
                    st.error(f"Error creating scatter plot: {str(e)}")

        # Target analysis if available
        if 'target_analysis' in eda_report:
            st.subheader("🎯 Target Variable Analysis")
            target_info = eda_report['target_analysis']['target_info']

            st.write("**Target Variable Information:**")
            for key, value in target_info.items():
                if key not in ['class_distribution']:
                    st.write(f"• **{key.replace('_', ' ').title()}**: {value}")

    with tab5:
        # Advanced Analysis Tab
        st.subheader("📐 Advanced Statistical Analysis")

        # Normality tests
        if 'distributions' in eda_report and 'normality_tests' in eda_report['distributions']:
            normality_tests = eda_report['distributions']['normality_tests']

            if normality_tests:
                st.subheader("📊 Normality Tests")

                normality_data = []
                for col, tests in normality_tests.items():
                    if tests:
                        row = {'Feature': col}
                        if tests.get('shapiro_wilk'):
                            sw = tests['shapiro_wilk']
                            row['Shapiro-Wilk p-value'] = sw['p_value']
                            row['Is Normal (SW)'] = sw['is_normal']
                        if tests.get('kolmogorov_smirnov'):
                            ks = tests['kolmogorov_smirnov']
                            row['KS p-value'] = ks['p_value']
                            row['Is Normal (KS)'] = ks['is_normal']
                        normality_data.append(row)

                if normality_data:
                    normality_df = pd.DataFrame(normality_data)
                    st.dataframe(normality_df, use_container_width=True)

        # Feature selection recommendations
        st.subheader("🎯 Analysis Insights")

        insights = []

        # Add insights based on analysis
        if 'correlations' in eda_report:
            high_corr_count = len(eda_report['correlations'].get('high_correlations', []))
            if high_corr_count > 0:
                insights.append(f"⚠️ Found {high_corr_count} pairs of highly correlated features - consider feature selection")

        if 'distributions' in eda_report:
            skewed_features = []
            for col, dist_info in eda_report['distributions']['distribution_insights'].items():
                if abs(dist_info['skewness']) > 1:
                    skewed_features.append(col)

            if skewed_features:
                insights.append(f"📊 {len(skewed_features)} features are highly skewed - consider transformations")

        numeric_features = len(df.select_dtypes(include=[np.number]).columns)
        categorical_features = len(df.select_dtypes(include=['object']).columns)

        insights.extend([
            f"📈 Dataset has {numeric_features} numeric and {categorical_features} categorical features",
            f"💾 Total memory usage: {eda_report['summary_statistics']['overall_insights']['memory_usage_mb']:.1f} MB"
        ])

        for insight in insights:
            st.write(insight)

def display_target_analysis_dashboard():
    """Display target analysis dashboard"""
    if not st.session_state.data_loaded:
        return

    df = st.session_state.df

    st.markdown('<div class="section-header">🎯 Target Variable Analysis</div>', unsafe_allow_html=True)

    # Target selection
    st.subheader("🎯 Target Variable Selection")

    # Auto-detect potential targets
    target_candidates = auto_detect_target_candidates(df)
    all_columns = df.columns.tolist()

    # Target selector
    col1, col2 = st.columns([3, 1])
    with col1:
        if target_candidates:
            st.info(f"💡 Potential target variables detected: {', '.join(target_candidates)}")

        # Safely determine the index for selectbox
        default_index = 0
        if st.session_state.selected_target is not None and st.session_state.selected_target in all_columns:
            default_index = all_columns.index(st.session_state.selected_target) + 1

        selected_target = st.selectbox(
            "Select target variable",
            ["None"] + all_columns,
            index=default_index,
            help="Choose the variable you want to predict"
        )

    with col2:
        analyze_button = st.button("🔍 Analyze Target", type="primary")

    if selected_target != "None" and (analyze_button or st.session_state.selected_target == selected_target):
        st.session_state.selected_target = selected_target

        # Perform target analysis
        with st.spinner('Analyzing target variable...'):
            target_analysis = analyze_target_variable(df, selected_target)
            st.session_state.target_analysis = target_analysis

            # Update model readiness with target info
            st.session_state.model_readiness = calculate_comprehensive_readiness_score(
                st.session_state.quality_report,
                target_analysis,
                st.session_state.feature_engineering_report,
                df
            )

        if 'error' in target_analysis:
            st.error(target_analysis['error'])
            return

        # Display analysis results
        basic_info = target_analysis['basic_info']
        task_type = basic_info['suggested_task_type']

        # Basic information
        st.subheader("📊 Target Variable Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Task Type", task_type.title())
        with col2:
            st.metric("Non-null Count", f"{basic_info['non_null_count']:,}")
        with col3:
            st.metric("Missing Values", f"{basic_info['missing_count']:,}")
        with col4:
            st.metric("Unique Values", basic_info['unique_values'])

        # Task-specific analysis
        if task_type == 'classification':
            st.subheader("📋 Classification Analysis")

            class_analysis = target_analysis['classification_analysis']

            # Class distribution
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Class Distribution:**")
                class_dist_df = pd.DataFrame([
                    {'Class': cls, 'Count': count, 'Percentage': f"{pct:.1f}%"}
                    for cls, count, pct in zip(
                        class_analysis['class_names'],
                        [class_analysis['class_distribution'][cls] for cls in class_analysis['class_names']],
                        [class_analysis['class_percentages'][cls] for cls in class_analysis['class_names']]
                    )
                ])
                st.dataframe(class_dist_df, use_container_width=True)

            with col2:
                st.write("**Balance Analysis:**")
                st.write(f"• **Balance Ratio:** {class_analysis['balance_ratio']}")
                st.write(f"• **Imbalance Severity:** {class_analysis['imbalance_severity']}")
                st.write(f"• **Entropy:** {class_analysis['entropy']}")
                st.write(f"• **Gini Impurity:** {class_analysis['gini_impurity']}")

                # Color-code severity
                severity = class_analysis['imbalance_severity']
                if severity == "Balanced":
                    st.success("✅ Well balanced classes")
                elif severity in ["Slightly Imbalanced", "Moderately Imbalanced"]:
                    st.warning("⚠️ Some imbalance present")
                else:
                    st.error("🚨 Severe imbalance detected")

            # Visualization
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                classes = list(class_analysis['class_distribution'].keys())
                counts = list(class_analysis['class_distribution'].values())

                bars = ax.bar(classes, counts, color='skyblue', alpha=0.7, edgecolor='black')
                ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
                ax.set_xlabel('Classes')
                ax.set_ylabel('Count')

                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
                           f'{count}', ha='center', va='bottom')

                plt.xticks(rotation=45 if len(str(classes[0])) > 5 else 0)
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")

        else:  # regression
            st.subheader("📈 Regression Analysis")

            reg_analysis = target_analysis['regression_analysis']

            # Statistics
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Descriptive Statistics:**")
                stats_df = pd.DataFrame([
                    {'Statistic': 'Mean', 'Value': reg_analysis['mean']},
                    {'Statistic': 'Median', 'Value': reg_analysis['median']},
                    {'Statistic': 'Std Dev', 'Value': reg_analysis['std']},
                    {'Statistic': 'Min', 'Value': reg_analysis['min']},
                    {'Statistic': 'Max', 'Value': reg_analysis['max']},
                    {'Statistic': 'Range', 'Value': reg_analysis['range']}
                ])
                st.dataframe(stats_df, use_container_width=True)

            with col2:
                st.write("**Distribution Analysis:**")
                st.write(f"• **Skewness:** {reg_analysis['skewness']} ({reg_analysis['skewness_interpretation']})")
                st.write(f"• **Kurtosis:** {reg_analysis['kurtosis']}")
                st.write(f"• **Distribution Shape:** {reg_analysis['distribution_shape']}")
                st.write(f"• **CV:** {reg_analysis['coefficient_of_variation']}")

                # Outliers
                outlier_info = reg_analysis['outliers']
                st.write(f"• **Outliers:** {outlier_info['count']} ({outlier_info['percentage']:.1f}%)")

            # Visualization
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Histogram
                target_data = df[selected_target].dropna()
                ax1.hist(target_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax1.axvline(reg_analysis['mean'], color='red', linestyle='--', label=f"Mean: {reg_analysis['mean']:.2f}")
                ax1.axvline(reg_analysis['median'], color='green', linestyle='--', label=f"Median: {reg_analysis['median']:.2f}")
                ax1.set_title('Target Distribution')
                ax1.set_xlabel(selected_target)
                ax1.set_ylabel('Frequency')
                ax1.legend()

                # Box plot
                ax2.boxplot(target_data)
                ax2.set_title('Target Box Plot')
                ax2.set_ylabel(selected_target)

                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")

            # Transformation suggestions
            if reg_analysis['transformation_suggestions']:
                st.subheader("🔄 Transformation Suggestions")
                for suggestion in reg_analysis['transformation_suggestions']:
                    st.write(f"• {suggestion}")

        # ML Readiness
        if 'ml_readiness' in target_analysis:
            st.subheader("🚀 ML Readiness Assessment")
            readiness = target_analysis['ml_readiness']

            col1, col2 = st.columns([1, 2])
            with col1:
                score = readiness['overall_score']
                if score >= 75:
                    st.success(f"Score: {score}/100 - {readiness['interpretation']}")
                elif score >= 50:
                    st.warning(f"Score: {score}/100 - {readiness['interpretation']}")
                else:
                    st.error(f"Score: {score}/100 - {readiness['interpretation']}")

            with col2:
                st.write("**Score Breakdown:**")
                st.write(f"• Missing Data: {readiness['missing_data_score']}/30")
                st.write(f"• Balance/Distribution: {readiness['balance_score']}/40")
                st.write(f"• Overall Quality: {readiness['distribution_score']}/30")

        # Recommendations
        st.subheader("💡 Recommendations")
        recommendations = target_analysis['recommendations']
        if recommendations:
            for rec in recommendations:
                st.write(rec)
        else:
            st.success("✅ No specific recommendations - target looks good!")

    elif selected_target == "None":
        st.info("👆 Please select a target variable to begin analysis")

def display_feature_engineering_dashboard():
    """Display feature engineering dashboard"""
    if not st.session_state.data_loaded or not st.session_state.feature_engineering_report:
        return

    df = st.session_state.df
    fe_report = st.session_state.feature_engineering_report

    st.markdown('<div class="section-header">🛠️ Feature Engineering Recommendations</div>', unsafe_allow_html=True)

    # Summary overview
    summary = fe_report['summary']
    st.subheader("📊 Feature Engineering Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Features", summary['total_features'])
    with col2:
        st.metric("Need Encoding", summary['recommendations_count']['encoding'])
    with col3:
        st.metric("Need Scaling", summary['recommendations_count']['scaling'])
    with col4:
        st.metric("To Drop", summary['recommendations_count']['to_drop'])

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏷️ Feature Types", "🔤 Encoding", "📏 Scaling", "➕ New Features", "🗑️ Remove Features"
    ])

    with tab1:
        # Feature classification
        st.subheader("🏷️ Feature Type Classification")

        classification = fe_report['feature_classification']
        feature_types = summary['feature_types']

        # Feature type overview
        type_df = pd.DataFrame([
            {'Type': type_name.title(), 'Count': count}
            for type_name, count in feature_types.items()
        ])
        st.dataframe(type_df, use_container_width=True)

        # Detailed feature information
        if st.checkbox("Show detailed feature information"):
            details_list = []
            for feature, details in classification['feature_details'].items():
                details_list.append({
                    'Feature': feature,
                    'Type': details['feature_type'],
                    'Data Type': details['data_type'],
                    'Unique Count': details['unique_count'],
                    'Unique Ratio': details['unique_ratio'],
                    'Missing %': round(details['missing_ratio'] * 100, 1),
                    'Notes': details.get('note', '')
                })

            details_df = pd.DataFrame(details_list)
            st.dataframe(details_df, use_container_width=True)

    with tab2:
        # Encoding recommendations
        st.subheader("🔤 Categorical Encoding Recommendations")

        encoding_recs = fe_report['encoding_recommendations']['categorical_encoding']

        if encoding_recs:
            encoding_list = []
            for feature, rec in encoding_recs.items():
                encoding_list.append({
                    'Feature': feature,
                    'Recommended Method': rec['method'],
                    'Reason': rec['reason'],
                    'Complexity': rec['complexity']
                })

            encoding_df = pd.DataFrame(encoding_list)
            st.dataframe(encoding_df, use_container_width=True)

            # Show examples
            st.subheader("📝 Implementation Examples")

            selected_feature = st.selectbox("Select feature for example", list(encoding_recs.keys()))
            if selected_feature:
                method = encoding_recs[selected_feature]['method']

                st.code(f"""
# {method} for {selected_feature}
# Method: {encoding_recs[selected_feature]['method']}
# Reason: {encoding_recs[selected_feature]['reason']}

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Sample implementation:
if method == 'One-Hot Encoding':
    df_encoded = pd.get_dummies(df, columns=['{selected_feature}'])
elif method == 'Label Encoding':
    le = LabelEncoder()
    df['{selected_feature}_encoded'] = le.fit_transform(df['{selected_feature}'])
""", language='python')

        else:
            st.success("✅ No categorical encoding needed!")

    with tab3:
        # Scaling recommendations
        st.subheader("📏 Scaling Recommendations")

        scaling_recs = fe_report['encoding_recommendations']['scaling_recommendations']

        if scaling_recs:
            scaling_list = []
            for feature, rec in scaling_recs.items():
                scaling_list.append({
                    'Feature': feature,
                    'Recommended Method': rec['method'],
                    'Reason': rec['reason'],
                    'Complexity': rec['complexity']
                })

            scaling_df = pd.DataFrame(scaling_list)
            st.dataframe(scaling_df, use_container_width=True)

            # Implementation example
            st.subheader("📝 Scaling Implementation")
            st.code("""
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# Standard Scaler (most common)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numeric_columns])

# Robust Scaler (for outliers)
robust_scaler = RobustScaler()
df_robust = robust_scaler.fit_transform(df[numeric_columns])

# Min-Max Scaler (0-1 range)
minmax_scaler = MinMaxScaler()
df_minmax = minmax_scaler.fit_transform(df[numeric_columns])
""", language='python')

        else:
            st.success("✅ No scaling needed!")

    with tab4:
        # Feature engineering suggestions
        st.subheader("➕ New Feature Suggestions")

        suggestions = fe_report['engineering_suggestions']

        # Interaction features
        if suggestions['interaction_features']:
            st.write("**🔗 Interaction Features:**")
            for interaction in suggestions['interaction_features']:
                with st.expander(f"{interaction['feature1']} × {interaction['feature2']}"):
                    st.write(f"**Reason:** {interaction['reason']}")
                    st.write(f"**Suggested operations:** {', '.join(interaction['suggested_operations'])}")
                    st.code(f"""
# Create interaction features
df['{interaction['feature1']}_x_{interaction['feature2']}'] = df['{interaction['feature1']}'] * df['{interaction['feature2']}']
df['{interaction['feature1']}_plus_{interaction['feature2']}'] = df['{interaction['feature1']}'] + df['{interaction['feature2']}']
df['{interaction['feature1']}_ratio_{interaction['feature2']}'] = df['{interaction['feature1']}'] / (df['{interaction['feature2']}'] + 1e-8)
""", language='python')

        # Polynomial features
        if suggestions['polynomial_features']:
            st.write("**📐 Polynomial Features:**")
            for poly in suggestions['polynomial_features']:
                with st.expander(f"Polynomial features for {poly['feature']}"):
                    st.write(f"**Reason:** {poly['reason']}")
                    st.write(f"**Suggested degrees:** {poly['degrees']}")

        # Binning suggestions
        if suggestions['binning_suggestions']:
            st.write("**📊 Binning Suggestions:**")
            for binning in suggestions['binning_suggestions']:
                with st.expander(f"Binning for {binning['feature']}"):
                    st.write(f"**Reason:** {binning['reason']}")
                    st.write(f"**Methods:** {', '.join(binning['methods'])}")

        # Domain-specific features
        if suggestions['domain_specific_features']:
            st.write("**🎯 Domain-Specific Suggestions:**")
            for domain in suggestions['domain_specific_features']:
                with st.expander(f"{domain['category']}"):
                    st.write(f"**Features:** {', '.join(domain['features'])}")
                    for suggestion in domain['suggestions']:
                        st.write(f"• {suggestion}")

    with tab5:
        # Features to drop
        st.subheader("🗑️ Features to Remove")

        to_drop = fe_report['features_to_drop']

        if to_drop:
            drop_df = pd.DataFrame(to_drop)
            st.dataframe(drop_df, use_container_width=True)

            # Severity breakdown
            severity_counts = pd.Series([item['severity'] for item in to_drop]).value_counts()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("High Severity", severity_counts.get('High', 0))
            with col2:
                st.metric("Medium Severity", severity_counts.get('Medium', 0))
            with col3:
                st.metric("Low Severity", severity_counts.get('Low', 0))

            # Implementation
            st.code(f"""
# Drop recommended features
features_to_drop = {[item['feature'] for item in to_drop]}
df_cleaned = df.drop(columns=features_to_drop)
""", language='python')

        else:
            st.success("✅ No features recommended for removal!")

def display_model_readiness_dashboard():
    """Display model readiness dashboard"""
    if not st.session_state.data_loaded:
        return

    st.markdown('<div class="section-header">📈 Model Readiness Assessment</div>', unsafe_allow_html=True)

    # Calculate or retrieve model readiness
    if st.session_state.model_readiness is None:
        with st.spinner('Calculating model readiness...'):
            st.session_state.model_readiness = calculate_comprehensive_readiness_score(
                st.session_state.quality_report,
                st.session_state.target_analysis,
                st.session_state.feature_engineering_report,
                st.session_state.df
            )

    readiness = st.session_state.model_readiness

    # Overall score display
    st.subheader("🎯 Overall Readiness Score")

    col1, col2 = st.columns([1, 2])

    with col1:
        overall_score = readiness['overall_score']
        max_score = 100

        # Create a large score display
        if overall_score >= 75:
            st.success(f"## {overall_score}/100")
            st.success(f"**{readiness['interpretation']}**")
        elif overall_score >= 50:
            st.warning(f"## {overall_score}/100")
            st.warning(f"**{readiness['interpretation']}**")
        else:
            st.error(f"## {overall_score}/100")
            st.error(f"**{readiness['interpretation']}**")

        st.write(readiness['description'])

    with col2:
        # Score breakdown
        st.write("**Score Breakdown:**")
        categories = readiness['category_scores']

        breakdown_data = []
        for category, score in categories.items():
            max_scores = {
                'data_quality': 25,
                'target_quality': 25,
                'feature_quality': 25,
                'data_leakage': 15,
                'engineering_readiness': 10
            }
            max_score = max_scores.get(category, 25)
            percentage = (score / max_score * 100) if max_score > 0 else 0

            breakdown_data.append({
                'Category': category.replace('_', ' ').title(),
                'Score': f"{score}/{max_score}",
                'Percentage': f"{percentage:.1f}%"
            })

        breakdown_df = pd.DataFrame(breakdown_data)
        st.dataframe(breakdown_df, use_container_width=True)

    st.markdown("---")

    # Detailed analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Detailed Assessment", "⚠️ Issues", "💡 Recommendations", "🚀 Next Steps"
    ])

    with tab1:
        # Detailed component scores
        st.subheader("📊 Component Analysis")

        for category, assessment in readiness['detailed_assessment'].items():
            with st.expander(f"{category.replace('_', ' ').title()} - Score: {assessment['score']}/{assessment['max_score']}"):

                if 'components' in assessment:
                    comp_df = pd.DataFrame([
                        {'Component': comp.replace('_', ' ').title(), 'Score': score}
                        for comp, score in assessment['components'].items()
                    ])
                    st.dataframe(comp_df, use_container_width=True)

                if assessment.get('issues'):
                    st.write("**Issues:**")
                    for issue in assessment['issues']:
                        st.write(f"• {issue}")

                if assessment.get('recommendations'):
                    st.write("**Recommendations:**")
                    for rec in assessment['recommendations']:
                        st.write(f"• {rec}")

    with tab2:
        # Critical issues
        st.subheader("⚠️ Critical Issues")

        critical_issues = readiness['critical_issues']
        if critical_issues:
            for issue in critical_issues:
                st.error(f"🚨 {issue}")
        else:
            st.success("✅ No critical issues detected!")

        # All issues
        st.subheader("📋 All Issues")
        all_issues = []
        for assessment in readiness['detailed_assessment'].values():
            all_issues.extend(assessment.get('issues', []))

        if all_issues:
            for issue in all_issues:
                st.write(f"• {issue}")
        else:
            st.success("✅ No issues detected!")

    with tab3:
        # Priority recommendations
        st.subheader("💡 Priority Actions")

        priority_actions = readiness['priority_actions']
        if priority_actions:
            for i, action in enumerate(priority_actions, 1):
                st.write(f"**{i}.** {action}")
        else:
            st.success("✅ No immediate actions required!")

        # Strengths
        st.subheader("💪 Dataset Strengths")
        strengths = readiness['strengths']
        if strengths:
            for strength in strengths:
                st.success(f"✅ {strength}")
        else:
            st.info("No specific strengths identified")

    with tab4:
        # Next steps
        st.subheader("🚀 Recommended Next Steps")

        next_steps = readiness['next_steps']
        for i, step in enumerate(next_steps, 1):
            st.write(f"**{i}.** {step}")

        # Action timeline based on score
        st.subheader("⏱️ Suggested Timeline")
        if overall_score >= 75:
            st.info("🚀 **Ready to proceed immediately** - You can start model development now")
        elif overall_score >= 60:
            st.warning("⏳ **1-2 days of prep work** - Address key issues before modeling")
        elif overall_score >= 40:
            st.warning("📅 **1 week of data work** - Significant improvements needed")
        else:
            st.error("🛑 **2+ weeks of data work** - Major data quality issues to resolve")

def display_leakage_detection_dashboard():
    """Display comprehensive data leakage detection dashboard."""
    if not st.session_state.data_loaded:
        st.warning("Please upload data first.")
        return

    st.title("🚨 Advanced Data Leakage Detection")
    st.markdown("Comprehensive analysis to identify potential data leakage that could lead to overly optimistic model performance.")

    df = st.session_state.df

    # Configuration section
    with st.expander("🔧 Analysis Configuration", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            # Target column selection
            target_col = st.selectbox(
                "Target Column (optional)",
                ["None"] + list(df.columns),
                key="leakage_target",
                help="Select the target variable for enhanced leakage detection"
            )
            target_col = None if target_col == "None" else target_col

            # Date columns
            date_cols = st.multiselect(
                "Date/Time Columns",
                df.columns.tolist(),
                help="Select columns containing date/time information"
            )

        with col2:
            # ID columns
            id_cols = st.multiselect(
                "Identifier Columns",
                df.columns.tolist(),
                help="Select columns containing identifiers (IDs, keys, etc.)"
            )

            # Run analysis button
            run_analysis = st.button("🔍 Run Leakage Analysis", type="primary")

    # Run analysis or show cached results
    if run_analysis or 'leakage_results' not in st.session_state:
        with st.spinner("🔍 Analyzing potential data leakage..."):
            try:
                # Run comprehensive leakage detection
                leakage_results = comprehensive_leakage_detection(
                    df=df,
                    target_col=target_col,
                    date_cols=date_cols,
                    id_cols=id_cols
                )
                st.session_state.leakage_results = leakage_results
                st.session_state.leakage_summary_stats = get_leakage_summary_stats(leakage_results)

            except Exception as e:
                st.error(f"Error during leakage analysis: {str(e)}")
                return

    if 'leakage_results' in st.session_state:
        results = st.session_state.leakage_results
        summary_stats = st.session_state.leakage_summary_stats

        # Overall Risk Assessment
        st.header("📊 Overall Risk Assessment")

        risk_level = results['overall_leakage_risk']
        risk_score = results['risk_score']

        # Risk level indicator
        risk_colors = {
            'Minimal': 'green',
            'Low': 'lightgreen',
            'Medium': 'yellow',
            'High': 'orange',
            'Critical': 'red'
        }

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Risk Level", risk_level, delta=None)
            st.progress(risk_score / 100)

        with col2:
            st.metric("Risk Score", f"{risk_score:.1f}/100", delta=None)

        with col3:
            st.metric("Suspicious Features", summary_stats['total_suspicious_features'])

        with col4:
            st.metric("Total Issues", summary_stats['total_issues'])

        # Risk breakdown by category
        st.subheader("🎯 Risk Breakdown by Category")

        leakage_types = results['leakage_types']
        breakdown_data = []

        for leak_type, analysis in leakage_types.items():
            if isinstance(analysis, dict) and 'risk_points' in analysis:
                type_name = leak_type.replace('_', ' ').title()
                risk_pct = (analysis['risk_points'] / analysis.get('max_risk_points', 1)) * 100
                breakdown_data.append({
                    'Category': type_name,
                    'Risk Points': analysis['risk_points'],
                    'Max Points': analysis.get('max_risk_points', 0),
                    'Risk %': f"{risk_pct:.1f}%"
                })

        if breakdown_data:
            breakdown_df = pd.DataFrame(breakdown_data)
            st.dataframe(breakdown_df, use_container_width=True)

        # Detailed Analysis Tabs
        st.header("🔍 Detailed Analysis")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎯 Target Leakage",
            "🕐 Temporal Leakage",
            "📋 Data Duplication",
            "🔗 Feature Leakage",
            "🆔 Identifier Leakage",
            "📊 Statistical Anomalies"
        ])

        # Target Leakage Tab
        with tab1:
            target_analysis = leakage_types['target_leakage']
            if target_analysis.get('issues'):
                st.warning(f"⚠️ Found {len(target_analysis['issues'])} target leakage issues")

                for issue in target_analysis['issues']:
                    st.write(f"• {issue}")

                if target_analysis.get('perfect_predictors'):
                    st.subheader("🚨 Perfect Predictors")
                    st.write("These features can perfectly predict the target:")
                    for col in target_analysis['perfect_predictors']:
                        st.code(col)

                if target_analysis.get('near_perfect_predictors'):
                    st.subheader("⚠️ Near-Perfect Predictors")
                    near_perfect_df = pd.DataFrame(target_analysis['near_perfect_predictors'])
                    if not near_perfect_df.empty:
                        st.dataframe(near_perfect_df, use_container_width=True)

            else:
                st.success("✅ No target leakage detected")

        # Temporal Leakage Tab
        with tab2:
            temporal_analysis = leakage_types['temporal_leakage']
            if temporal_analysis.get('issues'):
                st.warning(f"⚠️ Found {len(temporal_analysis['issues'])} temporal issues")

                for issue in temporal_analysis['issues']:
                    st.write(f"• {issue}")

                if temporal_analysis.get('future_info_columns'):
                    st.subheader("🔮 Columns with Future Information")
                    future_df = pd.DataFrame(temporal_analysis['future_info_columns'])
                    if not future_df.empty:
                        st.dataframe(future_df, use_container_width=True)

            else:
                st.success("✅ No temporal leakage detected")

        # Data Duplication Tab
        with tab3:
            dup_analysis = leakage_types['data_duplication']
            dup_stats = dup_analysis.get('duplicate_analysis', {})

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Exact Duplicates", dup_stats.get('exact_duplicates', 0))
            with col2:
                st.metric("Near Duplicates", dup_stats.get('near_duplicates', 0))
            with col3:
                st.metric("Duplicate %", f"{dup_stats.get('duplicate_percentage', 0):.1f}%")

            if dup_analysis.get('issues'):
                st.warning("⚠️ Duplication issues found:")
                for issue in dup_analysis['issues']:
                    st.write(f"• {issue}")
            else:
                st.success("✅ No significant duplication detected")

        # Feature Leakage Tab
        with tab4:
            feature_analysis = leakage_types['feature_leakage']
            if feature_analysis.get('issues'):
                st.warning(f"⚠️ Found {len(feature_analysis['issues'])} feature leakage issues")

                for issue in feature_analysis['issues']:
                    st.write(f"• {issue}")

                if feature_analysis.get('perfect_correlations'):
                    st.subheader("🔗 Perfect Correlations")
                    perfect_corr_df = pd.DataFrame(feature_analysis['perfect_correlations'])
                    if not perfect_corr_df.empty:
                        st.dataframe(perfect_corr_df, use_container_width=True)

                if feature_analysis.get('high_correlations'):
                    st.subheader("⚠️ High Correlations")
                    high_corr_df = pd.DataFrame(feature_analysis['high_correlations'])
                    if not high_corr_df.empty:
                        st.dataframe(high_corr_df, use_container_width=True)

            else:
                st.success("✅ No feature leakage detected")

        # Identifier Leakage Tab
        with tab5:
            id_analysis = leakage_types['identifier_leakage']
            if id_analysis.get('issues'):
                st.warning(f"⚠️ Found {len(id_analysis['issues'])} identifier issues")

                for issue in id_analysis['issues']:
                    st.write(f"• {issue}")

                if id_analysis.get('high_cardinality_features'):
                    st.subheader("📊 High Cardinality Features")
                    high_card_df = pd.DataFrame(id_analysis['high_cardinality_features'])
                    if not high_card_df.empty:
                        st.dataframe(high_card_df, use_container_width=True)

            else:
                st.success("✅ No identifier leakage detected")

        # Statistical Anomalies Tab
        with tab6:
            stat_analysis = leakage_types['statistical_leakage']
            if stat_analysis.get('issues'):
                st.warning(f"⚠️ Found {len(stat_analysis['issues'])} statistical anomalies")

                for issue in stat_analysis['issues']:
                    st.write(f"• {issue}")

                if stat_analysis.get('distribution_anomalies'):
                    st.subheader("📈 Distribution Anomalies")
                    dist_anom_df = pd.DataFrame(stat_analysis['distribution_anomalies'])
                    if not dist_anom_df.empty:
                        st.dataframe(dist_anom_df, use_container_width=True)

            else:
                st.success("✅ No statistical anomalies detected")

        # Suspicious Features Summary
        if results['suspicious_features']:
            st.header("⚠️ Suspicious Features Summary")
            st.write("The following features have been flagged for potential leakage:")

            suspicious_df = pd.DataFrame({
                'Feature': results['suspicious_features'],
                'Action': ['Review for leakage'] * len(results['suspicious_features'])
            })
            st.dataframe(suspicious_df, use_container_width=True)

        # Recommendations
        st.header("💡 Recommendations")
        if results['recommendations']:
            for i, rec in enumerate(results['recommendations'], 1):
                st.write(f"{i}. {rec}")
        else:
            st.success("✅ No specific recommendations - dataset appears clean")

        # Export Results
        st.header("📥 Export Results")

        # Create export data
        export_data = {
            'overall_assessment': {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'suspicious_features_count': len(results['suspicious_features']),
                'total_issues': len(results['detailed_findings'])
            },
            'detailed_findings': results['detailed_findings'],
            'suspicious_features': results['suspicious_features'],
            'recommendations': results['recommendations']
        }

        export_json = pd.DataFrame([export_data]).to_json(orient='records', indent=2)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📄 Download Detailed Report (JSON)",
                data=export_json,
                file_name=f"leakage_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        with col2:
            # Create summary text report
            summary_text = f"""Data Leakage Analysis Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL ASSESSMENT:
- Risk Level: {risk_level}
- Risk Score: {risk_score:.1f}/100
- Suspicious Features: {len(results['suspicious_features'])}
- Total Issues: {len(results['detailed_findings'])}

DETAILED FINDINGS:
"""
            for finding in results['detailed_findings']:
                summary_text += f"• {finding}\n"

            summary_text += f"\nRECOMMENDATIONS:\n"
            for i, rec in enumerate(results['recommendations'], 1):
                summary_text += f"{i}. {rec}\n"

            st.download_button(
                label="📝 Download Summary Report (TXT)",
                data=summary_text,
                file_name=f"leakage_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

def display_reports_dashboard():
    """Display reports generation dashboard"""
    if not st.session_state.data_loaded:
        return

    st.markdown('<div class="section-header">📄 Reports & Export</div>', unsafe_allow_html=True)

    st.subheader("📊 Available Reports")

    # Report status
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Analysis Status:**")
        reports_available = []

        if st.session_state.quality_report:
            reports_available.append("✅ Data Quality Report")
        else:
            reports_available.append("❌ Data Quality Report")

        if st.session_state.eda_report:
            reports_available.append("✅ EDA Report")
        else:
            reports_available.append("❌ EDA Report")

        if st.session_state.target_analysis:
            reports_available.append("✅ Target Analysis")
        else:
            reports_available.append("❌ Target Analysis")

        if st.session_state.feature_engineering_report:
            reports_available.append("✅ Feature Engineering")
        else:
            reports_available.append("❌ Feature Engineering")

        if st.session_state.model_readiness:
            reports_available.append("✅ Model Readiness")
        else:
            reports_available.append("❌ Model Readiness")

        for status in reports_available:
            st.write(status)

    with col2:
        st.write("**Export Options:**")
        st.write("📋 Summary Report (Text)")
        st.write("📊 Analysis Results (JSON)")
        st.write("🐍 Preprocessing Code (Python)")
        st.write("📈 Visualizations (Images)")

    st.markdown("---")

    # Generate reports
    tab1, tab2, tab3 = st.tabs(["📋 Summary Report", "📊 Detailed Exports", "🐍 Code Generation"])

    with tab1:
        st.subheader("📋 Executive Summary Report")

        if st.button("📄 Generate Summary Report", type="primary"):
            with st.spinner("Generating summary report..."):

                report_content = f"""
# 📊 Data Analysis Summary Report

**Dataset:** {st.session_state.filename}
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📈 Key Metrics

- **Dataset Shape:** {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]} columns
- **Memory Usage:** {st.session_state.metadata['memory_usage_mb']:.1f} MB
- **Missing Data:** {st.session_state.quality_report['missing_values']['overall_missing_percentage']:.1f}%
- **Duplicate Rows:** {st.session_state.quality_report['duplicates']['duplicate_percentage']:.1f}%

## 🎯 Data Quality Score: {st.session_state.quality_report['quality_score']['overall_score']}/100
**Interpretation:** {st.session_state.quality_report['quality_score']['interpretation']}

## 📊 Feature Summary
- **Numeric Features:** {len(st.session_state.feature_engineering_report['feature_classification']['numeric_features'])}
- **Categorical Features:** {len(st.session_state.feature_engineering_report['feature_classification']['categorical_features'])}
- **Features Needing Encoding:** {st.session_state.feature_engineering_report['summary']['recommendations_count']['encoding']}
- **Features to Drop:** {st.session_state.feature_engineering_report['summary']['recommendations_count']['to_drop']}
"""

                if st.session_state.target_analysis:
                    target_info = st.session_state.target_analysis['basic_info']
                    report_content += f"""
## 🎯 Target Analysis
- **Target Variable:** {target_info['target_column'] if 'target_column' in target_info else st.session_state.selected_target}
- **Task Type:** {target_info['suggested_task_type'].title()}
- **Missing Target Values:** {target_info['missing_percentage']:.1f}%
"""

                if st.session_state.model_readiness:
                    readiness = st.session_state.model_readiness
                    report_content += f"""
## 🚀 Model Readiness Score: {readiness['overall_score']}/100
**Status:** {readiness['interpretation']}

### Priority Actions:
"""
                    for i, action in enumerate(readiness['priority_actions'][:5], 1):
                        report_content += f"{i}. {action}\n"

                report_content += f"""
## 💡 Key Recommendations

### Data Quality:
- Address {st.session_state.quality_report['missing_values']['total_missing']:,} missing values
- Remove {st.session_state.quality_report['duplicates']['duplicate_rows']:,} duplicate rows
- Handle {st.session_state.quality_report['outliers_iqr']['summary']['total_outliers']:,} outliers

### Feature Engineering:
- Encode {st.session_state.feature_engineering_report['summary']['recommendations_count']['encoding']} categorical features
- Scale {st.session_state.feature_engineering_report['summary']['recommendations_count']['scaling']} numeric features
- Consider creating {len(st.session_state.feature_engineering_report['engineering_suggestions']['interaction_features'])} interaction features

---
*Report generated by EDA Tool*
"""

                st.text_area("📄 Summary Report", report_content, height=400)

                # Download button
                st.download_button(
                    label="💾 Download Summary Report",
                    data=report_content,
                    file_name=f"eda_summary_{st.session_state.filename.split('.')[0]}.txt",
                    mime="text/plain"
                )

    with tab2:
        st.subheader("📊 Detailed Data Exports")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Export Analysis Results (JSON)"):
                import json

                export_data = {
                    'metadata': st.session_state.metadata,
                    'quality_report': st.session_state.quality_report,
                    'eda_report': st.session_state.eda_report,
                    'target_analysis': st.session_state.target_analysis,
                    'feature_engineering': st.session_state.feature_engineering_report,
                    'model_readiness': st.session_state.model_readiness
                }

                # Convert to JSON (handling numpy types and circular references)
                def convert_to_serializable(obj, seen=None):
                    """Convert complex objects to JSON-serializable format with circular reference handling"""
                    if seen is None:
                        seen = set()

                    # Check for circular reference
                    obj_id = id(obj)
                    if obj_id in seen:
                        return "<circular reference>"

                    # Handle basic types
                    if obj is None or isinstance(obj, (bool, int, float, str)):
                        return obj

                    # Handle numpy types
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.bool_):
                        return bool(obj)

                    # Mark this object as seen
                    seen.add(obj_id)

                    try:
                        # Handle pandas objects
                        if isinstance(obj, pd.Series):
                            result = obj.to_dict()
                            seen.remove(obj_id)
                            return result
                        elif isinstance(obj, pd.DataFrame):
                            result = obj.to_dict(orient='records')
                            seen.remove(obj_id)
                            return result

                        # Handle dictionaries
                        elif isinstance(obj, dict):
                            result = {k: convert_to_serializable(v, seen.copy()) for k, v in obj.items()}
                            seen.remove(obj_id)
                            return result

                        # Handle lists and tuples
                        elif isinstance(obj, (list, tuple)):
                            result = [convert_to_serializable(item, seen.copy()) for item in obj]
                            seen.remove(obj_id)
                            return result

                        # For other objects, try to convert to string
                        else:
                            seen.remove(obj_id)
                            return str(obj)
                    except Exception as e:
                        seen.remove(obj_id)
                        return f"<error converting: {str(e)}>"

                try:
                    # Convert export data with circular reference handling
                    serializable_data = convert_to_serializable(export_data)
                    json_str = json.dumps(serializable_data, indent=2)

                    st.download_button(
                        label="💾 Download JSON",
                        data=json_str,
                        file_name=f"eda_analysis_{st.session_state.filename.split('.')[0]}.json",
                        mime="application/json"
                    )
                    st.success("✅ Export data prepared successfully!")
                except Exception as e:
                    st.error(f"❌ Error exporting data: {str(e)}")

        with col2:
            if st.button("📈 Export Feature Info (CSV)"):
                if st.session_state.feature_engineering_report:
                    feature_details = st.session_state.feature_engineering_report['feature_classification']['feature_details']

                    export_list = []
                    for feature, details in feature_details.items():
                        export_list.append({
                            'Feature': feature,
                            'Type': details['feature_type'],
                            'Data_Type': details['data_type'],
                            'Unique_Count': details['unique_count'],
                            'Unique_Ratio': details['unique_ratio'],
                            'Missing_Count': details['missing_count'],
                            'Missing_Ratio': details['missing_ratio'],
                            'Notes': details.get('note', '')
                        })

                    feature_df = pd.DataFrame(export_list)
                    csv_str = feature_df.to_csv(index=False)

                    st.download_button(
                        label="💾 Download CSV",
                        data=csv_str,
                        file_name=f"feature_info_{st.session_state.filename.split('.')[0]}.csv",
                        mime="text/csv"
                    )

    with tab3:
        st.subheader("🐍 Generate Preprocessing Code")

        if st.button("🔧 Generate Python Code", type="primary"):
            try:
                # Get missing data percentage safely
                missing_pct = 0.0
                if st.session_state.quality_report and 'missing_values' in st.session_state.quality_report:
                    missing_pct = st.session_state.quality_report['missing_values'].get('overall_missing_percentage', 0.0)

                code_content = f'''"""
Data Preprocessing Pipeline
Generated by EDA Tool for: {st.session_state.filename}
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    """
    Preprocessing pipeline based on EDA analysis
    """
    df_processed = df.copy()

    # 1. Handle missing values
    # Based on analysis: {missing_pct:.1f}% missing data
'''

                if st.session_state.feature_engineering_report:
                    # Add encoding recommendations
                    encoding_recs = st.session_state.feature_engineering_report['encoding_recommendations']['categorical_encoding']

                    if encoding_recs:
                        code_content += '''

    # 2. Categorical Encoding
'''
                        for feature, rec in list(encoding_recs.items())[:3]:  # Limit to first 3 examples
                            if 'One-Hot' in rec['method']:
                                code_content += f'''    # {rec['method']} for {feature}
    df_processed = pd.get_dummies(df_processed, columns=['{feature}'], prefix='{feature}')
'''
                            elif 'Label' in rec['method']:
                                code_content += f'''    # {rec['method']} for {feature}
    le_{feature} = LabelEncoder()
    df_processed['{feature}_encoded'] = le_{feature}.fit_transform(df_processed['{feature}'].fillna('Unknown'))
'''

                    # Add scaling recommendations
                    scaling_recs = st.session_state.feature_engineering_report['encoding_recommendations']['scaling_recommendations']

                    if scaling_recs:
                        numeric_features = list(scaling_recs.keys())[:5]  # Limit to first 5
                        code_content += f'''

    # 3. Feature Scaling
    numeric_features = {numeric_features}
    scaler = StandardScaler()
    df_processed[numeric_features] = scaler.fit_transform(df_processed[numeric_features])
'''

                # Add target analysis code if available
                if st.session_state.target_analysis:
                    target_col = st.session_state.selected_target
                    code_content += f'''

    return df_processed

def prepare_for_modeling(df):
    """
    Prepare data for machine learning
    """
    # Preprocess the data
    df_processed = preprocess_data(df)

    # Separate features and target
    target_col = '{target_col}'
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test

# Usage:
# df = pd.read_csv('{st.session_state.filename}')
# X_train, X_test, y_train, y_test = prepare_for_modeling(df)
'''

                st.code(code_content, language='python')

                st.download_button(
                    label="💾 Download Python Code",
                    data=code_content,
                    file_name=f"preprocessing_pipeline_{st.session_state.filename.split('.')[0]}.py",
                    mime="text/plain"
                )
                st.success("✅ Code generated successfully!")

            except Exception as e:
                st.error(f"❌ Error generating code: {str(e)}")

def main():
    """Main application function"""
    initialize_session_state()

    # Header
    st.markdown('<div class="main-header">📊 Automated Data Quality & EDA Tool</div>', unsafe_allow_html=True)
    st.markdown("**Comprehensive data analysis and quality assessment for machine learning readiness**")
    st.markdown("---")

    # Sidebar
    display_sidebar()

    # Main content based on current section
    current_section = st.session_state.current_section

    # If no data is loaded, force upload section
    if not st.session_state.data_loaded:
        st.markdown("## 🚀 Welcome to the AI-Powered EDA Tool!")
        st.markdown("""
        This tool helps you quickly assess your dataset's quality and readiness for machine learning.

        **Core Features:**
        - 📁 **Data Upload**: Support for CSV and Excel files with automatic encoding detection
        - 📊 **Data Quality Analysis**: Missing values, duplicates, outliers detection
        - 🔍 **Exploratory Data Analysis**: Distributions, correlations, statistical summaries
        - 🛠️ **Feature Engineering**: Automated recommendations for encoding and scaling
        - 📈 **Model Readiness Score**: Overall assessment with actionable insights
        - 📄 **Professional Reports**: Export findings as PDF or Excel

        **🤖 AI Features (Privacy-First):**
        - 🔒 **100% Local Processing** - Your data never leaves your computer
        - 💬 **AI Chat Assistant** - Ask questions about your data in natural language
        - 🧠 **Auto-Generated Insights** - AI-powered data analysis and recommendations
        - 📝 **Code Generation** - Generate Python/Pandas code from natural language
        - 🎯 **Smart Recommendations** - AI-powered data cleaning and feature engineering

        **Get Started:** Upload your dataset using the file uploader below, or configure AI features in the sidebar.
        """)
        display_upload_section()

    elif current_section == "upload":
        # File upload section
        display_upload_section()

    elif current_section == "ai_setup":
        # AI Setup wizard - available anytime
        display_ai_setup_wizard()

    elif current_section == "ai_chat":
        # AI Chat Assistant - available anytime
        display_ai_chat()

    elif current_section == "ai_insights":
        # AI-Generated Insights - available anytime
        display_ai_insights()

    elif current_section == "data_cleaning":
        # AI-Powered Data Cleaning - available anytime
        display_data_cleaning()

    elif current_section == "nl_query":
        # Natural Language Query Translator - available anytime
        display_nl_query_translator()

    elif current_section == "ai_feature_eng":
        # AI-Powered Feature Engineering - available anytime
        display_feature_engineering_ai()

    elif current_section == "anomaly_explanation":
        # AI-Powered Anomaly Explanation - available anytime
        display_anomaly_explanation()

    elif current_section == "overview" and st.session_state.data_loaded:
        # Data overview and preview
        display_data_overview()
        st.markdown("---")
        display_data_preview()

    elif current_section == "quality" and st.session_state.data_loaded:
        # Data quality dashboard
        display_data_quality_dashboard()

    elif current_section == "eda" and st.session_state.data_loaded:
        # EDA dashboard
        display_eda_dashboard()

    elif current_section == "target" and st.session_state.data_loaded:
        # Target analysis dashboard
        display_target_analysis_dashboard()

    elif current_section == "features" and st.session_state.data_loaded:
        # Feature engineering dashboard
        display_feature_engineering_dashboard()

    elif current_section == "readiness" and st.session_state.data_loaded:
        # Model readiness dashboard
        display_model_readiness_dashboard()

    elif current_section == "reports" and st.session_state.data_loaded:
        # Reports dashboard
        display_reports_dashboard()

    elif current_section == "leakage":
        display_leakage_detection_dashboard()

    else:
        # Default to upload if no valid section
        st.session_state.current_section = "upload"
        st.rerun()

if __name__ == "__main__":
    main()