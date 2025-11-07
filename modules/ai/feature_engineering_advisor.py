"""
AI-Powered Feature Engineering Advisor
Provides intelligent feature engineering recommendations based on target analysis
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from scipy import stats
from .llm_integration import LocalLLM, AIResponse


@dataclass
class FeatureSuggestion:
    """Represents a feature engineering suggestion"""
    id: str
    feature_type: str  # 'binning', 'transformation', 'interaction', 'time', 'encoding'
    priority: int  # 1-5 stars
    source_columns: List[str]
    new_feature_name: str
    description: str
    rationale: str  # AI-generated explanation
    expected_impact: str
    code: str
    preview: Dict[str, Any]
    confidence: float


class FeatureEngineeringAdvisor:
    """AI-powered feature engineering advisor"""

    def __init__(self, model_name: str = 'phi3:mini', temperature: float = 0.7):
        """
        Initialize Feature Engineering Advisor

        Args:
            model_name: Name of the LLM model to use
            temperature: Sampling temperature for responses
        """
        self.llm = LocalLLM(model_name, temperature)
        self.suggestions: List[FeatureSuggestion] = []

    def is_available(self) -> bool:
        """Check if AI is available"""
        return self.llm.is_available()

    def analyze_features(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        task_type: Optional[str] = None  # 'classification' or 'regression'
    ) -> List[FeatureSuggestion]:
        """
        Analyze features and generate engineering suggestions

        Args:
            df: DataFrame to analyze
            target_column: Target variable column name
            task_type: Type of ML task (classification/regression)

        Returns:
            List of feature engineering suggestions
        """
        self.suggestions = []

        # Detect task type if not provided
        if target_column and not task_type:
            task_type = self._detect_task_type(df, target_column)

        # Generate suggestions for different feature types
        self.suggestions.extend(self._suggest_binning(df, target_column, task_type))
        self.suggestions.extend(self._suggest_transformations(df, target_column, task_type))
        self.suggestions.extend(self._suggest_interactions(df, target_column, task_type))
        self.suggestions.extend(self._suggest_time_features(df))
        self.suggestions.extend(self._suggest_encoding(df, target_column, task_type))

        # Sort by priority
        self.suggestions.sort(key=lambda x: x.priority, reverse=True)

        return self.suggestions

    def _detect_task_type(self, df: pd.DataFrame, target_column: str) -> str:
        """Detect if task is classification or regression"""
        if target_column not in df.columns:
            return 'unknown'

        target = df[target_column]

        # Check if target is numeric
        if pd.api.types.is_numeric_dtype(target):
            # If few unique values, likely classification
            unique_ratio = target.nunique() / len(target)
            if unique_ratio < 0.05 or target.nunique() <= 10:
                return 'classification'
            else:
                return 'regression'
        else:
            return 'classification'

    def _suggest_binning(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        task_type: Optional[str]
    ) -> List[FeatureSuggestion]:
        """Suggest binning for numeric features"""
        suggestions = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Exclude target column
        if target_column:
            numeric_cols = [col for col in numeric_cols if col != target_column]

        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue

            # Suggest binning for columns with wide range
            if col_data.std() > 0:
                value_range = col_data.max() - col_data.min()
                if value_range > col_data.mean() * 2:  # High variance
                    # Determine appropriate bins
                    if 'age' in col.lower():
                        bins = [0, 18, 25, 35, 45, 55, 65, 100]
                        labels = ['Child', 'Young Adult', 'Adult', 'Middle Age', 'Senior', 'Elderly', 'Very Elderly']
                        bin_type = "age groups"
                    elif 'income' in col.lower() or 'salary' in col.lower():
                        q25, q50, q75 = col_data.quantile([0.25, 0.5, 0.75])
                        bins = [col_data.min(), q25, q50, q75, col_data.max()]
                        labels = ['Low', 'Medium-Low', 'Medium-High', 'High']
                        bin_type = "income brackets"
                    else:
                        # Use quantile-based binning
                        bins = 5
                        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
                        bin_type = "quantile bins"

                    new_feature_name = f"{col}_binned"

                    # Generate code
                    if isinstance(bins, list):
                        code = f"""# Bin {col} into {bin_type}
df['{new_feature_name}'] = pd.cut(
    df['{col}'],
    bins={bins},
    labels={labels},
    include_lowest=True
)
"""
                    else:
                        code = f"""# Bin {col} into {bins} quantile bins
df['{new_feature_name}'] = pd.qcut(
    df['{col}'],
    q={bins},
    labels={labels},
    duplicates='drop'
)
"""

                    # Generate preview
                    try:
                        if isinstance(bins, list):
                            preview_data = pd.cut(col_data, bins=bins, labels=labels, include_lowest=True)
                        else:
                            preview_data = pd.qcut(col_data, q=bins, labels=labels, duplicates='drop')

                        preview = {
                            'original_sample': col_data.head(5).tolist(),
                            'binned_sample': preview_data.head(5).tolist(),
                            'value_counts': preview_data.value_counts().to_dict()
                        }
                    except Exception:
                        preview = {}

                    rationale = self._generate_ai_rationale(
                        df, col, target_column, task_type,
                        f"binning into {bin_type}",
                        "Binning converts continuous values into discrete categories, which can help models detect non-linear patterns and improve interpretability."
                    )

                    suggestion = FeatureSuggestion(
                        id=f"bin_{col}_{datetime.now().timestamp()}",
                        feature_type='binning',
                        priority=4,
                        source_columns=[col],
                        new_feature_name=new_feature_name,
                        description=f"Bin '{col}' into {bin_type}",
                        rationale=rationale,
                        expected_impact="Captures non-linear relationships, improves model interpretability",
                        code=code,
                        preview=preview,
                        confidence=0.85
                    )
                    suggestions.append(suggestion)

        return suggestions

    def _suggest_transformations(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        task_type: Optional[str]
    ) -> List[FeatureSuggestion]:
        """Suggest mathematical transformations for skewed features"""
        suggestions = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Exclude target column
        if target_column:
            numeric_cols = [col for col in numeric_cols if col != target_column]

        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) == 0 or col_data.min() <= 0:
                continue

            # Check skewness
            skewness = col_data.skew()

            if abs(skewness) > 1:  # Highly skewed
                if skewness > 1:  # Right-skewed
                    transform_type = "log"
                    new_feature_name = f"{col}_log"
                    code = f"""# Log transformation for right-skewed data
df['{new_feature_name}'] = np.log1p(df['{col}'])  # log1p handles zeros
"""
                    description = f"Apply log transformation to '{col}' (right-skewed)"
                else:  # Left-skewed
                    transform_type = "square"
                    new_feature_name = f"{col}_squared"
                    code = f"""# Square transformation for left-skewed data
df['{new_feature_name}'] = df['{col}'] ** 2
"""
                    description = f"Apply square transformation to '{col}' (left-skewed)"

                # Generate preview
                try:
                    if transform_type == "log":
                        transformed = np.log1p(col_data)
                    else:
                        transformed = col_data ** 2

                    preview = {
                        'original_sample': col_data.head(5).tolist(),
                        'transformed_sample': transformed.head(5).tolist(),
                        'original_skew': float(skewness),
                        'transformed_skew': float(transformed.skew())
                    }
                except Exception:
                    preview = {}

                rationale = self._generate_ai_rationale(
                    df, col, target_column, task_type,
                    f"{transform_type} transformation",
                    f"This column has skewness of {skewness:.2f}. Transforming it will normalize the distribution and help linear models perform better."
                )

                suggestion = FeatureSuggestion(
                    id=f"transform_{col}_{datetime.now().timestamp()}",
                    feature_type='transformation',
                    priority=3,
                    source_columns=[col],
                    new_feature_name=new_feature_name,
                    description=description,
                    rationale=rationale,
                    expected_impact="Normalizes distribution, improves linear model performance",
                    code=code,
                    preview=preview,
                    confidence=0.80
                )
                suggestions.append(suggestion)

        return suggestions

    def _suggest_interactions(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        task_type: Optional[str]
    ) -> List[FeatureSuggestion]:
        """Suggest interaction features between columns"""
        suggestions = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Exclude target column
        if target_column:
            numeric_cols = [col for col in numeric_cols if col != target_column]

        # Common meaningful interactions
        interaction_patterns = [
            (['price', 'quantity'], 'multiply', 'total_value'),
            (['income', 'debt'], 'divide', 'debt_to_income_ratio'),
            (['length', 'width'], 'multiply', 'area'),
            (['distance', 'time'], 'divide', 'speed'),
        ]

        for pattern_cols, operation, suggested_name in interaction_patterns:
            # Find matching columns (case-insensitive partial match)
            matched_cols = []
            for pattern_col in pattern_cols:
                for df_col in numeric_cols:
                    if pattern_col.lower() in df_col.lower():
                        matched_cols.append(df_col)
                        break

            if len(matched_cols) == len(pattern_cols):
                col1, col2 = matched_cols[0], matched_cols[1]

                if operation == 'multiply':
                    new_feature_name = f"{col1}_x_{col2}"
                    code = f"""# Interaction: {col1} × {col2}
df['{new_feature_name}'] = df['{col1}'] * df['{col2}']
"""
                    description = f"Create interaction feature: {col1} × {col2}"
                    impact = "Captures multiplicative relationship"
                else:  # divide
                    new_feature_name = suggested_name
                    code = f"""# Ratio: {col1} / {col2}
df['{new_feature_name}'] = df['{col1}'] / (df['{col2}'] + 1)  # Add 1 to avoid division by zero
"""
                    description = f"Create ratio feature: {col1} / {col2}"
                    impact = "Captures relative relationship"

                # Generate preview
                try:
                    if operation == 'multiply':
                        result = df[col1] * df[col2]
                    else:
                        result = df[col1] / (df[col2] + 1)

                    preview = {
                        'col1_sample': df[col1].head(5).tolist(),
                        'col2_sample': df[col2].head(5).tolist(),
                        'result_sample': result.head(5).tolist()
                    }
                except Exception:
                    preview = {}

                rationale = self._generate_ai_rationale(
                    df, f"{col1} and {col2}", target_column, task_type,
                    f"{operation} interaction",
                    f"Combining {col1} and {col2} may reveal important relationships that individual features miss."
                )

                suggestion = FeatureSuggestion(
                    id=f"interact_{col1}_{col2}_{datetime.now().timestamp()}",
                    feature_type='interaction',
                    priority=5,
                    source_columns=[col1, col2],
                    new_feature_name=new_feature_name,
                    description=description,
                    rationale=rationale,
                    expected_impact=impact,
                    code=code,
                    preview=preview,
                    confidence=0.90
                )
                suggestions.append(suggestion)

        return suggestions

    def _suggest_time_features(self, df: pd.DataFrame) -> List[FeatureSuggestion]:
        """Suggest time-based features from datetime columns"""
        suggestions = []
        datetime_cols = df.select_dtypes(include=['datetime64']).columns

        for col in datetime_cols:
            # Extract various time features
            time_features = [
                ('day_of_week', f"df['{col}_day_of_week'] = df['{col}'].dt.dayofweek", "Day of week (0=Monday)"),
                ('month', f"df['{col}_month'] = df['{col}'].dt.month", "Month (1-12)"),
                ('quarter', f"df['{col}_quarter'] = df['{col}'].dt.quarter", "Quarter (1-4)"),
                ('is_weekend', f"df['{col}_is_weekend'] = df['{col}'].dt.dayofweek.isin([5, 6]).astype(int)", "Weekend indicator"),
                ('hour', f"df['{col}_hour'] = df['{col}'].dt.hour", "Hour of day (0-23)"),
            ]

            for feature_name, code_line, description in time_features:
                new_feature_name = f"{col}_{feature_name}"

                code = f"""# Extract {description} from {col}
{code_line}
"""

                # Generate preview
                try:
                    if feature_name == 'day_of_week':
                        preview_data = df[col].dt.dayofweek
                    elif feature_name == 'month':
                        preview_data = df[col].dt.month
                    elif feature_name == 'quarter':
                        preview_data = df[col].dt.quarter
                    elif feature_name == 'is_weekend':
                        preview_data = df[col].dt.dayofweek.isin([5, 6]).astype(int)
                    elif feature_name == 'hour':
                        preview_data = df[col].dt.hour
                    else:
                        preview_data = None

                    if preview_data is not None:
                        preview = {
                            'sample': preview_data.head(5).tolist(),
                            'value_counts': preview_data.value_counts().head(10).to_dict()
                        }
                    else:
                        preview = {}
                except Exception:
                    preview = {}

                suggestion = FeatureSuggestion(
                    id=f"time_{col}_{feature_name}_{datetime.now().timestamp()}",
                    feature_type='time',
                    priority=4,
                    source_columns=[col],
                    new_feature_name=new_feature_name,
                    description=f"Extract {description}",
                    rationale=f"Time-based features from {col} can reveal temporal patterns like seasonality, day-of-week effects, or time-of-day trends.",
                    expected_impact="Captures temporal patterns and cyclical behavior",
                    code=code,
                    preview=preview,
                    confidence=0.95
                )
                suggestions.append(suggestion)

        return suggestions

    def _suggest_encoding(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        task_type: Optional[str]
    ) -> List[FeatureSuggestion]:
        """Suggest categorical encoding strategies"""
        suggestions = []
        categorical_cols = df.select_dtypes(include=['object']).columns

        # Exclude target column if it's categorical
        if target_column and target_column in categorical_cols:
            categorical_cols = [col for col in categorical_cols if col != target_column]

        for col in categorical_cols:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue

            n_unique = col_data.nunique()

            # One-hot encoding for low cardinality
            if n_unique <= 10:
                new_feature_name = f"{col}_encoded"
                code = f"""# One-hot encode {col}
df = pd.get_dummies(df, columns=['{col}'], prefix='{col}', drop_first=True)
"""
                description = f"One-hot encode '{col}' ({n_unique} categories)"
                encoding_type = "one-hot"
                priority = 5

            # Frequency encoding for high cardinality
            else:
                new_feature_name = f"{col}_frequency"
                code = f"""# Frequency encode {col}
freq_map = df['{col}'].value_counts(normalize=True).to_dict()
df['{new_feature_name}'] = df['{col}'].map(freq_map)
"""
                description = f"Frequency encode '{col}' ({n_unique} categories)"
                encoding_type = "frequency"
                priority = 3

            # Generate preview
            try:
                if encoding_type == "one-hot":
                    encoded = pd.get_dummies(df[[col]], columns=[col], prefix=col, drop_first=True)
                    preview = {
                        'original_sample': col_data.head(5).tolist(),
                        'encoded_columns': encoded.columns.tolist()[:5],
                        'n_new_features': len(encoded.columns)
                    }
                else:
                    freq_map = col_data.value_counts(normalize=True).to_dict()
                    encoded = col_data.map(freq_map)
                    preview = {
                        'original_sample': col_data.head(5).tolist(),
                        'frequency_sample': encoded.head(5).tolist()
                    }
            except Exception:
                preview = {}

            rationale = f"Categorical variable '{col}' needs numeric encoding for ML models. {encoding_type.title()} encoding is appropriate for {n_unique} unique values."

            suggestion = FeatureSuggestion(
                id=f"encode_{col}_{datetime.now().timestamp()}",
                feature_type='encoding',
                priority=priority,
                source_columns=[col],
                new_feature_name=new_feature_name,
                description=description,
                rationale=rationale,
                expected_impact="Converts categorical data to numeric format for ML models",
                code=code,
                preview=preview,
                confidence=0.95
            )
            suggestions.append(suggestion)

        return suggestions

    def _generate_ai_rationale(
        self,
        df: pd.DataFrame,
        feature: str,
        target_column: Optional[str],
        task_type: Optional[str],
        transformation: str,
        default_rationale: str
    ) -> str:
        """Generate AI explanation for feature engineering suggestion"""

        if not self.is_available():
            return default_rationale

        # Create context for LLM
        context = f"""
Feature Engineering Task:
- Feature: {feature}
- Target: {target_column if target_column else 'Not specified'}
- Task Type: {task_type if task_type else 'Unknown'}
- Suggested Transformation: {transformation}
- Dataset: {len(df)} rows, {len(df.columns)} columns

Explain in 2-3 sentences why this transformation would be beneficial for model performance.
"""

        try:
            response = self.llm.generate(
                prompt=context,
                system_prompt="You are a feature engineering expert. Explain transformations clearly and concisely.",
                max_tokens=150
            )

            if response.success:
                return response.content
            else:
                return default_rationale

        except Exception:
            return default_rationale

    def apply_suggestion(
        self,
        df: pd.DataFrame,
        suggestion: FeatureSuggestion
    ) -> Tuple[bool, pd.DataFrame, str]:
        """
        Apply a feature engineering suggestion to the DataFrame

        Args:
            df: DataFrame to modify
            suggestion: Suggestion to apply

        Returns:
            Tuple of (success, modified_df, message)
        """
        try:
            df_copy = df.copy()

            # Execute the suggestion code
            local_vars = {'df': df_copy, 'pd': pd, 'np': np}
            exec(suggestion.code, {}, local_vars)
            df_result = local_vars['df']

            return True, df_result, f"Successfully created feature: {suggestion.new_feature_name}"

        except Exception as e:
            return False, df, f"Failed to create feature: {str(e)}"


def get_feature_history() -> List[Dict[str, Any]]:
    """Get feature engineering history from session state"""
    return st.session_state.get('feature_history', [])


def add_to_feature_history(action: Dict[str, Any]):
    """Add action to feature history"""
    if 'feature_history' not in st.session_state:
        st.session_state.feature_history = []

    st.session_state.feature_history.append(action)

    # Limit history to last 20 actions
    if len(st.session_state.feature_history) > 20:
        st.session_state.feature_history = st.session_state.feature_history[-20:]


def clear_feature_history():
    """Clear feature engineering history"""
    if 'feature_history' in st.session_state:
        st.session_state.feature_history = []


def display_feature_engineering_ai():
    """Display AI-powered feature engineering interface"""

    st.markdown('<div class="section-header">🛠️ AI Feature Engineering</div>', unsafe_allow_html=True)

    # Check if AI is available
    from .model_manager import ModelManager
    manager = ModelManager()

    if not manager.is_ollama_installed() or not manager.get_default_model():
        st.warning("⚠️ AI features not configured. Please set up AI first.")
        if st.button("🤖 Go to AI Setup"):
            st.session_state.current_section = "ai_setup"
            st.rerun()
        return

    # Check if data is loaded
    if not st.session_state.get('data_loaded', False):
        st.info("📁 Please upload a dataset first to use feature engineering.")
        return

    # Get default model
    default_model = manager.get_default_model()

    # Display feature info
    st.markdown("""
    **AI-Powered Feature Engineering** analyzes your features and provides:
    - 🎯 Intelligent transformation suggestions
    - 📊 Binning recommendations
    - 🔄 Interaction feature ideas
    - ⏰ Time-based feature extraction
    - 🏷️ Categorical encoding strategies
    - 🤖 AI-generated rationales
    """)

    # Target selection
    col1, col2 = st.columns([3, 1])
    with col1:
        target_column = st.selectbox(
            "Select Target Variable (optional):",
            options=['None'] + list(st.session_state.df.columns),
            help="Selecting a target helps generate more relevant features"
        )
        if target_column == 'None':
            target_column = None

    with col2:
        if target_column:
            task_type = st.radio(
                "Task Type:",
                options=['Auto-detect', 'Classification', 'Regression'],
                help="Type of ML task"
            )
            if task_type == 'Auto-detect':
                task_type = None
            else:
                task_type = task_type.lower()
        else:
            task_type = None

    # Main controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🔍 Analyze Features & Get Suggestions", type="primary"):
            with st.spinner("🤖 Analyzing features and generating suggestions..."):
                advisor = FeatureEngineeringAdvisor(model_name=default_model)

                # Analyze features
                suggestions = advisor.analyze_features(
                    df=st.session_state.df,
                    target_column=target_column,
                    task_type=task_type
                )

                # Store in session state
                st.session_state.feature_suggestions = suggestions
                st.session_state.feature_advisor = advisor
                st.session_state.feature_target = target_column

    with col2:
        if st.button("📜 View History"):
            st.session_state.show_feature_history = not st.session_state.get('show_feature_history', False)

    with col3:
        if st.button("🔄 Clear History"):
            clear_feature_history()
            st.success("History cleared!")
            st.rerun()

    st.markdown("---")

    # Display feature history if requested
    if st.session_state.get('show_feature_history', False):
        history = get_feature_history()
        if history:
            st.markdown("### 📜 Feature Engineering History")
            for i, action in enumerate(reversed(history)):
                with st.expander(f"Action {len(history) - i}: {action.get('feature_name', 'Unknown')}"):
                    st.markdown(f"**Type:** {action.get('feature_type', 'Unknown')}")
                    st.markdown(f"**Timestamp:** {action.get('timestamp', 'Unknown')}")
                    st.markdown(f"**Status:** {'✅ Success' if action.get('success', False) else '❌ Failed'}")
                    if action.get('message'):
                        st.info(action['message'])
            st.markdown("---")
        else:
            st.info("No feature engineering actions yet.")
            st.markdown("---")

    # Display suggestions
    if not st.session_state.get('feature_suggestions'):
        st.info("💡 Click 'Analyze Features' to get AI-powered feature engineering suggestions.")
        return

    suggestions = st.session_state.feature_suggestions

    if not suggestions:
        st.success("✅ No additional features suggested. Your current features look good!")
        return

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Suggestions", len(suggestions))
    with col2:
        high_priority = sum(1 for s in suggestions if s.priority >= 4)
        st.metric("High Priority", high_priority)
    with col3:
        feature_types = len(set(s.feature_type for s in suggestions))
        st.metric("Feature Types", feature_types)
    with col4:
        avg_confidence = sum(s.confidence for s in suggestions) / len(suggestions)
        st.metric("Avg Confidence", f"{avg_confidence*100:.0f}%")

    st.markdown("---")

    # Filter by type
    filter_type = st.multiselect(
        "Filter by Type:",
        options=['binning', 'transformation', 'interaction', 'time', 'encoding'],
        default=['binning', 'transformation', 'interaction', 'time', 'encoding']
    )

    filtered_suggestions = [s for s in suggestions if s.feature_type in filter_type]

    # Display each suggestion
    for suggestion in filtered_suggestions:
        # Priority stars
        priority_stars = "⭐" * suggestion.priority

        # Type emoji
        type_emojis = {
            'binning': '📊',
            'transformation': '🔄',
            'interaction': '🔗',
            'time': '⏰',
            'encoding': '🏷️'
        }
        type_emoji = type_emojis.get(suggestion.feature_type, '✨')

        # Suggestion header
        st.markdown(f"### {type_emoji} {priority_stars} {suggestion.description}")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**New Feature:** `{suggestion.new_feature_name}`")
            st.markdown(f"**Source:** {', '.join(suggestion.source_columns)}")
            st.markdown(f"**Type:** {suggestion.feature_type.title()}")
        with col2:
            confidence_pct = suggestion.confidence * 100
            st.metric("Confidence", f"{confidence_pct:.0f}%")

        # Suggestion details
        with st.expander("📋 View Details", expanded=True):
            # AI rationale
            st.markdown("**Why This Feature?**")
            st.info(suggestion.rationale)

            # Expected impact
            st.markdown(f"**Expected Impact:** {suggestion.expected_impact}")

            # Show code
            with st.expander("🐍 View Generated Code"):
                st.code(suggestion.code, language='python')

            # Preview
            if suggestion.preview:
                st.markdown("**Preview:**")

                if 'original_sample' in suggestion.preview and 'binned_sample' in suggestion.preview:
                    # Binning preview
                    preview_df = pd.DataFrame({
                        'Original': suggestion.preview['original_sample'],
                        'Binned': suggestion.preview['binned_sample']
                    })
                    st.dataframe(preview_df, use_container_width=True)

                    if 'value_counts' in suggestion.preview:
                        st.caption("**Distribution:**")
                        st.json(suggestion.preview['value_counts'])

                elif 'original_sample' in suggestion.preview and 'transformed_sample' in suggestion.preview:
                    # Transformation preview
                    preview_df = pd.DataFrame({
                        'Original': suggestion.preview['original_sample'],
                        'Transformed': suggestion.preview['transformed_sample']
                    })
                    st.dataframe(preview_df, use_container_width=True)

                    if 'original_skew' in suggestion.preview:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(f"Original Skew: {suggestion.preview['original_skew']:.2f}")
                        with col2:
                            st.caption(f"Transformed Skew: {suggestion.preview['transformed_skew']:.2f}")

                elif 'col1_sample' in suggestion.preview:
                    # Interaction preview
                    preview_df = pd.DataFrame({
                        suggestion.source_columns[0]: suggestion.preview['col1_sample'],
                        suggestion.source_columns[1]: suggestion.preview['col2_sample'],
                        suggestion.new_feature_name: suggestion.preview['result_sample']
                    })
                    st.dataframe(preview_df, use_container_width=True)

                else:
                    st.json(suggestion.preview)

        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("✅ Apply", key=f"apply_{suggestion.id}"):
                advisor = st.session_state.get('feature_advisor')
                if advisor:
                    success, df_result, message = advisor.apply_suggestion(
                        df=st.session_state.df,
                        suggestion=suggestion
                    )

                    if success:
                        # Update the dataframe
                        st.session_state.df = df_result

                        # Add to history
                        add_to_feature_history({
                            'feature_name': suggestion.new_feature_name,
                            'feature_type': suggestion.feature_type,
                            'success': True,
                            'message': message,
                            'timestamp': datetime.now().isoformat()
                        })

                        st.success(message)
                        st.info(f"💾 Feature '{suggestion.new_feature_name}' added! Data shape: {df_result.shape}")

                        # Clear current suggestions to force re-analysis
                        if 'feature_suggestions' in st.session_state:
                            del st.session_state.feature_suggestions

                        st.rerun()
                    else:
                        add_to_feature_history({
                            'feature_name': suggestion.new_feature_name,
                            'feature_type': suggestion.feature_type,
                            'success': False,
                            'message': message,
                            'timestamp': datetime.now().isoformat()
                        })
                        st.error(message)

        with col2:
            if st.button("⏭️ Skip", key=f"skip_{suggestion.id}"):
                st.info("Suggestion skipped.")

        st.markdown("---")
