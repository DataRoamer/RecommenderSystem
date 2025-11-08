"""
AI-Powered Anomaly Explainer
Provides intelligent explanations for outliers and anomalies with root cause analysis
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from scipy import stats
from .llm_integration import LocalLLM, AIResponse


@dataclass
class OutlierExplanation:
    """Represents an explanation for an outlier"""
    column: str
    value: Any
    row_index: int
    z_score: float
    percentile: float
    legitimacy: str  # 'legitimate', 'suspicious', 'error'
    explanation: str  # AI-generated explanation
    likely_causes: List[str]
    recommendation: str  # 'keep', 'remove', 'investigate'
    rationale: str
    confidence: float
    context: Dict[str, Any]


class AnomalyExplainer:
    """AI-powered anomaly and outlier explainer"""

    def __init__(self, model_name: str = 'phi3:mini', temperature: float = 0.7):
        """
        Initialize Anomaly Explainer

        Args:
            model_name: Name of the LLM model to use
            temperature: Sampling temperature for responses
        """
        self.llm = LocalLLM(model_name, temperature)

    def is_available(self) -> bool:
        """Check if AI is available"""
        return self.llm.is_available()

    def detect_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        method: str = 'iqr',  # 'iqr', 'zscore', or 'both'
        threshold: float = 1.5
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Detect outliers in a column

        Args:
            df: DataFrame
            column: Column name to analyze
            method: Detection method ('iqr', 'zscore', 'both')
            threshold: Threshold for outlier detection

        Returns:
            Tuple of (outlier_mask, stats_dict)
        """
        col_data = df[column].dropna()

        if not pd.api.types.is_numeric_dtype(col_data):
            return pd.Series([False] * len(df), index=df.index), {}

        stats_dict = {
            'mean': col_data.mean(),
            'median': col_data.median(),
            'std': col_data.std(),
            'min': col_data.min(),
            'max': col_data.max(),
            'q1': col_data.quantile(0.25),
            'q3': col_data.quantile(0.75)
        }

        if method in ['iqr', 'both']:
            # IQR method
            Q1 = stats_dict['q1']
            Q3 = stats_dict['q3']
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            iqr_outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            stats_dict['iqr_lower'] = lower_bound
            stats_dict['iqr_upper'] = upper_bound

        if method in ['zscore', 'both']:
            # Z-score method
            z_scores = np.abs(stats.zscore(col_data, nan_policy='omit'))
            zscore_outliers = pd.Series([False] * len(df), index=df.index)
            zscore_outliers.loc[col_data.index] = z_scores > 3

            stats_dict['zscore_threshold'] = 3

        # Combine methods
        if method == 'iqr':
            outlier_mask = iqr_outliers
        elif method == 'zscore':
            outlier_mask = zscore_outliers
        else:  # both
            outlier_mask = iqr_outliers | zscore_outliers

        return outlier_mask, stats_dict

    def analyze_outlier_patterns(
        self,
        df: pd.DataFrame,
        column: str,
        outlier_mask: pd.Series
    ) -> Dict[str, Any]:
        """
        Analyze patterns in outliers

        Args:
            df: DataFrame
            column: Column with outliers
            outlier_mask: Boolean mask of outliers

        Returns:
            Dictionary with pattern analysis
        """
        patterns = {
            'total_outliers': outlier_mask.sum(),
            'outlier_percentage': (outlier_mask.sum() / len(df)) * 100,
            'clustering': None,
            'temporal_pattern': None,
            'correlation_with_others': {}
        }

        outlier_df = df[outlier_mask]

        # Check for clustering
        if len(outlier_df) > 1:
            outlier_values = outlier_df[column].values
            value_range = outlier_values.max() - outlier_values.min()
            overall_range = df[column].max() - df[column].min()

            if value_range < overall_range * 0.1:
                patterns['clustering'] = 'tight'  # Outliers are clustered together
            elif value_range < overall_range * 0.3:
                patterns['clustering'] = 'moderate'
            else:
                patterns['clustering'] = 'dispersed'

        # Check temporal patterns (if datetime column exists)
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            date_col = datetime_cols[0]
            outlier_dates = outlier_df[date_col]

            # Check if outliers occur on weekends
            if hasattr(outlier_dates.dt, 'dayofweek'):
                weekend_outliers = outlier_dates.dt.dayofweek.isin([5, 6]).sum()
                weekend_ratio = weekend_outliers / len(outlier_dates) if len(outlier_dates) > 0 else 0

                if weekend_ratio > 0.7:
                    patterns['temporal_pattern'] = 'weekend_spike'
                elif weekend_ratio < 0.2:
                    patterns['temporal_pattern'] = 'weekday_concentrated'

        # Check correlation with other columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for other_col in numeric_cols:
            if other_col != column:
                # Compare correlation in outliers vs overall
                overall_corr = df[[column, other_col]].corr().iloc[0, 1]
                outlier_corr = outlier_df[[column, other_col]].corr().iloc[0, 1] if len(outlier_df) > 1 else 0

                if abs(outlier_corr) > 0.7:
                    patterns['correlation_with_others'][other_col] = {
                        'correlation': outlier_corr,
                        'strength': 'strong'
                    }

        return patterns

    def explain_outlier(
        self,
        df: pd.DataFrame,
        column: str,
        value: Any,
        row_index: int,
        stats_dict: Dict[str, Any],
        patterns: Dict[str, Any]
    ) -> OutlierExplanation:
        """
        Generate AI-powered explanation for a single outlier

        Args:
            df: DataFrame
            column: Column name
            value: Outlier value
            row_index: Row index of outlier
            stats_dict: Statistical context
            patterns: Pattern analysis

        Returns:
            OutlierExplanation object
        """
        # Calculate Z-score and percentile
        col_data = df[column].dropna()
        z_score = (value - stats_dict['mean']) / stats_dict['std'] if stats_dict['std'] > 0 else 0
        percentile = stats.percentileofscore(col_data, value)

        # Get row context
        row_data = df.iloc[row_index].to_dict()

        # Determine legitimacy and generate explanation
        legitimacy, explanation, likely_causes = self._assess_legitimacy(
            df, column, value, row_data, stats_dict, patterns
        )

        # Generate recommendation
        recommendation, rationale, confidence = self._generate_recommendation(
            legitimacy, z_score, patterns, explanation
        )

        return OutlierExplanation(
            column=column,
            value=value,
            row_index=row_index,
            z_score=z_score,
            percentile=percentile,
            legitimacy=legitimacy,
            explanation=explanation,
            likely_causes=likely_causes,
            recommendation=recommendation,
            rationale=rationale,
            confidence=confidence,
            context={
                'stats': stats_dict,
                'patterns': patterns,
                'row_data': row_data
            }
        )

    def _assess_legitimacy(
        self,
        df: pd.DataFrame,
        column: str,
        value: Any,
        row_data: Dict,
        stats_dict: Dict,
        patterns: Dict
    ) -> Tuple[str, str, List[str]]:
        """Assess if outlier is legitimate or erroneous"""

        # Generate AI explanation if available
        if self.is_available():
            ai_explanation = self._generate_ai_explanation(
                df, column, value, row_data, stats_dict, patterns
            )
        else:
            ai_explanation = None

        # Statistical assessment
        z_score = abs((value - stats_dict['mean']) / stats_dict['std']) if stats_dict['std'] > 0 else 0

        # Heuristic assessment
        likely_causes = []
        legitimacy = 'suspicious'

        # Check for extreme values
        if z_score > 5:
            likely_causes.append("Extreme statistical deviation (Z-score > 5)")
            legitimacy = 'suspicious'
        elif z_score > 3:
            likely_causes.append("Moderate statistical deviation")

        # Check for negative values in positive-only columns
        if any(keyword in column.lower() for keyword in ['age', 'price', 'count', 'quantity']) and value < 0:
            likely_causes.append("Negative value in typically positive field")
            legitimacy = 'error'

        # Check for impossible values
        if 'age' in column.lower() and (value < 0 or value > 120):
            likely_causes.append("Value outside realistic age range (0-120)")
            legitimacy = 'error'

        # Check clustering pattern
        if patterns.get('clustering') == 'tight':
            likely_causes.append("Part of clustered outlier group (may indicate segment)")
            legitimacy = 'legitimate'

        # Check correlations
        if patterns.get('correlation_with_others'):
            likely_causes.append("Correlated with other features (suggests legitimacy)")
            legitimacy = 'legitimate'

        # Check temporal patterns
        if patterns.get('temporal_pattern') == 'weekend_spike':
            likely_causes.append("Occurs predominantly on weekends")
            legitimacy = 'legitimate'

        # Default to suspicious if no clear indication
        if not likely_causes:
            likely_causes.append("Statistical outlier - requires investigation")
            legitimacy = 'suspicious'

        # Use AI explanation if available
        if ai_explanation:
            explanation = ai_explanation
        else:
            explanation = f"Value {value} in '{column}' is {z_score:.1f} standard deviations from mean ({stats_dict['mean']:.2f}). "
            explanation += " ".join(likely_causes)

        return legitimacy, explanation, likely_causes

    def _generate_ai_explanation(
        self,
        df: pd.DataFrame,
        column: str,
        value: Any,
        row_data: Dict,
        stats_dict: Dict,
        patterns: Dict
    ) -> Optional[str]:
        """Generate AI explanation using LLM"""

        # Build context for LLM
        context = f"""
Outlier Analysis:

Column: {column}
Value: {value}
Z-score: {abs((value - stats_dict['mean']) / stats_dict['std']):.2f}
Percentile: {stats.percentileofscore(df[column].dropna(), value):.1f}%

Statistics:
- Mean: {stats_dict['mean']:.2f}
- Median: {stats_dict['median']:.2f}
- Std Dev: {stats_dict['std']:.2f}
- Range: [{stats_dict['min']:.2f}, {stats_dict['max']:.2f}]

Row Context:
{str({k: v for k, v in list(row_data.items())[:5]})}

Patterns:
- Total outliers: {patterns['total_outliers']}
- Clustering: {patterns.get('clustering', 'unknown')}
- Temporal: {patterns.get('temporal_pattern', 'none detected')}

Question: Is this outlier likely legitimate, an error, or suspicious? Explain in 2-3 sentences why it exists and whether it should be kept.
"""

        try:
            response = self.llm.generate(
                prompt=context,
                system_prompt="You are a data quality expert analyzing outliers. Be concise and specific.",
                max_tokens=200
            )

            if response.success:
                return response.content
            else:
                return None

        except Exception:
            return None

    def _generate_recommendation(
        self,
        legitimacy: str,
        z_score: float,
        patterns: Dict,
        explanation: str
    ) -> Tuple[str, str, float]:
        """Generate action recommendation"""

        if legitimacy == 'error':
            recommendation = 'remove'
            rationale = "Clear data quality issue detected. Removing is recommended."
            confidence = 0.9
        elif legitimacy == 'legitimate':
            recommendation = 'keep'
            rationale = "Appears to be legitimate data representing valid edge case or segment."
            confidence = 0.85
        else:  # suspicious
            recommendation = 'investigate'
            rationale = "Requires manual review to determine if legitimate or erroneous."
            confidence = 0.7

        # Adjust based on Z-score
        if abs(z_score) > 5 and recommendation == 'keep':
            recommendation = 'investigate'
            rationale += " However, extreme Z-score warrants verification."
            confidence = 0.6

        # Adjust based on patterns
        if patterns.get('clustering') == 'tight' and recommendation != 'keep':
            rationale += " Note: Part of clustered group, may represent valid segment."

        return recommendation, rationale, confidence

    def batch_explain_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        max_outliers: int = 10,
        method: str = 'iqr'
    ) -> List[OutlierExplanation]:
        """
        Explain multiple outliers in a column

        Args:
            df: DataFrame
            column: Column to analyze
            max_outliers: Maximum number of outliers to explain
            method: Detection method ('iqr', 'zscore', 'both')

        Returns:
            List of OutlierExplanation objects
        """
        # Detect outliers
        outlier_mask, stats_dict = self.detect_outliers(df, column, method=method)

        # Analyze patterns
        patterns = self.analyze_outlier_patterns(df, column, outlier_mask)

        # Get outlier indices
        outlier_indices = df[outlier_mask].index.tolist()

        # Limit to max_outliers
        if len(outlier_indices) > max_outliers:
            # Sample from extreme outliers
            outlier_values = df.loc[outlier_indices, column]
            z_scores = np.abs((outlier_values - stats_dict['mean']) / stats_dict['std'])
            top_indices = z_scores.nlargest(max_outliers).index.tolist()
        else:
            top_indices = outlier_indices

        # Generate explanations
        explanations = []
        for idx in top_indices:
            value = df.loc[idx, column]
            explanation = self.explain_outlier(
                df, column, value, idx, stats_dict, patterns
            )
            explanations.append(explanation)

        return explanations


def get_outlier_history() -> List[Dict[str, Any]]:
    """Get outlier action history from session state"""
    return st.session_state.get('outlier_history', [])


def add_to_outlier_history(action: Dict[str, Any]):
    """Add action to outlier history"""
    if 'outlier_history' not in st.session_state:
        st.session_state.outlier_history = []

    st.session_state.outlier_history.append(action)

    # Limit history to last 50 actions
    if len(st.session_state.outlier_history) > 50:
        st.session_state.outlier_history = st.session_state.outlier_history[-50:]


def clear_outlier_history():
    """Clear outlier history"""
    if 'outlier_history' in st.session_state:
        st.session_state.outlier_history = []


def display_anomaly_explanation():
    """Display AI-powered anomaly explanation interface"""

    st.markdown('<div class="section-header">🔍 Anomaly Explanation</div>', unsafe_allow_html=True)

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
        st.info("📁 Please upload a dataset first to analyze anomalies.")
        return

    # Get default model
    default_model = manager.get_default_model()

    # Display feature info
    st.markdown("""
    **AI-Powered Anomaly Explanation** helps you understand outliers:
    - 🤖 AI explains why outliers exist
    - 🔍 Root cause analysis
    - 📊 Pattern detection
    - ✅ Keep/Remove/Investigate recommendations
    - 💡 Legitimacy assessment
    """)

    # Column selection
    numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.warning("⚠️ No numeric columns found in dataset.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_column = st.selectbox(
            "Select Column to Analyze:",
            options=numeric_cols,
            help="Choose a numeric column to analyze for outliers"
        )

    with col2:
        detection_method = st.selectbox(
            "Detection Method:",
            options=['IQR', 'Z-Score', 'Both'],
            help="Method for outlier detection"
        )

    # Main controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🔍 Analyze Outliers", type="primary"):
            with st.spinner("🤖 Detecting and analyzing outliers..."):
                explainer = AnomalyExplainer(model_name=default_model)

                # Detect and explain outliers
                method_map = {'IQR': 'iqr', 'Z-Score': 'zscore', 'Both': 'both'}
                selected_method = method_map[detection_method]

                explanations = explainer.batch_explain_outliers(
                    df=st.session_state.df,
                    column=selected_column,
                    max_outliers=10,
                    method=selected_method
                )

                # Store in session state
                st.session_state.outlier_explanations = explanations
                st.session_state.outlier_explainer = explainer
                st.session_state.outlier_column = selected_column
                st.session_state.outlier_method = detection_method

            # Show success message
            if explanations:
                st.success(f"✅ Found {len(explanations)} outliers using {detection_method} method")
            else:
                st.info("✅ No outliers detected")
            st.rerun()

    with col2:
        if st.button("📜 View History"):
            st.session_state.show_outlier_history = not st.session_state.get('show_outlier_history', False)

    with col3:
        if st.button("🔄 Clear History"):
            clear_outlier_history()
            st.success("History cleared!")
            st.rerun()

    st.markdown("---")

    # Display history if requested
    if st.session_state.get('show_outlier_history', False):
        history = get_outlier_history()
        if history:
            st.markdown("### 📜 Outlier Action History")
            for i, action in enumerate(reversed(history)):
                with st.expander(f"Action {len(history) - i}: {action.get('column', 'Unknown')} = {action.get('value', 'N/A')}"):
                    st.markdown(f"**Action:** {action.get('action', 'Unknown')}")
                    st.markdown(f"**Column:** {action.get('column', 'Unknown')}")
                    st.markdown(f"**Value:** {action.get('value', 'N/A')}")
                    st.markdown(f"**Timestamp:** {action.get('timestamp', 'Unknown')}")
                    if action.get('message'):
                        st.info(action['message'])
            st.markdown("---")
        else:
            st.info("No outlier actions yet.")
            st.markdown("---")

    # Display explanations
    if not st.session_state.get('outlier_explanations'):
        st.info("💡 Select a column and click 'Analyze Outliers' to get AI-powered explanations.")
        return

    explanations = st.session_state.outlier_explanations

    if not explanations:
        st.success("✅ No outliers detected in this column!")
        return

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Outliers", len(explanations))
    with col2:
        legitimate_count = sum(1 for e in explanations if e.legitimacy == 'legitimate')
        st.metric("Legitimate", legitimate_count)
    with col3:
        error_count = sum(1 for e in explanations if e.legitimacy == 'error')
        st.metric("Errors", error_count)
    with col4:
        suspicious_count = sum(1 for e in explanations if e.legitimacy == 'suspicious')
        st.metric("Suspicious", suspicious_count)

    st.markdown("---")

    # Display each outlier explanation
    for i, explanation in enumerate(explanations):
        # Legitimacy badge
        legitimacy_badges = {
            'legitimate': '✅ LIKELY LEGITIMATE',
            'error': '❌ LIKELY ERROR',
            'suspicious': '⚠️ SUSPICIOUS'
        }
        legitimacy_colors = {
            'legitimate': 'green',
            'error': 'red',
            'suspicious': 'orange'
        }

        badge = legitimacy_badges.get(explanation.legitimacy, '❓ UNKNOWN')
        color = legitimacy_colors.get(explanation.legitimacy, 'gray')

        # Outlier header
        st.markdown(f"### Outlier #{i+1}: {explanation.column} = {explanation.value}")

        # Status badge
        st.markdown(f"**Status:** :{color}[{badge}]")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Z-Score", f"{explanation.z_score:.2f}")
        with col2:
            st.metric("Percentile", f"{explanation.percentile:.1f}%")
        with col3:
            confidence_pct = explanation.confidence * 100
            st.metric("Confidence", f"{confidence_pct:.0f}%")

        # Explanation details
        with st.expander("📋 View Detailed Analysis", expanded=True):
            # AI explanation
            st.markdown("**AI Analysis:**")
            st.info(explanation.explanation)

            # Likely causes
            if explanation.likely_causes:
                st.markdown("**Likely Causes:**")
                for cause in explanation.likely_causes:
                    st.markdown(f"- {cause}")

            # Recommendation
            st.markdown("**Recommendation:**")
            rec_icons = {
                'keep': '✅',
                'remove': '❌',
                'investigate': '🔍'
            }
            rec_icon = rec_icons.get(explanation.recommendation, '❓')
            st.markdown(f"{rec_icon} **{explanation.recommendation.upper()}**")
            st.markdown(explanation.rationale)

            # Context
            with st.expander("📊 Statistical Context"):
                stats = explanation.context['stats']
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Mean: {stats['mean']:.2f}")
                    st.caption(f"Median: {stats['median']:.2f}")
                    st.caption(f"Std Dev: {stats['std']:.2f}")
                with col2:
                    st.caption(f"Min: {stats['min']:.2f}")
                    st.caption(f"Max: {stats['max']:.2f}")
                    st.caption(f"IQR: {stats['q3'] - stats['q1']:.2f}")

            # Patterns
            patterns = explanation.context['patterns']
            if patterns.get('clustering') or patterns.get('temporal_pattern') or patterns.get('correlation_with_others'):
                with st.expander("🔎 Pattern Analysis"):
                    if patterns.get('clustering'):
                        st.caption(f"**Clustering:** {patterns['clustering'].title()}")
                    if patterns.get('temporal_pattern'):
                        st.caption(f"**Temporal:** {patterns['temporal_pattern'].replace('_', ' ').title()}")
                    if patterns.get('correlation_with_others'):
                        st.caption("**Correlations:**")
                        for col, info in patterns['correlation_with_others'].items():
                            st.caption(f"  - {col}: {info['correlation']:.2f} ({info['strength']})")

        # Action buttons
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        with col1:
            if st.button("✅ Keep", key=f"keep_{explanation.row_index}"):
                add_to_outlier_history({
                    'action': 'keep',
                    'column': explanation.column,
                    'value': explanation.value,
                    'row_index': explanation.row_index,
                    'message': f"Marked as legitimate - kept in dataset",
                    'timestamp': datetime.now().isoformat()
                })
                st.success(f"Outlier marked as legitimate and kept.")

        with col2:
            if st.button("❌ Remove", key=f"remove_{explanation.row_index}"):
                # Remove the row from dataframe
                st.session_state.df = st.session_state.df.drop(explanation.row_index)

                add_to_outlier_history({
                    'action': 'remove',
                    'column': explanation.column,
                    'value': explanation.value,
                    'row_index': explanation.row_index,
                    'message': f"Removed outlier from dataset",
                    'timestamp': datetime.now().isoformat()
                })

                st.success(f"Outlier removed! Data shape: {st.session_state.df.shape}")

                # Clear explanations to force re-analysis
                if 'outlier_explanations' in st.session_state:
                    del st.session_state.outlier_explanations

                st.rerun()

        with col3:
            if st.button("🔍 Investigate", key=f"investigate_{explanation.row_index}"):
                add_to_outlier_history({
                    'action': 'investigate',
                    'column': explanation.column,
                    'value': explanation.value,
                    'row_index': explanation.row_index,
                    'message': f"Flagged for investigation",
                    'timestamp': datetime.now().isoformat()
                })
                st.info(f"Outlier flagged for manual investigation.")

        # Show full row data
        with st.expander("📄 View Full Row Data"):
            row_data = st.session_state.df.iloc[explanation.row_index]
            st.dataframe(row_data.to_frame().T, use_container_width=True)

        st.markdown("---")
