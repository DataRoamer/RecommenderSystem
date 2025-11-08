"""
AI-Powered Data Cleaning Advisor
Provides intelligent recommendations for data quality issues with one-click fixes
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import re
from .llm_integration import LocalLLM, AIResponse
from .prompts import DATA_QUALITY_PROMPT


@dataclass
class CleaningIssue:
    """Represents a data quality issue"""
    id: str
    type: str  # 'invalid_values', 'duplicates', 'outliers', 'missing', 'format'
    severity: str  # 'critical', 'high', 'medium', 'low'
    column: Optional[str]
    affected_rows: int
    description: str
    examples: List[Any]
    details: Dict[str, Any]


@dataclass
class CleaningRecommendation:
    """Represents a cleaning recommendation"""
    issue_id: str
    fix_type: str
    description: str
    rationale: str  # AI-generated explanation
    confidence: float  # 0.0-1.0
    impact: str
    preview: Dict[str, Any]
    alternatives: List[Dict[str, Any]]
    code: str  # Generated code for the fix


class DataCleaningAdvisor:
    """AI-powered data cleaning advisor"""

    def __init__(self, model_name: str = 'phi3:mini', temperature: float = 0.7):
        """
        Initialize Data Cleaning Advisor

        Args:
            model_name: Name of the LLM model to use
            temperature: Sampling temperature for responses
        """
        self.llm = LocalLLM(model_name, temperature)
        self.issues: List[CleaningIssue] = []
        self.recommendations: Dict[str, CleaningRecommendation] = {}

    def is_available(self) -> bool:
        """Check if AI is available"""
        return self.llm.is_available()

    def analyze_quality_issues(
        self,
        df: pd.DataFrame,
        quality_report: Dict[str, Any]
    ) -> List[CleaningIssue]:
        """
        Analyze data quality report and detect issues

        Args:
            df: DataFrame to analyze
            quality_report: Quality report from comprehensive_data_quality_report()

        Returns:
            List of detected issues
        """
        self.issues = []

        # Detect invalid values
        self.issues.extend(self._detect_invalid_values(df))

        # Detect duplicates
        self.issues.extend(self._detect_duplicates(df, quality_report))

        # Detect outliers
        self.issues.extend(self._detect_outliers(df, quality_report))

        # Detect missing value patterns
        self.issues.extend(self._detect_missing_patterns(df, quality_report))

        # Detect format inconsistencies
        self.issues.extend(self._detect_format_issues(df))

        return self.issues

    def _detect_invalid_values(self, df: pd.DataFrame) -> List[CleaningIssue]:
        """Detect invalid values (negative ages, out-of-range, etc.)"""
        issues = []

        # Check numeric columns for invalid values
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue

            # Check for negative values in likely positive columns
            if any(keyword in col.lower() for keyword in ['age', 'price', 'count', 'quantity', 'amount']):
                negative_mask = col_data < 0
                if negative_mask.any():
                    negative_count = negative_mask.sum()
                    negative_values = col_data[negative_mask].head(5).tolist()

                    issue = CleaningIssue(
                        id=f"invalid_{col}_{datetime.now().timestamp()}",
                        type='invalid_values',
                        severity='critical',
                        column=col,
                        affected_rows=negative_count,
                        description=f"Column '{col}' contains {negative_count} negative value(s)",
                        examples=negative_values,
                        details={
                            'issue_type': 'negative_values',
                            'expected_range': 'positive values only'
                        }
                    )
                    issues.append(issue)

            # Check for unrealistic ages
            if 'age' in col.lower():
                unrealistic_mask = (col_data < 0) | (col_data > 120)
                if unrealistic_mask.any():
                    unrealistic_count = unrealistic_mask.sum()
                    unrealistic_values = col_data[unrealistic_mask].head(5).tolist()

                    issue = CleaningIssue(
                        id=f"invalid_{col}_range_{datetime.now().timestamp()}",
                        type='invalid_values',
                        severity='high',
                        column=col,
                        affected_rows=unrealistic_count,
                        description=f"Column '{col}' contains {unrealistic_count} unrealistic value(s)",
                        examples=unrealistic_values,
                        details={
                            'issue_type': 'out_of_range',
                            'expected_range': '0-120 years'
                        }
                    )
                    issues.append(issue)

        return issues

    def _detect_duplicates(
        self,
        df: pd.DataFrame,
        quality_report: Dict[str, Any]
    ) -> List[CleaningIssue]:
        """Detect duplicate records"""
        issues = []

        duplicates_info = quality_report.get('duplicates', {})
        duplicate_count = duplicates_info.get('duplicate_rows', 0)

        if duplicate_count > 0:
            severity = 'high' if duplicate_count > len(df) * 0.1 else 'medium'

            issue = CleaningIssue(
                id=f"duplicates_{datetime.now().timestamp()}",
                type='duplicates',
                severity=severity,
                column=None,  # Affects all columns
                affected_rows=duplicate_count,
                description=f"Found {duplicate_count} duplicate record(s) ({duplicates_info.get('duplicate_percentage', 0):.1f}%)",
                examples=[],
                details={
                    'duplicate_percentage': duplicates_info.get('duplicate_percentage', 0),
                    'unique_rows': duplicates_info.get('unique_rows', 0)
                }
            )
            issues.append(issue)

        return issues

    def _detect_outliers(
        self,
        df: pd.DataFrame,
        quality_report: Dict[str, Any]
    ) -> List[CleaningIssue]:
        """Detect outliers in numeric columns"""
        issues = []

        outliers_info = quality_report.get('outliers', {})
        outliers_by_column = outliers_info.get('outliers_by_column', {})

        for col, col_outliers in outliers_by_column.items():
            outlier_count = col_outliers.get('outlier_count', 0)

            if outlier_count > 0:
                outlier_percentage = (outlier_count / len(df)) * 100
                severity = 'high' if outlier_percentage > 5 else 'medium'

                outlier_values = col_outliers.get('outlier_values', [])[:5]

                issue = CleaningIssue(
                    id=f"outliers_{col}_{datetime.now().timestamp()}",
                    type='outliers',
                    severity=severity,
                    column=col,
                    affected_rows=outlier_count,
                    description=f"Column '{col}' contains {outlier_count} outlier(s) ({outlier_percentage:.1f}%)",
                    examples=outlier_values,
                    details={
                        'method': col_outliers.get('method', 'IQR'),
                        'outlier_percentage': outlier_percentage
                    }
                )
                issues.append(issue)

        return issues

    def _detect_missing_patterns(
        self,
        df: pd.DataFrame,
        quality_report: Dict[str, Any]
    ) -> List[CleaningIssue]:
        """Detect missing value patterns"""
        issues = []

        missing_info = quality_report.get('missing_values', {})
        columns_with_missing = missing_info.get('columns_with_missing', {})

        for col, col_missing in columns_with_missing.items():
            missing_count = col_missing.get('count', 0)
            missing_pct = col_missing.get('percentage', 0)

            if missing_pct > 50:
                severity = 'critical'
            elif missing_pct > 20:
                severity = 'high'
            elif missing_pct > 5:
                severity = 'medium'
            else:
                severity = 'low'

            issue = CleaningIssue(
                id=f"missing_{col}_{datetime.now().timestamp()}",
                type='missing',
                severity=severity,
                column=col,
                affected_rows=missing_count,
                description=f"Column '{col}' has {missing_count} missing value(s) ({missing_pct:.1f}%)",
                examples=[],
                details={
                    'missing_percentage': missing_pct,
                    'data_type': col_missing.get('data_type', 'unknown')
                }
            )
            issues.append(issue)

        return issues

    def _detect_format_issues(self, df: pd.DataFrame) -> List[CleaningIssue]:
        """Detect format inconsistencies (emails, dates, etc.)"""
        issues = []

        # Check for email format issues
        for col in df.select_dtypes(include=['object']).columns:
            if 'email' in col.lower():
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                invalid_emails = df[col].dropna().apply(
                    lambda x: not re.match(email_pattern, str(x))
                )

                if invalid_emails.any():
                    invalid_count = invalid_emails.sum()
                    invalid_examples = df[col][invalid_emails].head(5).tolist()

                    issue = CleaningIssue(
                        id=f"format_{col}_{datetime.now().timestamp()}",
                        type='format',
                        severity='medium',
                        column=col,
                        affected_rows=invalid_count,
                        description=f"Column '{col}' has {invalid_count} invalid email format(s)",
                        examples=invalid_examples,
                        details={
                            'format_type': 'email',
                            'expected_pattern': 'user@domain.com'
                        }
                    )
                    issues.append(issue)

        return issues

    def generate_recommendations(
        self,
        df: pd.DataFrame,
        issues: List[CleaningIssue]
    ) -> Dict[str, CleaningRecommendation]:
        """
        Generate AI-powered recommendations for detected issues

        Args:
            df: DataFrame
            issues: List of detected issues

        Returns:
            Dictionary of recommendations keyed by issue ID
        """
        recommendations = {}

        for issue in issues:
            rec = self._generate_recommendation_for_issue(df, issue)
            if rec:
                recommendations[issue.id] = rec

        self.recommendations = recommendations
        return recommendations

    def _generate_recommendation_for_issue(
        self,
        df: pd.DataFrame,
        issue: CleaningIssue
    ) -> Optional[CleaningRecommendation]:
        """Generate recommendation for a single issue"""

        # Base recommendation based on issue type
        if issue.type == 'invalid_values':
            return self._recommend_invalid_value_fix(df, issue)
        elif issue.type == 'duplicates':
            return self._recommend_duplicate_fix(df, issue)
        elif issue.type == 'outliers':
            return self._recommend_outlier_fix(df, issue)
        elif issue.type == 'missing':
            return self._recommend_missing_fix(df, issue)
        elif issue.type == 'format':
            return self._recommend_format_fix(df, issue)

        return None

    def _recommend_invalid_value_fix(
        self,
        df: pd.DataFrame,
        issue: CleaningIssue
    ) -> CleaningRecommendation:
        """Recommend fix for invalid values"""

        col = issue.column
        col_data = df[col].dropna()
        median_value = col_data[col_data >= 0].median() if len(col_data) > 0 else 0

        # Generate AI rationale
        rationale = self._generate_ai_rationale(df, issue, f"replace with median ({median_value:.1f})")

        code = f"""# Replace negative values with median
df.loc[df['{col}'] < 0, '{col}'] = {median_value:.2f}
"""

        preview = {
            'before': issue.examples,
            'after': [median_value] * len(issue.examples),
            'affected_rows': issue.affected_rows
        }

        return CleaningRecommendation(
            issue_id=issue.id,
            fix_type='replace_invalid',
            description=f"Replace negative values in '{col}' with median ({median_value:.1f})",
            rationale=rationale,
            confidence=0.9,
            impact=f"Will replace {issue.affected_rows} invalid value(s)",
            preview=preview,
            alternatives=[
                {
                    'fix_type': 'remove_rows',
                    'description': f"Remove rows with invalid values",
                    'confidence': 0.7
                },
                {
                    'fix_type': 'flag_for_review',
                    'description': f"Flag for manual review",
                    'confidence': 1.0
                }
            ],
            code=code
        )

    def _recommend_duplicate_fix(
        self,
        df: pd.DataFrame,
        issue: CleaningIssue
    ) -> CleaningRecommendation:
        """Recommend fix for duplicates"""

        rationale = self._generate_ai_rationale(df, issue, "remove exact duplicates (keep first)")

        code = f"""# Remove duplicate rows (keep first occurrence)
df = df.drop_duplicates(keep='first')
"""

        return CleaningRecommendation(
            issue_id=issue.id,
            fix_type='remove_duplicates',
            description=f"Remove {issue.affected_rows} duplicate record(s)",
            rationale=rationale,
            confidence=0.95,
            impact=f"Will remove {issue.affected_rows} duplicate row(s), keeping first occurrence",
            preview={
                'before_count': len(df),
                'after_count': len(df) - issue.affected_rows,
                'removed_count': issue.affected_rows
            },
            alternatives=[
                {
                    'fix_type': 'keep_last',
                    'description': 'Keep last occurrence instead of first',
                    'confidence': 0.95
                }
            ],
            code=code
        )

    def _recommend_outlier_fix(
        self,
        df: pd.DataFrame,
        issue: CleaningIssue
    ) -> CleaningRecommendation:
        """Recommend fix for outliers"""

        col = issue.column
        rationale = self._generate_ai_rationale(df, issue, "flag for manual review")

        code = f"""# Flag outliers for review
df['{col}_outlier_flag'] = False
# Mark outliers based on IQR method
Q1 = df['{col}'].quantile(0.25)
Q3 = df['{col}'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df.loc[(df['{col}'] < lower_bound) | (df['{col}'] > upper_bound), '{col}_outlier_flag'] = True
"""

        return CleaningRecommendation(
            issue_id=issue.id,
            fix_type='flag_outliers',
            description=f"Flag {issue.affected_rows} outlier(s) in '{col}' for review",
            rationale=rationale,
            confidence=0.8,
            impact=f"Will add '{col}_outlier_flag' column to mark outliers",
            preview={
                'outlier_count': issue.affected_rows,
                'examples': issue.examples
            },
            alternatives=[
                {
                    'fix_type': 'cap_outliers',
                    'description': 'Cap outliers at 95th percentile',
                    'confidence': 0.6
                },
                {
                    'fix_type': 'remove_outliers',
                    'description': 'Remove outlier rows',
                    'confidence': 0.5
                }
            ],
            code=code
        )

    def _recommend_missing_fix(
        self,
        df: pd.DataFrame,
        issue: CleaningIssue
    ) -> CleaningRecommendation:
        """Recommend fix for missing values"""

        col = issue.column
        dtype = df[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            strategy = f"median ({df[col].median():.2f})"
            fill_value = df[col].median()
            code = f"""# Fill missing values with median
df['{col}'].fillna({fill_value:.2f}, inplace=True)
"""
        else:
            strategy = "mode (most frequent value)"
            fill_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
            code = f"""# Fill missing values with mode
df['{col}'].fillna('{fill_value}', inplace=True)
"""

        rationale = self._generate_ai_rationale(df, issue, f"impute with {strategy}")

        return CleaningRecommendation(
            issue_id=issue.id,
            fix_type='impute_missing',
            description=f"Fill missing values in '{col}' with {strategy}",
            rationale=rationale,
            confidence=0.75,
            impact=f"Will fill {issue.affected_rows} missing value(s)",
            preview={
                'fill_value': fill_value,
                'missing_count': issue.affected_rows
            },
            alternatives=[
                {
                    'fix_type': 'forward_fill',
                    'description': 'Forward fill (use previous value)',
                    'confidence': 0.6
                },
                {
                    'fix_type': 'drop_rows',
                    'description': 'Remove rows with missing values',
                    'confidence': 0.5
                }
            ],
            code=code
        )

    def _recommend_format_fix(
        self,
        df: pd.DataFrame,
        issue: CleaningIssue
    ) -> CleaningRecommendation:
        """Recommend fix for format issues"""

        col = issue.column
        rationale = self._generate_ai_rationale(df, issue, "flag for manual review")

        code = f"""# Flag invalid format for review
df['{col}_invalid_format'] = df['{col}'].apply(
    lambda x: not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{{2,}}$', str(x))
)
"""

        return CleaningRecommendation(
            issue_id=issue.id,
            fix_type='flag_format',
            description=f"Flag {issue.affected_rows} invalid format(s) in '{col}'",
            rationale=rationale,
            confidence=0.85,
            impact=f"Will add '{col}_invalid_format' flag column",
            preview={
                'invalid_count': issue.affected_rows,
                'examples': issue.examples
            },
            alternatives=[
                {
                    'fix_type': 'remove_invalid',
                    'description': 'Remove rows with invalid format',
                    'confidence': 0.6
                }
            ],
            code=code
        )

    def _generate_ai_rationale(
        self,
        df: pd.DataFrame,
        issue: CleaningIssue,
        recommendation: str
    ) -> str:
        """Generate AI explanation for recommendation"""

        if not self.is_available():
            return f"Recommendation: {recommendation}. This is a standard approach for this type of issue."

        # Create context for LLM
        context = f"""
Data Quality Issue:
- Type: {issue.type}
- Column: {issue.column}
- Severity: {issue.severity}
- Affected Rows: {issue.affected_rows}
- Description: {issue.description}
- Examples: {issue.examples[:3]}

Dataset Context:
- Total Rows: {len(df)}
- Columns: {len(df.columns)}

Recommended Fix: {recommendation}

Explain why this recommendation is appropriate for this issue. Be concise (2-3 sentences).
"""

        try:
            response = self.llm.generate(
                prompt=context,
                system_prompt="You are a data quality expert. Explain cleaning recommendations clearly and concisely.",
                max_tokens=150
            )

            if response.success:
                return response.content
            else:
                return f"Recommendation: {recommendation}. Standard approach for {issue.type} issues."

        except Exception:
            return f"Recommendation: {recommendation}. Standard approach for {issue.type} issues."

    def apply_fix(
        self,
        df: pd.DataFrame,
        recommendation: CleaningRecommendation
    ) -> Tuple[bool, pd.DataFrame, str]:
        """
        Apply a cleaning fix to the DataFrame

        Args:
            df: DataFrame to clean
            recommendation: Recommendation to apply

        Returns:
            Tuple of (success, modified_df, message)
        """
        try:
            df_copy = df.copy()

            # Execute the recommendation code
            local_vars = {'df': df_copy, 're': re, 'pd': pd, 'np': np}
            exec(recommendation.code, {}, local_vars)
            df_result = local_vars['df']

            return True, df_result, f"Successfully applied fix: {recommendation.description}"

        except Exception as e:
            return False, df, f"Failed to apply fix: {str(e)}"


def get_cleaning_history() -> List[Dict[str, Any]]:
    """Get cleaning history from session state"""
    return st.session_state.get('cleaning_history', [])


def add_to_cleaning_history(action: Dict[str, Any]):
    """Add action to cleaning history"""
    if 'cleaning_history' not in st.session_state:
        st.session_state.cleaning_history = []

    st.session_state.cleaning_history.append(action)

    # Limit history to last 20 actions
    if len(st.session_state.cleaning_history) > 20:
        st.session_state.cleaning_history = st.session_state.cleaning_history[-20:]


def clear_cleaning_history():
    """Clear cleaning history"""
    if 'cleaning_history' in st.session_state:
        st.session_state.cleaning_history = []


def display_data_cleaning():
    """Display AI-powered data cleaning interface"""

    st.markdown('<div class="section-header">🧹 Smart Data Cleaning</div>', unsafe_allow_html=True)

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
        st.info("📁 Please upload a dataset first to use data cleaning features.")
        return

    # Check if quality report is available
    if not st.session_state.get('quality_report'):
        st.warning("⚠️ Please run Data Quality analysis first.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Go to Data Quality"):
                st.session_state.current_section = "quality"
                st.rerun()
        return

    # Get default model
    default_model = manager.get_default_model()

    # Display feature info
    st.markdown("""
    **AI-Powered Data Cleaning** analyzes your data quality issues and provides:
    - 🔍 Automatic issue detection
    - 💡 Intelligent recommendations
    - 📊 Preview before applying
    - ↩️ Undo capability
    - 🤖 AI-generated explanations
    """)

    # Main controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🔍 Analyze Data Quality Issues", type="primary"):
            with st.spinner("🤖 Analyzing data quality..."):
                advisor = DataCleaningAdvisor(model_name=default_model)

                # Analyze issues
                issues = advisor.analyze_quality_issues(
                    df=st.session_state.df,
                    quality_report=st.session_state.quality_report
                )

                # Generate recommendations
                recommendations = advisor.generate_recommendations(
                    df=st.session_state.df,
                    issues=issues
                )

                # Store in session state
                st.session_state.cleaning_issues = issues
                st.session_state.cleaning_recommendations = recommendations
                st.session_state.cleaning_advisor = advisor

    with col2:
        if st.button("📜 View History"):
            st.session_state.show_cleaning_history = not st.session_state.get('show_cleaning_history', False)

    with col3:
        if st.button("🔄 Clear History"):
            clear_cleaning_history()
            st.success("History cleared!")
            st.rerun()

    st.markdown("---")

    # Display cleaning history if requested
    if st.session_state.get('show_cleaning_history', False):
        history = get_cleaning_history()
        if history:
            st.markdown("### 📜 Cleaning History")
            for i, action in enumerate(reversed(history)):
                with st.expander(f"Action {len(history) - i}: {action.get('description', 'Unknown')}"):
                    st.markdown(f"**Type:** {action.get('fix_type', 'Unknown')}")
                    st.markdown(f"**Timestamp:** {action.get('timestamp', 'Unknown')}")
                    st.markdown(f"**Status:** {'✅ Success' if action.get('success', False) else '❌ Failed'}")
                    if action.get('message'):
                        st.info(action['message'])
            st.markdown("---")
        else:
            st.info("No cleaning actions yet.")
            st.markdown("---")

    # Display issues and recommendations
    if not st.session_state.get('cleaning_issues'):
        st.info("💡 Click 'Analyze Data Quality Issues' to get AI-powered cleaning recommendations.")
        return

    issues = st.session_state.cleaning_issues
    recommendations = st.session_state.cleaning_recommendations

    if not issues:
        st.success("🎉 No data quality issues detected! Your data looks clean.")
        return

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Issues", len(issues))
    with col2:
        critical_count = sum(1 for i in issues if i.severity == 'critical')
        st.metric("Critical", critical_count)
    with col3:
        high_count = sum(1 for i in issues if i.severity == 'high')
        st.metric("High", high_count)
    with col4:
        total_affected = sum(i.affected_rows for i in issues)
        st.metric("Rows Affected", total_affected)

    st.markdown("---")

    # Display each issue with recommendations
    for issue in issues:
        # Severity badge color
        severity_colors = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }
        badge = severity_colors.get(issue.severity, '⚪')

        # Issue header
        st.markdown(f"### {badge} {issue.severity.upper()}: {issue.description}")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Column:** {issue.column if issue.column else 'All columns'}")
            st.markdown(f"**Affected Rows:** {issue.affected_rows}")
            st.markdown(f"**Type:** {issue.type.replace('_', ' ').title()}")
        with col2:
            if issue.examples:
                st.caption("**Examples:**")
                for ex in issue.examples[:3]:
                    st.caption(f"• {ex}")

        # Get recommendation
        recommendation = recommendations.get(issue.id)

        if recommendation:
            # Recommendation details
            with st.expander("📋 View Recommendation Details", expanded=True):
                st.markdown(f"**Recommendation:** {recommendation.description}")

                # AI rationale
                st.markdown("**AI Analysis:**")
                st.info(recommendation.rationale)

                # Confidence
                confidence_pct = recommendation.confidence * 100
                confidence_stars = "⭐" * int(recommendation.confidence * 5)
                st.markdown(f"**Confidence:** {confidence_stars} ({confidence_pct:.0f}%)")

                # Impact
                st.markdown(f"**Impact:** {recommendation.impact}")

                # Show code
                with st.expander("🐍 View Generated Code"):
                    st.code(recommendation.code, language='python')

                # Preview
                if recommendation.preview:
                    st.markdown("**Preview:**")
                    preview_data = recommendation.preview

                    if 'before' in preview_data and 'after' in preview_data:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption("Before:")
                            st.write(preview_data['before'])
                        with col2:
                            st.caption("After:")
                            st.write(preview_data['after'])
                    else:
                        st.json(preview_data)

                # Alternatives
                if recommendation.alternatives:
                    st.markdown("**Alternative Options:**")
                    for alt in recommendation.alternatives:
                        st.caption(f"• {alt['description']} (confidence: {alt['confidence']*100:.0f}%)")

            # Action buttons
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                if st.button("✅ Apply Fix", key=f"apply_{issue.id}"):
                    advisor = st.session_state.get('cleaning_advisor')
                    if advisor:
                        success, df_result, message = advisor.apply_fix(
                            df=st.session_state.df,
                            recommendation=recommendation
                        )

                        if success:
                            # Update the dataframe
                            st.session_state.df = df_result

                            # Add to history
                            add_to_cleaning_history({
                                'description': recommendation.description,
                                'fix_type': recommendation.fix_type,
                                'success': True,
                                'message': message,
                                'timestamp': datetime.now().isoformat()
                            })

                            st.success(message)
                            st.info("💾 Data updated! Remember to re-run analyses to see the impact.")

                            # Clear current issues to force re-analysis
                            if 'cleaning_issues' in st.session_state:
                                del st.session_state.cleaning_issues
                            if 'cleaning_recommendations' in st.session_state:
                                del st.session_state.cleaning_recommendations

                            st.rerun()
                        else:
                            add_to_cleaning_history({
                                'description': recommendation.description,
                                'fix_type': recommendation.fix_type,
                                'success': False,
                                'message': message,
                                'timestamp': datetime.now().isoformat()
                            })
                            st.error(message)

            with col2:
                if st.button("⏭️ Skip", key=f"skip_{issue.id}"):
                    st.info("Issue skipped. You can analyze again later.")

        st.markdown("---")
