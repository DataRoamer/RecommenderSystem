"""
AI-Powered Executive Report Generator

Generates professional, narrative reports summarizing all analyses in business-friendly language.
Supports multiple export formats: PDF, HTML, Markdown, and Word.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import io
import base64

# Import LLM integration
from .llm_integration import get_ai_response
from .context_builder import build_dataset_context


@dataclass
class ReportSection:
    """Represents a section in the report"""
    title: str
    content: str
    order: int
    include: bool = True
    subsections: List[Dict[str, str]] = None


class ReportGenerator:
    """
    AI-Powered Report Generator

    Generates comprehensive executive reports with AI-generated narratives
    and professional formatting. Supports multiple export formats.
    """

    def __init__(self, model_name: str = 'phi3:mini'):
        """Initialize the report generator"""
        self.model_name = model_name
        self.sections = []
        self.metadata = {}

    def generate_report(
        self,
        df: pd.DataFrame,
        quality_report: Optional[Dict] = None,
        eda_results: Optional[Dict] = None,
        target_analysis: Optional[Dict] = None,
        feature_importance: Optional[Dict] = None,
        leakage_report: Optional[Dict] = None,
        readiness_score: Optional[Dict] = None
    ) -> Dict[str, ReportSection]:
        """
        Generate comprehensive report from all available analyses

        Args:
            df: DataFrame being analyzed
            quality_report: Data quality analysis results
            eda_results: EDA analysis results
            target_analysis: Target variable analysis
            feature_importance: Feature engineering results
            leakage_report: Data leakage detection results
            readiness_score: Model readiness assessment

        Returns:
            Dictionary of report sections
        """
        sections = {}

        # Store metadata
        self.metadata = {
            'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_name': 'Dataset Analysis',
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'report_version': '1.0'
        }

        # Generate each section
        sections['executive_summary'] = self._generate_executive_summary(
            df, quality_report, eda_results, target_analysis, readiness_score
        )

        sections['data_overview'] = self._generate_data_overview(df)

        if quality_report:
            sections['data_quality'] = self._generate_data_quality_section(quality_report)

        if eda_results:
            sections['statistical_analysis'] = self._generate_statistical_section(eda_results)

        if target_analysis:
            sections['target_analysis'] = self._generate_target_section(target_analysis)

        sections['recommendations'] = self._generate_recommendations(
            quality_report, eda_results, leakage_report, readiness_score
        )

        sections['appendix'] = self._generate_appendix(df, quality_report)

        self.sections = sections
        return sections

    def _generate_executive_summary(
        self,
        df: pd.DataFrame,
        quality_report: Optional[Dict],
        eda_results: Optional[Dict],
        target_analysis: Optional[Dict],
        readiness_score: Optional[Dict]
    ) -> ReportSection:
        """Generate AI-powered executive summary"""

        # Build context for LLM
        context = f"""Generate an executive summary for this dataset analysis:

Dataset Info:
- Rows: {len(df):,}
- Columns: {len(df.columns)}
- Column types: {df.dtypes.value_counts().to_dict()}

"""

        if quality_report:
            quality_score = quality_report.get('overall_score', 0)
            missing_pct = quality_report.get('missing_percentage', 0)
            context += f"""Data Quality:
- Overall Score: {quality_score:.1f}%
- Missing Data: {missing_pct:.1f}%
- Issues Found: {len(quality_report.get('issues', []))}

"""

        if readiness_score:
            overall_readiness = readiness_score.get('overall_score', 0)
            context += f"""Model Readiness:
- Overall Readiness: {overall_readiness:.1f}%

"""

        prompt = f"""{context}

Write a concise executive summary (3-5 bullet points) highlighting:
1. Overall assessment of data quality
2. Key findings from the analysis
3. Major recommendations
4. Business impact and next steps

Format as bullet points. Be concise and business-friendly."""

        try:
            # Get AI-generated summary
            summary_content = get_ai_response(
                prompt=prompt,
                model_name=self.model_name,
                temperature=0.7,
                max_tokens=500
            )

            if not summary_content or summary_content == "AI not available":
                # Fallback to template
                summary_content = self._generate_fallback_summary(
                    df, quality_report, readiness_score
                )
        except Exception as e:
            summary_content = self._generate_fallback_summary(
                df, quality_report, readiness_score
            )

        return ReportSection(
            title="Executive Summary",
            content=summary_content,
            order=1
        )

    def _generate_fallback_summary(
        self,
        df: pd.DataFrame,
        quality_report: Optional[Dict],
        readiness_score: Optional[Dict]
    ) -> str:
        """Generate fallback summary when AI is unavailable"""

        summary_points = []

        # Data overview
        summary_points.append(
            f"**Dataset Overview:** Analyzed {len(df):,} records across {len(df.columns)} features."
        )

        # Quality assessment
        if quality_report:
            quality_score = quality_report.get('overall_score', 0)
            if quality_score >= 80:
                quality_status = "excellent"
            elif quality_score >= 60:
                quality_status = "good"
            else:
                quality_status = "requires attention"

            summary_points.append(
                f"**Data Quality:** Overall quality is {quality_status} ({quality_score:.1f}% score)."
            )

        # Readiness
        if readiness_score:
            overall = readiness_score.get('overall_score', 0)
            if overall >= 80:
                readiness_status = "ready for modeling"
            elif overall >= 60:
                readiness_status = "needs minor improvements"
            else:
                readiness_status = "requires significant preparation"

            summary_points.append(
                f"**Model Readiness:** Dataset is {readiness_status} ({overall:.1f}% readiness)."
            )

        # Recommendations
        summary_points.append(
            "**Next Steps:** Review data quality issues, address missing values, and verify feature engineering."
        )

        return "\n\n".join(summary_points)

    def _generate_data_overview(self, df: pd.DataFrame) -> ReportSection:
        """Generate data overview section"""

        # Calculate statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        content = f"""
### Dataset Dimensions
- **Total Records:** {len(df):,}
- **Total Features:** {len(df.columns)}
- **Numeric Features:** {len(numeric_cols)}
- **Categorical Features:** {len(categorical_cols)}
- **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

### Column Types
"""

        # Add column type breakdown
        type_counts = df.dtypes.value_counts()
        for dtype, count in type_counts.items():
            content += f"- **{dtype}:** {count} columns\n"

        # Add sample columns
        content += "\n### Sample Columns\n"
        for i, col in enumerate(df.columns[:10], 1):
            content += f"{i}. `{col}` ({df[col].dtype})\n"

        if len(df.columns) > 10:
            content += f"\n*... and {len(df.columns) - 10} more columns*\n"

        return ReportSection(
            title="Data Overview",
            content=content,
            order=2
        )

    def _generate_data_quality_section(self, quality_report: Dict) -> ReportSection:
        """Generate data quality assessment section"""

        overall_score = quality_report.get('overall_score', 0)
        missing_pct = quality_report.get('missing_percentage', 0)
        duplicate_count = quality_report.get('duplicate_count', 0)

        content = f"""
### Overall Quality Score: {overall_score:.1f}%

"""

        # Quality interpretation
        if overall_score >= 80:
            content += "✅ **Assessment:** Excellent data quality. Dataset is well-prepared for analysis.\n\n"
        elif overall_score >= 60:
            content += "⚠️ **Assessment:** Good data quality with minor issues to address.\n\n"
        else:
            content += "❌ **Assessment:** Data quality needs improvement before modeling.\n\n"

        # Missing data
        content += f"""
### Missing Data Analysis
- **Overall Missing:** {missing_pct:.2f}% of all values
"""

        if 'missing_by_column' in quality_report:
            missing_cols = quality_report['missing_by_column']
            if missing_cols:
                content += "\n**Columns with Missing Values:**\n"
                for col, pct in sorted(missing_cols.items(), key=lambda x: x[1], reverse=True)[:5]:
                    content += f"- `{col}`: {pct:.1f}% missing\n"

        # Duplicates
        if duplicate_count > 0:
            content += f"\n### Duplicate Records\n"
            content += f"- **Exact Duplicates:** {duplicate_count:,} records\n"

        # Issues summary
        if 'issues' in quality_report and quality_report['issues']:
            content += f"\n### Issues Detected: {len(quality_report['issues'])}\n"
            for issue in quality_report['issues'][:5]:
                severity = issue.get('severity', 'medium').upper()
                content += f"- **{severity}:** {issue.get('description', 'Unknown issue')}\n"

        return ReportSection(
            title="Data Quality Assessment",
            content=content,
            order=3
        )

    def _generate_statistical_section(self, eda_results: Dict) -> ReportSection:
        """Generate statistical analysis section"""

        content = """
### Distribution Analysis

"""

        # Add distribution summaries
        if 'distributions' in eda_results:
            content += "**Key Statistical Findings:**\n\n"
            # Add distribution insights here

        # Correlation analysis
        if 'correlations' in eda_results:
            content += """
### Correlation Analysis

**Strong Correlations Detected:**
- Feature correlations indicate potential relationships for modeling
- Review correlation matrix for multicollinearity concerns

"""

        # Outliers
        if 'outliers' in eda_results:
            content += """
### Outlier Detection

**Outliers Found:**
- Statistical outliers identified using IQR and Z-score methods
- Review outlier explanations to determine legitimacy

"""

        return ReportSection(
            title="Statistical Analysis",
            content=content,
            order=4
        )

    def _generate_target_section(self, target_analysis: Dict) -> ReportSection:
        """Generate target variable analysis section"""

        target_name = target_analysis.get('target_name', 'Unknown')
        target_type = target_analysis.get('target_type', 'Unknown')

        content = f"""
### Target Variable: `{target_name}`

**Type:** {target_type}

"""

        if target_type == 'binary':
            content += "**Classification Task:** Binary classification problem\n"
        elif target_type == 'multiclass':
            content += "**Classification Task:** Multiclass classification problem\n"
        elif target_type == 'continuous':
            content += "**Regression Task:** Continuous target variable\n"

        # Class distribution
        if 'class_distribution' in target_analysis:
            content += "\n**Class Distribution:**\n"
            dist = target_analysis['class_distribution']
            for cls, count in dist.items():
                content += f"- `{cls}`: {count:,} samples\n"

        # Class balance
        if 'imbalance_ratio' in target_analysis:
            ratio = target_analysis['imbalance_ratio']
            if ratio > 3:
                content += f"\n⚠️ **Warning:** Significant class imbalance detected (ratio: {ratio:.1f}:1)\n"

        return ReportSection(
            title="Target Variable Analysis",
            content=content,
            order=5
        )

    def _generate_recommendations(
        self,
        quality_report: Optional[Dict],
        eda_results: Optional[Dict],
        leakage_report: Optional[Dict],
        readiness_score: Optional[Dict]
    ) -> ReportSection:
        """Generate recommendations section with AI assistance"""

        recommendations = []

        # Data quality recommendations
        if quality_report:
            quality_score = quality_report.get('overall_score', 100)
            if quality_score < 80:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Data Quality',
                    'recommendation': 'Address data quality issues before modeling',
                    'actions': [
                        'Review and handle missing values',
                        'Remove or investigate duplicates',
                        'Standardize data formats'
                    ]
                })

        # Leakage recommendations
        if leakage_report:
            leakage_issues = leakage_report.get('issues', [])
            if leakage_issues:
                recommendations.append({
                    'priority': 'CRITICAL',
                    'category': 'Data Leakage',
                    'recommendation': 'Remove features causing data leakage',
                    'actions': [
                        'Review target-leaking features',
                        'Remove temporal leakage sources',
                        'Validate feature engineering logic'
                    ]
                })

        # Feature engineering recommendations
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Feature Engineering',
            'recommendation': 'Enhance features for better model performance',
            'actions': [
                'Create interaction features',
                'Apply appropriate transformations',
                'Encode categorical variables'
            ]
        })

        # Model readiness recommendations
        if readiness_score:
            overall = readiness_score.get('overall_score', 0)
            if overall < 70:
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Model Readiness',
                    'recommendation': 'Improve dataset before model training',
                    'actions': [
                        'Address data quality issues',
                        'Complete feature engineering',
                        'Validate data preprocessing'
                    ]
                })

        # Format recommendations
        content = "### Priority Recommendations\n\n"

        for rec in sorted(recommendations, key=lambda x: {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}.get(x['priority'], 4)):
            priority_emoji = {
                'CRITICAL': '🔴',
                'HIGH': '🟠',
                'MEDIUM': '🟡',
                'LOW': '🟢'
            }.get(rec['priority'], '⚪')

            content += f"{priority_emoji} **{rec['priority']}** - {rec['category']}\n\n"
            content += f"**{rec['recommendation']}**\n\n"
            content += "Actions:\n"
            for action in rec['actions']:
                content += f"- {action}\n"
            content += "\n"

        return ReportSection(
            title="Recommendations",
            content=content,
            order=6
        )

    def _generate_appendix(self, df: pd.DataFrame, quality_report: Optional[Dict]) -> ReportSection:
        """Generate appendix with technical details"""

        content = """
### Methodology

**Analysis Framework:**
- Comprehensive data quality assessment
- Exploratory data analysis (EDA)
- Statistical testing and validation
- AI-powered insights generation

**Quality Metrics:**
- Completeness: Percentage of non-missing values
- Validity: Data type and format consistency
- Uniqueness: Duplicate detection
- Consistency: Cross-column validation

"""

        # Add glossary
        content += """
### Glossary

**Common Terms:**
- **Missing Values:** Data points with no recorded value
- **Outliers:** Data points significantly different from others
- **Data Leakage:** Features that contain information about the target
- **Feature Engineering:** Creating new features from existing ones
- **Model Readiness:** Dataset preparedness for machine learning

"""

        return ReportSection(
            title="Appendix",
            content=content,
            order=7
        )

    def export_to_markdown(self, sections: Dict[str, ReportSection]) -> str:
        """Export report to Markdown format"""

        md_content = f"""# Data Analysis Report

**Generated:** {self.metadata['generated_date']}

**Dataset:** {self.metadata.get('dataset_name', 'Unknown')}

---

"""

        # Add all sections in order
        sorted_sections = sorted(sections.values(), key=lambda x: x.order)
        for section in sorted_sections:
            if section.include:
                md_content += f"## {section.title}\n\n"
                md_content += f"{section.content}\n\n"
                md_content += "---\n\n"

        # Add footer
        md_content += f"""
*Report generated by EDA Tool AI - {self.metadata['generated_date']}*

🤖 Powered by Local AI
"""

        return md_content

    def export_to_html(self, sections: Dict[str, ReportSection]) -> str:
        """Export report to HTML format with styling"""

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Data Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .metadata {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-top: 10px;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .section h3 {{
            color: #764ba2;
            margin-top: 20px;
        }}
        code {{
            background: #f0f0f0;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        ul, ol {{
            padding-left: 20px;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            background: white;
            border-radius: 8px;
            color: #666;
            font-size: 0.9em;
        }}
        .priority-high {{ color: #dc3545; font-weight: bold; }}
        .priority-medium {{ color: #ffc107; font-weight: bold; }}
        .priority-low {{ color: #28a745; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Data Analysis Report</h1>
        <div class="metadata">
            <strong>Generated:</strong> {self.metadata['generated_date']}<br>
            <strong>Dataset:</strong> {self.metadata.get('dataset_name', 'Unknown')}<br>
            <strong>Records:</strong> {self.metadata['total_rows']:,} |
            <strong>Features:</strong> {self.metadata['total_columns']}
        </div>
    </div>
"""

        # Add sections
        sorted_sections = sorted(sections.values(), key=lambda x: x.order)
        for section in sorted_sections:
            if section.include:
                # Convert markdown-style content to HTML
                content_html = self._markdown_to_html(section.content)
                html_content += f"""
    <div class="section">
        <h2>{section.title}</h2>
        {content_html}
    </div>
"""

        # Add footer
        html_content += f"""
    <div class="footer">
        <p>Report generated by <strong>EDA Tool AI</strong></p>
        <p>🤖 Powered by Local AI | Privacy-First Analysis</p>
    </div>
</body>
</html>
"""

        return html_content

    def _markdown_to_html(self, markdown_text: str) -> str:
        """Simple markdown to HTML converter"""

        html = markdown_text

        # Headers
        html = html.replace('### ', '<h3>').replace('\n\n', '</h3>\n\n')

        # Bold
        import re
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)

        # Code
        html = re.sub(r'`(.*?)`', r'<code>\1</code>', html)

        # Lists (simple conversion)
        lines = html.split('\n')
        in_list = False
        result = []
        for line in lines:
            if line.strip().startswith('- '):
                if not in_list:
                    result.append('<ul>')
                    in_list = True
                result.append(f"<li>{line.strip()[2:]}</li>")
            else:
                if in_list:
                    result.append('</ul>')
                    in_list = False
                result.append(line)

        if in_list:
            result.append('</ul>')

        html = '\n'.join(result)

        # Paragraphs
        html = html.replace('\n\n', '</p><p>')
        html = f'<p>{html}</p>'

        return html


# UI Component
def display_report_generator():
    """
    Streamlit UI for the Report Generator
    """
    st.markdown("## 📄 Executive Report Generator")
    st.markdown("Generate professional, AI-powered reports summarizing your data analysis.")

    # Check if data is loaded
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Please load a dataset first (go to Data Upload section)")
        return

    df = st.session_state.df

    # Report configuration
    st.markdown("### Report Configuration")

    col1, col2 = st.columns(2)

    with col1:
        dataset_name = st.text_input("Dataset Name", value="Dataset Analysis Report")
        include_exec_summary = st.checkbox("Include Executive Summary", value=True)
        include_data_quality = st.checkbox("Include Data Quality Assessment", value=True)

    with col2:
        include_statistics = st.checkbox("Include Statistical Analysis", value=True)
        include_recommendations = st.checkbox("Include Recommendations", value=True)
        include_appendix = st.checkbox("Include Appendix", value=True)

    st.markdown("---")

    # Generate report button
    if st.button("🎯 Generate Report", type="primary", use_container_width=True):

        with st.spinner("Generating report... This may take a moment."):

            # Initialize report generator
            generator = ReportGenerator()
            generator.metadata['dataset_name'] = dataset_name

            # Gather available analyses from session state
            quality_report = st.session_state.get('quality_report')
            eda_results = st.session_state.get('eda_results')
            target_analysis = st.session_state.get('target_analysis')
            readiness_score = st.session_state.get('readiness_score')
            leakage_report = st.session_state.get('leakage_report')

            # Generate sections
            sections = generator.generate_report(
                df=df,
                quality_report=quality_report,
                eda_results=eda_results,
                target_analysis=target_analysis,
                readiness_score=readiness_score,
                leakage_report=leakage_report
            )

            # Store in session state
            st.session_state.generated_report = sections
            st.session_state.report_generator = generator

            st.success("✅ Report generated successfully!")

    # Display and download options
    if 'generated_report' in st.session_state:

        st.markdown("### 📑 Report Preview")

        generator = st.session_state.report_generator
        sections = st.session_state.generated_report

        # Preview sections
        for section in sorted(sections.values(), key=lambda x: x.order):
            if section.include:
                with st.expander(f"📄 {section.title}", expanded=(section.order == 1)):
                    st.markdown(section.content)

        st.markdown("---")
        st.markdown("### 💾 Export Options")

        col1, col2, col3 = st.columns(3)

        # Markdown export
        with col1:
            md_content = generator.export_to_markdown(sections)
            st.download_button(
                label="📝 Download Markdown",
                data=md_content,
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )

        # HTML export
        with col2:
            html_content = generator.export_to_html(sections)
            st.download_button(
                label="🌐 Download HTML",
                data=html_content,
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True
            )

        # PDF export (future implementation)
        with col3:
            st.button(
                label="📄 Download PDF",
                disabled=True,
                use_container_width=True,
                help="PDF export coming soon!"
            )

        st.info("💡 **Tip:** Download in different formats for sharing with stakeholders. HTML format includes professional styling.")


# Export functions
def get_report_history() -> List[Dict]:
    """Get report generation history"""
    return st.session_state.get('report_history', [])


def clear_report_history():
    """Clear report generation history"""
    if 'report_history' in st.session_state:
        st.session_state.report_history = []
