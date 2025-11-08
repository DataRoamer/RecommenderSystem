"""
Context Builder for LLM Prompts
Creates concise, informative context from datasets and analysis results
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Any


def build_dataset_context(
    df: pd.DataFrame,
    max_rows_sample: int = 5,
    max_columns: int = 20,
    include_sample: bool = True
) -> str:
    """
    Build a concise context about the dataset for LLM

    Args:
        df: DataFrame to analyze
        max_rows_sample: Number of sample rows to include
        max_columns: Maximum columns to include in detail
        include_sample: Whether to include sample data

    Returns:
        Formatted context string
    """
    context_parts = []

    # Basic info
    context_parts.append(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    context_parts.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

    context_parts.append(f"\nColumn Types:")
    context_parts.append(f"- Numeric: {len(numeric_cols)} columns")
    context_parts.append(f"- Categorical: {len(categorical_cols)} columns")
    context_parts.append(f"- DateTime: {len(datetime_cols)} columns")

    # Missing data summary
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0

    context_parts.append(f"\nMissing Data: {total_missing:,} cells ({missing_pct:.1f}%)")

    # Column details (limited to most important)
    columns_to_describe = df.columns[:max_columns]

    context_parts.append(f"\nColumn Details:")
    for col in columns_to_describe:
        col_info = []
        col_info.append(f"  - {col}")
        col_info.append(f"Type: {df[col].dtype}")
        col_info.append(f"Missing: {df[col].isnull().sum()} ({df[col].isnull().mean()*100:.1f}%)")

        if col in numeric_cols:
            col_info.append(f"Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
            col_info.append(f"Mean: {df[col].mean():.2f}")
        elif col in categorical_cols:
            col_info.append(f"Unique: {df[col].nunique()}")
            if df[col].nunique() <= 5:
                top_vals = df[col].value_counts().head(3).to_dict()
                col_info.append(f"Top values: {top_vals}")

        context_parts.append(", ".join(col_info))

    # Sample data (if requested)
    if include_sample and max_rows_sample > 0:
        context_parts.append(f"\nSample Data (first {max_rows_sample} rows):")
        sample = df.head(max_rows_sample).to_dict(orient='records')
        context_parts.append(json.dumps(sample, indent=2, default=str))

    return "\n".join(context_parts)


def build_analysis_context(
    quality_report: Optional[Dict] = None,
    eda_report: Optional[Dict] = None,
    target_analysis: Optional[Dict] = None
) -> str:
    """
    Build context from analysis reports

    Args:
        quality_report: Data quality analysis results
        eda_report: EDA analysis results
        target_analysis: Target variable analysis results

    Returns:
        Formatted context string
    """
    context_parts = []

    # Quality report summary
    if quality_report:
        context_parts.append("=== DATA QUALITY ANALYSIS ===")

        quality_score = quality_report.get('quality_score', {})
        context_parts.append(f"Overall Quality Score: {quality_score.get('overall_score', 0)}/100")
        context_parts.append(f"Interpretation: {quality_score.get('interpretation', 'N/A')}")

        # Missing values
        missing = quality_report.get('missing_values', {})
        context_parts.append(f"\nMissing Values:")
        context_parts.append(f"- Total: {missing.get('total_missing', 0):,}")
        context_parts.append(f"- Percentage: {missing.get('overall_missing_percentage', 0):.1f}%")

        if missing.get('critical_columns'):
            context_parts.append(f"- Critical columns (>50% missing): {', '.join(missing['critical_columns'])}")

        # Duplicates
        duplicates = quality_report.get('duplicates', {})
        context_parts.append(f"\nDuplicates:")
        context_parts.append(f"- Duplicate rows: {duplicates.get('duplicate_rows', 0):,}")
        context_parts.append(f"- Percentage: {duplicates.get('duplicate_percentage', 0):.1f}%")

        # Outliers
        outliers = quality_report.get('outliers_iqr', {}).get('summary', {})
        context_parts.append(f"\nOutliers (IQR method):")
        context_parts.append(f"- Total outliers: {outliers.get('total_outliers', 0):,}")

    # EDA report summary
    if eda_report:
        context_parts.append("\n=== EXPLORATORY DATA ANALYSIS ===")

        summary = eda_report.get('summary_statistics', {}).get('overall_insights', {})
        context_parts.append(f"Dataset Statistics:")
        context_parts.append(f"- Total rows: {summary.get('total_rows', 0):,}")
        context_parts.append(f"- Total columns: {summary.get('total_columns', 0)}")
        context_parts.append(f"- Memory usage: {summary.get('memory_usage_mb', 0):.1f} MB")
        context_parts.append(f"- Data sparsity: {summary.get('sparsity', 0):.1f}%")

        # High correlations
        if 'correlations' in eda_report:
            high_corr = eda_report['correlations'].get('high_correlations', [])
            if high_corr:
                context_parts.append(f"\nHigh Correlations:")
                for corr in high_corr[:5]:  # Limit to top 5
                    context_parts.append(
                        f"- {corr['feature1']} <-> {corr['feature2']}: {corr['correlation']:.3f}"
                    )

    # Target analysis summary
    if target_analysis:
        context_parts.append("\n=== TARGET VARIABLE ANALYSIS ===")

        basic_info = target_analysis.get('basic_info', {})
        task_type = basic_info.get('suggested_task_type', 'unknown')

        context_parts.append(f"Task Type: {task_type.title()}")
        context_parts.append(f"Target Column: {basic_info.get('target_column', 'N/A')}")
        context_parts.append(f"Missing Values: {basic_info.get('missing_percentage', 0):.1f}%")
        context_parts.append(f"Unique Values: {basic_info.get('unique_values', 0)}")

        if task_type == 'classification':
            class_analysis = target_analysis.get('classification_analysis', {})
            context_parts.append(f"\nClass Distribution:")
            context_parts.append(f"- Number of classes: {class_analysis.get('number_of_classes', 0)}")
            context_parts.append(f"- Balance ratio: {class_analysis.get('balance_ratio', 'N/A')}")
            context_parts.append(f"- Imbalance severity: {class_analysis.get('imbalance_severity', 'N/A')}")

        elif task_type == 'regression':
            reg_analysis = target_analysis.get('regression_analysis', {})
            context_parts.append(f"\nTarget Statistics:")
            context_parts.append(f"- Mean: {reg_analysis.get('mean', 0):.2f}")
            context_parts.append(f"- Median: {reg_analysis.get('median', 0):.2f}")
            context_parts.append(f"- Std Dev: {reg_analysis.get('std', 0):.2f}")
            context_parts.append(f"- Skewness: {reg_analysis.get('skewness', 0):.2f}")

        # ML Readiness
        if 'ml_readiness' in target_analysis:
            readiness = target_analysis['ml_readiness']
            context_parts.append(f"\nML Readiness Score: {readiness.get('overall_score', 0)}/100")
            context_parts.append(f"Interpretation: {readiness.get('interpretation', 'N/A')}")

    return "\n".join(context_parts)


def build_code_generation_context(
    df: pd.DataFrame,
    task_description: str,
    column_names: Optional[List[str]] = None
) -> str:
    """
    Build context for code generation tasks

    Args:
        df: DataFrame
        task_description: What code to generate
        column_names: Specific columns to focus on

    Returns:
        Context string for code generation
    """
    context_parts = []

    context_parts.append(f"Task: {task_description}")
    context_parts.append(f"\nDataset Info:")
    context_parts.append(f"- Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Column info
    if column_names:
        context_parts.append(f"\nRelevant Columns:")
        for col in column_names:
            if col in df.columns:
                context_parts.append(f"- {col}: {df[col].dtype}, {df[col].nunique()} unique values")
    else:
        context_parts.append(f"\nAll Columns:")
        context_parts.append(f"- {', '.join(df.columns.tolist())}")

    context_parts.append(f"\nAvailable libraries: pandas, numpy, matplotlib, seaborn, sklearn")
    context_parts.append(f"DataFrame variable name: df")

    return "\n".join(context_parts)


def build_insight_context(
    df: pd.DataFrame,
    quality_report: Dict,
    eda_report: Dict,
    focus_areas: Optional[List[str]] = None
) -> str:
    """
    Build focused context for insight generation

    Args:
        df: DataFrame
        quality_report: Quality analysis
        eda_report: EDA analysis
        focus_areas: Specific areas to focus on (e.g., ['correlations', 'outliers'])

    Returns:
        Focused context string
    """
    context_parts = []

    # Always include basic info
    context_parts.append(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

    quality_score = quality_report.get('quality_score', {}).get('overall_score', 0)
    context_parts.append(f"Quality Score: {quality_score}/100")

    # Focus on specific areas if requested
    if not focus_areas:
        focus_areas = ['missing_values', 'correlations', 'outliers', 'distributions']

    if 'missing_values' in focus_areas:
        missing = quality_report.get('missing_values', {})
        if missing.get('total_missing', 0) > 0:
            context_parts.append(f"\nMissing Data: {missing['overall_missing_percentage']:.1f}%")
            cols_with_missing = missing.get('columns_with_missing', {})
            if cols_with_missing:
                top_missing = sorted(
                    cols_with_missing.items(),
                    key=lambda x: x[1]['percentage'],
                    reverse=True
                )[:3]
                context_parts.append("Top columns with missing data:")
                for col, info in top_missing:
                    context_parts.append(f"  - {col}: {info['percentage']:.1f}%")

    if 'correlations' in focus_areas:
        if 'correlations' in eda_report:
            high_corr = eda_report['correlations'].get('high_correlations', [])
            if high_corr:
                context_parts.append(f"\nHigh Correlations ({len(high_corr)} pairs):")
                for corr in high_corr[:3]:
                    context_parts.append(
                        f"  - {corr['feature1']} <-> {corr['feature2']}: {corr['correlation']:.3f}"
                    )

    if 'outliers' in focus_areas:
        outliers = quality_report.get('outliers_iqr', {})
        if outliers.get('summary', {}).get('total_outliers', 0) > 0:
            context_parts.append(f"\nOutliers: {outliers['summary']['total_outliers']:,} detected")

    if 'distributions' in focus_areas:
        if 'distributions' in eda_report:
            dist_insights = eda_report['distributions'].get('distribution_insights', {})
            skewed_features = [
                col for col, info in dist_insights.items()
                if abs(info.get('skewness', 0)) > 1
            ]
            if skewed_features:
                context_parts.append(f"\nHighly Skewed Features: {len(skewed_features)}")
                context_parts.append(f"  {', '.join(skewed_features[:5])}")

    return "\n".join(context_parts)


def truncate_context(context: str, max_tokens: int = 4000) -> str:
    """
    Truncate context to fit within token limit
    Simple heuristic: ~4 characters per token

    Args:
        context: Full context string
        max_tokens: Maximum tokens allowed

    Returns:
        Truncated context
    """
    max_chars = max_tokens * 4
    if len(context) <= max_chars:
        return context

    # Truncate and add notice
    truncated = context[:max_chars]
    return truncated + "\n\n[Context truncated to fit token limit]"
