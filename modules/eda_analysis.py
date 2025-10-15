import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import warnings

warnings.filterwarnings('ignore')

def generate_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive summary statistics for the dataset.

    Args:
        df: Input DataFrame

    Returns:
        Dict containing summary statistics
    """
    summary = {
        'numeric_summary': {},
        'categorical_summary': {},
        'datetime_summary': {},
        'overall_insights': {}
    }

    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_stats = df[numeric_cols].describe()

        # Add additional statistics
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 1:
                summary['numeric_summary'][col] = {
                    'count': len(col_data),
                    'missing': df[col].isnull().sum(),
                    'missing_pct': round((df[col].isnull().sum() / len(df)) * 100, 2),
                    'mean': round(col_data.mean(), 3),
                    'median': round(col_data.median(), 3),
                    'std': round(col_data.std(), 3),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'q25': round(col_data.quantile(0.25), 3),
                    'q75': round(col_data.quantile(0.75), 3),
                    'skewness': round(col_data.skew(), 3),
                    'kurtosis': round(col_data.kurtosis(), 3),
                    'cv': round(col_data.std() / col_data.mean(), 3) if col_data.mean() != 0 else 0,
                    'range': col_data.max() - col_data.min(),
                    'iqr': col_data.quantile(0.75) - col_data.quantile(0.25)
                }

    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            value_counts = col_data.value_counts()

            summary['categorical_summary'][col] = {
                'count': len(col_data),
                'missing': df[col].isnull().sum(),
                'missing_pct': round((df[col].isnull().sum() / len(df)) * 100, 2),
                'unique': col_data.nunique(),
                'unique_pct': round((col_data.nunique() / len(col_data)) * 100, 2),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'most_frequent_pct': round((value_counts.iloc[0] / len(col_data)) * 100, 2) if len(value_counts) > 0 else 0,
                'top_5_values': dict(value_counts.head(5)),
                'entropy': calculate_entropy(col_data)
            }

    # DateTime columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            summary['datetime_summary'][col] = {
                'count': len(col_data),
                'missing': df[col].isnull().sum(),
                'missing_pct': round((df[col].isnull().sum() / len(df)) * 100, 2),
                'min_date': col_data.min(),
                'max_date': col_data.max(),
                'date_range_days': (col_data.max() - col_data.min()).days if len(col_data) > 1 else 0,
                'unique_dates': col_data.nunique()
            }

    # Overall insights
    summary['overall_insights'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numeric_columns': len(numeric_cols),
        'categorical_columns': len(categorical_cols),
        'datetime_columns': len(datetime_cols),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        'sparsity': round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2)
    }

    return summary

def analyze_distributions(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """
    Analyze distributions of numeric features.

    Args:
        df: Input DataFrame
        numeric_cols: List of numeric columns

    Returns:
        Dict containing distribution analysis
    """
    distribution_analysis = {
        'normality_tests': {},
        'distribution_insights': {},
        'skewness_analysis': {},
        'transformation_suggestions': {}
    }

    for col in numeric_cols:
        col_data = df[col].dropna()

        if len(col_data) > 3:  # Need at least 4 points for analysis
            # Normality tests
            normality = {}

            # Shapiro-Wilk test (for smaller samples)
            if len(col_data) <= 5000:
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(col_data)
                    normality['shapiro_wilk'] = {
                        'statistic': round(shapiro_stat, 4),
                        'p_value': round(shapiro_p, 6),
                        'is_normal': shapiro_p > 0.05
                    }
                except:
                    normality['shapiro_wilk'] = None

            # Kolmogorov-Smirnov test
            try:
                ks_stat, ks_p = stats.kstest(col_data, 'norm')
                normality['kolmogorov_smirnov'] = {
                    'statistic': round(ks_stat, 4),
                    'p_value': round(ks_p, 6),
                    'is_normal': ks_p > 0.05
                }
            except:
                normality['kolmogorov_smirnov'] = None

            distribution_analysis['normality_tests'][col] = normality

            # Distribution insights
            skewness = col_data.skew()
            kurtosis = col_data.kurtosis()

            insights = {
                'skewness': round(skewness, 3),
                'kurtosis': round(kurtosis, 3),
                'skewness_interpretation': interpret_skewness(skewness),
                'kurtosis_interpretation': interpret_kurtosis(kurtosis),
                'distribution_shape': classify_distribution_shape(skewness, kurtosis)
            }

            distribution_analysis['distribution_insights'][col] = insights

            # Transformation suggestions
            suggestions = suggest_transformations(col_data, skewness)
            distribution_analysis['transformation_suggestions'][col] = suggestions

    return distribution_analysis

def calculate_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate various types of correlations.

    Args:
        df: Input DataFrame

    Returns:
        Dict containing correlation analysis
    """
    correlation_analysis = {
        'pearson_correlation': {},
        'spearman_correlation': {},
        'high_correlations': [],
        'correlation_summary': {}
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 1:
        # Pearson correlation
        pearson_corr = df[numeric_cols].corr(method='pearson')
        correlation_analysis['pearson_correlation'] = pearson_corr

        # Spearman correlation
        spearman_corr = df[numeric_cols].corr(method='spearman')
        correlation_analysis['spearman_correlation'] = spearman_corr

        # Find high correlations
        high_corr_pairs = find_high_correlations(pearson_corr, threshold=0.7)
        correlation_analysis['high_correlations'] = high_corr_pairs

        # Correlation summary
        correlation_analysis['correlation_summary'] = {
            'total_pairs': len(numeric_cols) * (len(numeric_cols) - 1) // 2,
            'high_correlation_pairs': len(high_corr_pairs),
            'max_correlation': float(pearson_corr.abs().values[np.triu_indices_from(pearson_corr.values, k=1)].max()) if len(numeric_cols) > 1 else 0,
            'mean_absolute_correlation': float(pearson_corr.abs().values[np.triu_indices_from(pearson_corr.values, k=1)].mean()) if len(numeric_cols) > 1 else 0
        }

    return correlation_analysis

def analyze_feature_target_relationship(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Analyze relationship between features and target variable.

    Args:
        df: Input DataFrame
        target_col: Name of target column

    Returns:
        Dict containing feature-target analysis
    """
    if target_col not in df.columns:
        return {'error': f'Target column {target_col} not found in dataset'}

    analysis = {
        'target_info': {},
        'numeric_feature_relationships': {},
        'categorical_feature_relationships': {},
        'feature_importance_proxy': {}
    }

    target_data = df[target_col].dropna()

    # Analyze target variable
    target_info = {
        'type': 'numeric' if pd.api.types.is_numeric_dtype(df[target_col]) else 'categorical',
        'unique_values': df[target_col].nunique(),
        'missing_count': df[target_col].isnull().sum(),
        'missing_pct': round((df[target_col].isnull().sum() / len(df)) * 100, 2)
    }

    if pd.api.types.is_numeric_dtype(df[target_col]):
        target_info.update({
            'mean': round(target_data.mean(), 3),
            'std': round(target_data.std(), 3),
            'min': target_data.min(),
            'max': target_data.max(),
            'skewness': round(target_data.skew(), 3)
        })
    else:
        value_counts = target_data.value_counts()
        target_info.update({
            'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
            'class_distribution': dict(value_counts),
            'balance_ratio': round(value_counts.min() / value_counts.max(), 3) if len(value_counts) > 1 else 1.0
        })

    analysis['target_info'] = target_info

    # Analyze numeric features vs target
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_cols if col != target_col]

    for feature in numeric_features:
        feature_data = df[feature].dropna()
        target_subset = df.loc[feature_data.index, target_col]

        relationship = {}

        if pd.api.types.is_numeric_dtype(df[target_col]):
            # Numeric target - calculate correlation
            correlation = feature_data.corr(target_subset)
            relationship = {
                'correlation': round(correlation, 4),
                'correlation_strength': interpret_correlation_strength(abs(correlation)),
                'relationship_type': 'positive' if correlation > 0 else 'negative' if correlation < 0 else 'none'
            }
        else:
            # Categorical target - calculate mutual information
            try:
                mi_score = mutual_info_classif(feature_data.values.reshape(-1, 1), target_subset, random_state=42)[0]
                relationship = {
                    'mutual_information': round(mi_score, 4),
                    'relationship_strength': interpret_mutual_information(mi_score)
                }
            except:
                relationship = {'error': 'Could not calculate mutual information'}

        analysis['numeric_feature_relationships'][feature] = relationship

    # Analyze categorical features vs target
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_features = [col for col in categorical_cols if col != target_col]

    for feature in categorical_features:
        # Create contingency table and perform chi-square test
        try:
            contingency_table = pd.crosstab(df[feature], df[target_col])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

            relationship = {
                'chi_square': round(chi2, 4),
                'p_value': round(p_value, 6),
                'degrees_of_freedom': dof,
                'significant': p_value < 0.05,
                'cramers_v': calculate_cramers_v(chi2, contingency_table.sum().sum(), min(contingency_table.shape) - 1)
            }
        except:
            relationship = {'error': 'Could not perform chi-square test'}

        analysis['categorical_feature_relationships'][feature] = relationship

    return analysis

def generate_pairplot_data(df: pd.DataFrame, selected_cols: List[str], target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate data for pairplot visualization.

    Args:
        df: Input DataFrame
        selected_cols: List of columns to include in pairplot
        target_col: Optional target column for color coding

    Returns:
        Dict containing pairplot data and recommendations
    """
    if len(selected_cols) > 10:
        return {
            'error': 'Too many columns selected. Please select 10 or fewer columns for performance.',
            'recommendation': 'Select the most important features based on correlation or domain knowledge.'
        }

    pairplot_data = {
        'selected_columns': selected_cols,
        'target_column': target_col,
        'data_subset': df[selected_cols].copy(),
        'correlation_matrix': df[selected_cols].select_dtypes(include=[np.number]).corr(),
        'recommendations': []
    }

    # Add recommendations
    numeric_cols = df[selected_cols].select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        pairplot_data['recommendations'].append("Add more numeric columns for meaningful pairplot analysis.")

    if target_col and target_col in df.columns:
        pairplot_data['data_subset'][target_col] = df[target_col]
        if df[target_col].nunique() > 10:
            pairplot_data['recommendations'].append("Target has many unique values. Consider binning for better visualization.")

    return pairplot_data

# Helper functions

def calculate_entropy(series: pd.Series) -> float:
    """Calculate entropy of a categorical series."""
    value_counts = series.value_counts()
    probabilities = value_counts / len(series)
    entropy = -sum(probabilities * np.log2(probabilities + 1e-10))
    return round(entropy, 3)

def interpret_skewness(skewness: float) -> str:
    """Interpret skewness value."""
    if abs(skewness) < 0.5:
        return "Approximately symmetric"
    elif abs(skewness) < 1:
        return "Moderately skewed"
    else:
        return "Highly skewed"

def interpret_kurtosis(kurtosis: float) -> str:
    """Interpret kurtosis value."""
    if kurtosis < -1:
        return "Platykurtic (flatter than normal)"
    elif kurtosis > 1:
        return "Leptokurtic (more peaked than normal)"
    else:
        return "Mesokurtic (similar to normal)"

def classify_distribution_shape(skewness: float, kurtosis: float) -> str:
    """Classify overall distribution shape."""
    if abs(skewness) < 0.5 and abs(kurtosis) < 1:
        return "Approximately normal"
    elif skewness > 0.5:
        return "Right-skewed"
    elif skewness < -0.5:
        return "Left-skewed"
    elif kurtosis > 1:
        return "Heavy-tailed"
    elif kurtosis < -1:
        return "Light-tailed"
    else:
        return "Non-normal"

def suggest_transformations(data: pd.Series, skewness: float) -> List[str]:
    """Suggest appropriate transformations based on data characteristics."""
    suggestions = []

    if abs(skewness) < 0.5:
        suggestions.append("No transformation needed - data is approximately normal")
    elif skewness > 1:
        suggestions.extend([
            "Log transformation to reduce right skewness",
            "Square root transformation",
            "Box-Cox transformation"
        ])
    elif skewness < -1:
        suggestions.extend([
            "Square transformation to reduce left skewness",
            "Exponential transformation"
        ])
    else:
        suggestions.append("Consider standardization or normalization")

    # Check for zeros and negative values
    if (data <= 0).any():
        suggestions.append("Note: Data contains zeros or negative values - some transformations may not be applicable")

    return suggestions

def find_high_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
    """Find pairs of features with high correlation."""
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': round(corr_value, 4),
                    'abs_correlation': round(abs(corr_value), 4)
                })

    # Sort by absolute correlation descending
    high_corr_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)

    return high_corr_pairs

def interpret_correlation_strength(correlation: float) -> str:
    """Interpret correlation strength."""
    abs_corr = abs(correlation)
    if abs_corr < 0.1:
        return "Very weak"
    elif abs_corr < 0.3:
        return "Weak"
    elif abs_corr < 0.5:
        return "Moderate"
    elif abs_corr < 0.7:
        return "Strong"
    else:
        return "Very strong"

def interpret_mutual_information(mi_score: float) -> str:
    """Interpret mutual information score."""
    if mi_score < 0.1:
        return "Very weak relationship"
    elif mi_score < 0.3:
        return "Weak relationship"
    elif mi_score < 0.5:
        return "Moderate relationship"
    else:
        return "Strong relationship"

def calculate_cramers_v(chi2: float, n: int, df: int) -> float:
    """Calculate Cramer's V for categorical association."""
    if df == 0:
        return 0
    return round(np.sqrt(chi2 / (n * df)), 4)

def comprehensive_eda_report(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive EDA report.

    Args:
        df: Input DataFrame
        target_col: Optional target column name

    Returns:
        Dict containing complete EDA analysis
    """
    report = {
        'summary_statistics': generate_summary_statistics(df),
        'correlations': calculate_correlations(df),
        'distributions': analyze_distributions(df, df.select_dtypes(include=[np.number]).columns.tolist()),
    }

    if target_col and target_col in df.columns:
        report['target_analysis'] = analyze_feature_target_relationship(df, target_col)

    return report