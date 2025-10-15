import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze missing values in the DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Dict containing missing value analysis
    """
    missing_data = {
        'total_missing': df.isnull().sum().sum(),
        'total_cells': df.shape[0] * df.shape[1],
        'overall_missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
        'columns_with_missing': {},
        'missing_patterns': {},
        'critical_columns': []
    }

    # Per-column missing analysis
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100

        if missing_count > 0:
            missing_data['columns_with_missing'][col] = {
                'count': int(missing_count),
                'percentage': round(missing_pct, 2),
                'data_type': str(df[col].dtype)
            }

            # Flag critical columns (>50% missing)
            if missing_pct > 50:
                missing_data['critical_columns'].append(col)

    # Missing value patterns (combinations)
    missing_combinations = df.isnull().value_counts()
    missing_data['missing_patterns'] = {
        'total_patterns': len(missing_combinations),
        'most_common_pattern': missing_combinations.index[0] if len(missing_combinations) > 0 else None,
        'complete_rows': int(missing_combinations.get(tuple([False] * len(df.columns)), 0))
    }

    return missing_data

def detect_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Detect duplicate rows in the DataFrame.

    Args:
        df: Input DataFrame
        subset: List of columns to consider for duplicates

    Returns:
        Dict containing duplicate analysis
    """
    duplicate_analysis = {
        'total_rows': len(df),
        'duplicate_rows': 0,
        'duplicate_percentage': 0.0,
        'unique_rows': 0,
        'subset_used': subset
    }

    # Check for duplicates
    if subset:
        duplicates = df.duplicated(subset=subset)
    else:
        duplicates = df.duplicated()

    duplicate_count = duplicates.sum()

    duplicate_analysis.update({
        'duplicate_rows': int(duplicate_count),
        'duplicate_percentage': round((duplicate_count / len(df)) * 100, 2),
        'unique_rows': len(df) - duplicate_count
    })

    # Find specific duplicate examples
    if duplicate_count > 0:
        duplicate_indices = df[duplicates].index.tolist()
        duplicate_analysis['duplicate_indices'] = duplicate_indices[:10]  # First 10 examples

        # Get the actual duplicate rows for inspection
        if subset:
            duplicate_examples = df[df.duplicated(subset=subset, keep=False)].head(10)
        else:
            duplicate_examples = df[df.duplicated(keep=False)].head(10)
        duplicate_analysis['duplicate_examples'] = duplicate_examples

    return duplicate_analysis

def find_constant_features(df: pd.DataFrame, threshold: float = 0.01) -> Dict[str, Any]:
    """
    Find features with constant or near-constant values.

    Args:
        df: Input DataFrame
        threshold: Threshold for near-zero variance (default 0.01)

    Returns:
        Dict containing constant feature analysis
    """
    constant_analysis = {
        'constant_features': [],
        'near_constant_features': [],
        'low_variance_features': [],
        'analysis_details': {}
    }

    for col in df.columns:
        col_data = df[col].dropna()

        if len(col_data) == 0:
            continue

        # Check for constant features
        unique_values = col_data.nunique()
        unique_ratio = unique_values / len(col_data)

        analysis_detail = {
            'unique_count': unique_values,
            'unique_ratio': round(unique_ratio, 4),
            'most_common_value': None,
            'most_common_frequency': 0
        }

        # Get most common value
        if len(col_data) > 0:
            value_counts = col_data.value_counts()
            analysis_detail['most_common_value'] = value_counts.index[0]
            analysis_detail['most_common_frequency'] = round(value_counts.iloc[0] / len(col_data), 4)

        # Categorize features
        if unique_values == 1:
            constant_analysis['constant_features'].append(col)
        elif unique_ratio < 0.01:  # Less than 1% unique values
            constant_analysis['near_constant_features'].append(col)
        elif df[col].dtype in ['int64', 'float64'] and unique_values > 1:
            # Check variance for numeric columns
            try:
                variance = col_data.var()
                if variance < threshold:
                    constant_analysis['low_variance_features'].append(col)
                    analysis_detail['variance'] = variance
            except:
                pass

        constant_analysis['analysis_details'][col] = analysis_detail

    return constant_analysis

def check_data_type_consistency(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check for data type consistency issues.

    Args:
        df: Input DataFrame

    Returns:
        Dict containing data type analysis
    """
    type_analysis = {
        'mixed_type_columns': [],
        'potential_numeric': [],
        'potential_categorical': [],
        'potential_datetime': [],
        'potential_boolean': [],
        'recommendations': {}
    }

    for col in df.columns:
        col_data = df[col].dropna().astype(str)

        if len(col_data) == 0:
            continue

        recommendations = []

        # Check if object column could be numeric
        if df[col].dtype == 'object':
            # Try to convert to numeric
            numeric_count = 0
            for val in col_data.head(100):  # Sample first 100 values
                try:
                    float(val)
                    numeric_count += 1
                except:
                    break

            if numeric_count == len(col_data.head(100)):
                type_analysis['potential_numeric'].append(col)
                recommendations.append("Convert to numeric (int/float)")

            # Check for boolean-like values
            unique_vals = set(col_data.str.lower().unique())
            boolean_patterns = [
                {'true', 'false'},
                {'yes', 'no'},
                {'y', 'n'},
                {'1', '0'},
                {'on', 'off'}
            ]

            for pattern in boolean_patterns:
                if unique_vals.issubset(pattern) and len(unique_vals) == 2:
                    type_analysis['potential_boolean'].append(col)
                    recommendations.append("Convert to boolean")
                    break

            # Check for datetime patterns
            datetime_count = 0
            for val in col_data.head(20):
                try:
                    pd.to_datetime(val)
                    datetime_count += 1
                except:
                    break

            if datetime_count > 15:  # Most values look like dates
                type_analysis['potential_datetime'].append(col)
                recommendations.append("Convert to datetime")

        # Check for high cardinality that might need different treatment
        unique_count = df[col].nunique()
        unique_ratio = unique_count / len(df)

        if df[col].dtype == 'object' and unique_ratio > 0.95:
            recommendations.append("High cardinality - consider if this is an ID column")
        elif df[col].dtype in ['int64', 'float64'] and unique_count < 10:
            type_analysis['potential_categorical'].append(col)
            recommendations.append("Consider converting to categorical")

        if recommendations:
            type_analysis['recommendations'][col] = recommendations

    return type_analysis

def get_column_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive statistics for each column.

    Args:
        df: Input DataFrame

    Returns:
        Dict containing column statistics
    """
    stats_summary = {
        'numeric_stats': {},
        'categorical_stats': {},
        'datetime_stats': {},
        'overall_summary': {}
    }

    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_stats = df[numeric_cols].describe()

        for col in numeric_cols:
            col_data = df[col].dropna()

            stats_summary['numeric_stats'][col] = {
                'count': len(col_data),
                'mean': round(col_data.mean(), 3) if len(col_data) > 0 else None,
                'median': round(col_data.median(), 3) if len(col_data) > 0 else None,
                'std': round(col_data.std(), 3) if len(col_data) > 0 else None,
                'min': col_data.min() if len(col_data) > 0 else None,
                'max': col_data.max() if len(col_data) > 0 else None,
                'q25': round(col_data.quantile(0.25), 3) if len(col_data) > 0 else None,
                'q75': round(col_data.quantile(0.75), 3) if len(col_data) > 0 else None,
                'skewness': round(col_data.skew(), 3) if len(col_data) > 1 else None,
                'kurtosis': round(col_data.kurtosis(), 3) if len(col_data) > 1 else None
            }

    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        col_data = df[col].dropna()

        if len(col_data) > 0:
            value_counts = col_data.value_counts()

            stats_summary['categorical_stats'][col] = {
                'count': len(col_data),
                'unique': col_data.nunique(),
                'unique_ratio': round(col_data.nunique() / len(col_data), 3),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else None,
                'most_frequent_percentage': round((value_counts.iloc[0] / len(col_data)) * 100, 2) if len(value_counts) > 0 else None,
                'top_5_values': dict(value_counts.head(5))
            }

    # DateTime columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        col_data = df[col].dropna()

        if len(col_data) > 0:
            stats_summary['datetime_stats'][col] = {
                'count': len(col_data),
                'min_date': col_data.min(),
                'max_date': col_data.max(),
                'date_range_days': (col_data.max() - col_data.min()).days if len(col_data) > 1 else None
            }

    # Overall summary
    stats_summary['overall_summary'] = {
        'total_columns': len(df.columns),
        'numeric_columns': len(numeric_cols),
        'categorical_columns': len(categorical_cols),
        'datetime_columns': len(datetime_cols),
        'total_rows': len(df),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }

    return stats_summary

def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3) -> Dict[str, Any]:
    """
    Detect outliers using Z-score method.

    Args:
        df: Input DataFrame
        threshold: Z-score threshold (default 3)

    Returns:
        Dict containing outlier analysis
    """
    outlier_analysis = {
        'method': 'z-score',
        'threshold': threshold,
        'outliers_by_column': {},
        'summary': {}
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        col_data = df[col].dropna()

        if len(col_data) > 1:
            z_scores = np.abs(stats.zscore(col_data))
            outlier_mask = z_scores > threshold
            outlier_count = outlier_mask.sum()

            outlier_analysis['outliers_by_column'][col] = {
                'outlier_count': int(outlier_count),
                'outlier_percentage': round((outlier_count / len(col_data)) * 100, 2),
                'outlier_indices': col_data[outlier_mask].index.tolist(),
                'outlier_values': col_data[outlier_mask].tolist()
            }

    # Summary
    total_outliers = sum([info['outlier_count'] for info in outlier_analysis['outliers_by_column'].values()])
    outlier_analysis['summary'] = {
        'total_outliers': total_outliers,
        'columns_with_outliers': len([col for col, info in outlier_analysis['outliers_by_column'].items() if info['outlier_count'] > 0])
    }

    return outlier_analysis

def detect_outliers_iqr(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect outliers using IQR method.

    Args:
        df: Input DataFrame

    Returns:
        Dict containing outlier analysis
    """
    outlier_analysis = {
        'method': 'iqr',
        'outliers_by_column': {},
        'summary': {}
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        col_data = df[col].dropna()

        if len(col_data) > 1:
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
            outlier_count = outlier_mask.sum()

            outlier_analysis['outliers_by_column'][col] = {
                'outlier_count': int(outlier_count),
                'outlier_percentage': round((outlier_count / len(col_data)) * 100, 2),
                'lower_bound': round(lower_bound, 3),
                'upper_bound': round(upper_bound, 3),
                'Q1': round(Q1, 3),
                'Q3': round(Q3, 3),
                'IQR': round(IQR, 3),
                'outlier_indices': col_data[outlier_mask].index.tolist(),
                'outlier_values': col_data[outlier_mask].tolist()
            }

    # Summary
    total_outliers = sum([info['outlier_count'] for info in outlier_analysis['outliers_by_column'].values()])
    outlier_analysis['summary'] = {
        'total_outliers': total_outliers,
        'columns_with_outliers': len([col for col, info in outlier_analysis['outliers_by_column'].items() if info['outlier_count'] > 0])
    }

    return outlier_analysis

def comprehensive_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive data quality report.

    Args:
        df: Input DataFrame

    Returns:
        Dict containing complete data quality analysis
    """
    report = {
        'dataset_info': {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict()
        },
        'missing_values': analyze_missing_values(df),
        'duplicates': detect_duplicates(df),
        'constant_features': find_constant_features(df),
        'data_types': check_data_type_consistency(df),
        'statistics': get_column_statistics(df),
        'outliers_zscore': detect_outliers_zscore(df),
        'outliers_iqr': detect_outliers_iqr(df)
    }

    # Generate quality score (0-100)
    quality_score = calculate_quality_score(report)
    report['quality_score'] = quality_score

    return report

def calculate_quality_score(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate an overall data quality score.

    Args:
        report: Data quality report

    Returns:
        Dict containing quality score breakdown
    """
    score_breakdown = {
        'overall_score': 0,
        'missing_data_score': 0,
        'duplicate_score': 0,
        'consistency_score': 0,
        'outlier_score': 0
    }

    # Missing data score (30 points max)
    missing_pct = report['missing_values']['overall_missing_percentage']
    if missing_pct == 0:
        score_breakdown['missing_data_score'] = 30
    elif missing_pct < 5:
        score_breakdown['missing_data_score'] = 25
    elif missing_pct < 15:
        score_breakdown['missing_data_score'] = 20
    elif missing_pct < 30:
        score_breakdown['missing_data_score'] = 15
    else:
        score_breakdown['missing_data_score'] = 10

    # Duplicate score (25 points max)
    duplicate_pct = report['duplicates']['duplicate_percentage']
    if duplicate_pct == 0:
        score_breakdown['duplicate_score'] = 25
    elif duplicate_pct < 1:
        score_breakdown['duplicate_score'] = 20
    elif duplicate_pct < 5:
        score_breakdown['duplicate_score'] = 15
    else:
        score_breakdown['duplicate_score'] = 10

    # Consistency score (25 points max)
    constant_features = len(report['constant_features']['constant_features'])
    total_columns = len(report['dataset_info']['columns'])
    constant_ratio = constant_features / total_columns if total_columns > 0 else 0

    if constant_ratio == 0:
        score_breakdown['consistency_score'] = 25
    elif constant_ratio < 0.1:
        score_breakdown['consistency_score'] = 20
    elif constant_ratio < 0.2:
        score_breakdown['consistency_score'] = 15
    else:
        score_breakdown['consistency_score'] = 10

    # Outlier score (20 points max)
    total_outliers_iqr = report['outliers_iqr']['summary']['total_outliers']
    total_data_points = report['dataset_info']['shape'][0] * len(report['statistics']['numeric_stats'])
    outlier_ratio = total_outliers_iqr / total_data_points if total_data_points > 0 else 0

    if outlier_ratio < 0.01:
        score_breakdown['outlier_score'] = 20
    elif outlier_ratio < 0.05:
        score_breakdown['outlier_score'] = 15
    elif outlier_ratio < 0.1:
        score_breakdown['outlier_score'] = 10
    else:
        score_breakdown['outlier_score'] = 5

    # Calculate overall score
    score_breakdown['overall_score'] = sum([
        score_breakdown['missing_data_score'],
        score_breakdown['duplicate_score'],
        score_breakdown['consistency_score'],
        score_breakdown['outlier_score']
    ])

    # Add interpretation
    if score_breakdown['overall_score'] >= 90:
        score_breakdown['interpretation'] = "Excellent"
        score_breakdown['color'] = "green"
    elif score_breakdown['overall_score'] >= 75:
        score_breakdown['interpretation'] = "Good"
        score_breakdown['color'] = "lightgreen"
    elif score_breakdown['overall_score'] >= 60:
        score_breakdown['interpretation'] = "Fair"
        score_breakdown['color'] = "yellow"
    elif score_breakdown['overall_score'] >= 40:
        score_breakdown['interpretation'] = "Poor"
        score_breakdown['color'] = "orange"
    else:
        score_breakdown['interpretation'] = "Critical Issues"
        score_breakdown['color'] = "red"

    return score_breakdown