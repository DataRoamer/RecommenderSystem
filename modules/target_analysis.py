import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

def analyze_target_variable(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Comprehensive analysis of target variable for ML readiness.

    Args:
        df: Input DataFrame
        target_col: Name of target column

    Returns:
        Dict containing target analysis results
    """
    if target_col not in df.columns:
        return {'error': f'Target column "{target_col}" not found in dataset'}

    target_data = df[target_col].dropna()

    analysis = {
        'target_column': target_col,
        'basic_info': {},
        'classification_analysis': {},
        'regression_analysis': {},
        'recommendations': [],
        'ml_readiness': {}
    }

    # Basic information
    analysis['basic_info'] = {
        'total_rows': len(df),
        'non_null_count': len(target_data),
        'missing_count': df[target_col].isnull().sum(),
        'missing_percentage': round((df[target_col].isnull().sum() / len(df)) * 100, 2),
        'unique_values': df[target_col].nunique(),
        'data_type': str(df[target_col].dtype)
    }

    # Determine if classification or regression
    is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
    unique_count = df[target_col].nunique()

    if is_numeric and unique_count > 20:
        task_type = 'regression'
    else:
        task_type = 'classification'

    analysis['basic_info']['suggested_task_type'] = task_type

    if task_type == 'classification':
        analysis['classification_analysis'] = analyze_classification_target(df, target_col)
    else:
        analysis['regression_analysis'] = analyze_regression_target(df, target_col)

    # Generate recommendations
    analysis['recommendations'] = generate_target_recommendations(analysis, task_type)

    # ML readiness assessment
    analysis['ml_readiness'] = assess_target_ml_readiness(analysis, task_type)

    return analysis

def analyze_classification_target(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Analyze classification target variable.

    Args:
        df: Input DataFrame
        target_col: Target column name

    Returns:
        Dict containing classification analysis
    """
    target_data = df[target_col].dropna()
    value_counts = target_data.value_counts()

    analysis = {
        'class_distribution': dict(value_counts),
        'class_percentages': dict((value_counts / len(target_data) * 100).round(2)),
        'num_classes': len(value_counts),
        'minority_class_size': int(value_counts.min()),
        'majority_class_size': int(value_counts.max()),
        'balance_ratio': round(value_counts.min() / value_counts.max(), 3),
        'imbalance_severity': classify_imbalance_severity(value_counts.min() / value_counts.max()),
        'class_names': list(value_counts.index),
        'binary_classification': len(value_counts) == 2
    }

    # Calculate entropy (measure of class diversity)
    probabilities = value_counts / len(target_data)
    entropy = -sum(probabilities * np.log2(probabilities + 1e-10))
    analysis['entropy'] = round(entropy, 3)

    # Gini impurity
    gini = 1 - sum(probabilities ** 2)
    analysis['gini_impurity'] = round(gini, 3)

    return analysis

def analyze_regression_target(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Analyze regression target variable.

    Args:
        df: Input DataFrame
        target_col: Target column name

    Returns:
        Dict containing regression analysis
    """
    target_data = df[target_col].dropna()

    analysis = {
        'mean': round(target_data.mean(), 3),
        'median': round(target_data.median(), 3),
        'std': round(target_data.std(), 3),
        'min': target_data.min(),
        'max': target_data.max(),
        'range': target_data.max() - target_data.min(),
        'q25': round(target_data.quantile(0.25), 3),
        'q75': round(target_data.quantile(0.75), 3),
        'iqr': round(target_data.quantile(0.75) - target_data.quantile(0.25), 3),
        'skewness': round(target_data.skew(), 3),
        'kurtosis': round(target_data.kurtosis(), 3),
        'coefficient_of_variation': round(target_data.std() / target_data.mean(), 3) if target_data.mean() != 0 else 0
    }

    # Distribution shape analysis
    analysis['distribution_shape'] = classify_distribution_shape(analysis['skewness'], analysis['kurtosis'])
    analysis['skewness_interpretation'] = interpret_skewness(analysis['skewness'])

    # Normality tests
    if len(target_data) <= 5000:  # Shapiro-Wilk for smaller samples
        try:
            shapiro_stat, shapiro_p = stats.shapiro(target_data)
            analysis['normality_tests'] = {
                'shapiro_wilk': {
                    'statistic': round(shapiro_stat, 4),
                    'p_value': round(shapiro_p, 6),
                    'is_normal': shapiro_p > 0.05
                }
            }
        except:
            analysis['normality_tests'] = {}

    # Outlier detection
    Q1, Q3 = target_data.quantile(0.25), target_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = target_data[(target_data < lower_bound) | (target_data > upper_bound)]

    analysis['outliers'] = {
        'count': len(outliers),
        'percentage': round((len(outliers) / len(target_data)) * 100, 2),
        'lower_bound': round(lower_bound, 3),
        'upper_bound': round(upper_bound, 3)
    }

    # Transformation suggestions
    analysis['transformation_suggestions'] = suggest_regression_transformations(analysis)

    return analysis

def classify_imbalance_severity(balance_ratio: float) -> str:
    """
    Classify the severity of class imbalance.

    Args:
        balance_ratio: Ratio of minority to majority class

    Returns:
        String describing imbalance severity
    """
    if balance_ratio >= 0.8:
        return "Balanced"
    elif balance_ratio >= 0.5:
        return "Slightly Imbalanced"
    elif balance_ratio >= 0.2:
        return "Moderately Imbalanced"
    elif balance_ratio >= 0.1:
        return "Severely Imbalanced"
    else:
        return "Extremely Imbalanced"

def classify_distribution_shape(skewness: float, kurtosis: float) -> str:
    """
    Classify the overall shape of the distribution.

    Args:
        skewness: Skewness value
        kurtosis: Kurtosis value

    Returns:
        String describing distribution shape
    """
    if abs(skewness) < 0.5 and abs(kurtosis) < 1:
        return "Approximately Normal"
    elif skewness > 1:
        return "Right-skewed (Positive skew)"
    elif skewness < -1:
        return "Left-skewed (Negative skew)"
    elif kurtosis > 2:
        return "Heavy-tailed (Leptokurtic)"
    elif kurtosis < -1:
        return "Light-tailed (Platykurtic)"
    else:
        return "Non-normal distribution"

def interpret_skewness(skewness: float) -> str:
    """
    Interpret skewness value.

    Args:
        skewness: Skewness value

    Returns:
        String interpretation
    """
    if abs(skewness) < 0.5:
        return "Approximately symmetric"
    elif abs(skewness) < 1:
        return "Moderately skewed"
    else:
        return "Highly skewed"

def suggest_regression_transformations(analysis: Dict[str, Any]) -> List[str]:
    """
    Suggest transformations for regression target.

    Args:
        analysis: Regression analysis results

    Returns:
        List of transformation suggestions
    """
    suggestions = []
    skewness = analysis['skewness']

    if abs(skewness) < 0.5:
        suggestions.append("No transformation needed - target is approximately normal")
    elif skewness > 1:
        suggestions.extend([
            "Log transformation to reduce right skewness",
            "Square root transformation",
            "Box-Cox transformation to find optimal lambda"
        ])
    elif skewness < -1:
        suggestions.extend([
            "Square transformation to reduce left skewness",
            "Exponential transformation",
            "Yeo-Johnson transformation (handles negative values)"
        ])

    # Check for outliers
    if analysis['outliers']['percentage'] > 5:
        suggestions.append("Consider robust scaling due to outliers")

    # Check for constant values
    if analysis['std'] == 0:
        suggestions.append("Warning: Target has no variation - unsuitable for regression")

    return suggestions

def generate_target_recommendations(analysis: Dict[str, Any], task_type: str) -> List[str]:
    """
    Generate recommendations based on target analysis.

    Args:
        analysis: Target analysis results
        task_type: 'classification' or 'regression'

    Returns:
        List of recommendations
    """
    recommendations = []
    basic_info = analysis['basic_info']

    # Missing values recommendations
    if basic_info['missing_percentage'] > 0:
        if basic_info['missing_percentage'] < 5:
            recommendations.append(f"⚠️ Target has {basic_info['missing_percentage']:.1f}% missing values - consider imputation")
        elif basic_info['missing_percentage'] < 20:
            recommendations.append(f"⚠️ Target has {basic_info['missing_percentage']:.1f}% missing values - investigate patterns before imputation")
        else:
            recommendations.append(f"🚨 Target has {basic_info['missing_percentage']:.1f}% missing values - data collection issue?")

    if task_type == 'classification':
        class_analysis = analysis['classification_analysis']

        # Class balance recommendations
        if class_analysis['imbalance_severity'] == "Balanced":
            recommendations.append("✅ Classes are well balanced - good for standard algorithms")
        elif class_analysis['imbalance_severity'] in ["Slightly Imbalanced", "Moderately Imbalanced"]:
            recommendations.append("⚠️ Consider class weighting or stratified sampling")
        else:
            recommendations.extend([
                "🚨 Severe class imbalance detected",
                "💡 Consider: SMOTE, class weights, ensemble methods, or threshold tuning",
                "💡 Use stratified cross-validation"
            ])

        # Number of classes
        if class_analysis['num_classes'] == 2:
            recommendations.append("📊 Binary classification - use binary metrics (AUC, precision, recall)")
        elif class_analysis['num_classes'] > 10:
            recommendations.append("📊 Multi-class problem - consider hierarchical classification")

        # Class size recommendations
        if class_analysis['minority_class_size'] < 30:
            recommendations.append("⚠️ Very small minority class - consider data augmentation")

    else:  # regression
        reg_analysis = analysis['regression_analysis']

        # Distribution recommendations
        if abs(reg_analysis['skewness']) > 1:
            recommendations.append("📊 Target is highly skewed - consider transformation")

        # Outlier recommendations
        if reg_analysis['outliers']['percentage'] > 10:
            recommendations.append("⚠️ Many outliers detected - consider robust regression methods")

        # Variance recommendations
        if reg_analysis['coefficient_of_variation'] > 2:
            recommendations.append("📊 High target variance - consider log transformation")

    return recommendations

def assess_target_ml_readiness(analysis: Dict[str, Any], task_type: str) -> Dict[str, Any]:
    """
    Assess ML readiness of target variable.

    Args:
        analysis: Target analysis results
        task_type: 'classification' or 'regression'

    Returns:
        Dict containing readiness assessment
    """
    readiness = {
        'overall_score': 0,
        'missing_data_score': 0,
        'distribution_score': 0,
        'balance_score': 0,
        'issues': [],
        'strengths': []
    }

    basic_info = analysis['basic_info']

    # Missing data score (30 points)
    missing_pct = basic_info['missing_percentage']
    if missing_pct == 0:
        readiness['missing_data_score'] = 30
        readiness['strengths'].append("No missing target values")
    elif missing_pct < 5:
        readiness['missing_data_score'] = 25
    elif missing_pct < 15:
        readiness['missing_data_score'] = 20
        readiness['issues'].append("Some missing target values")
    else:
        readiness['missing_data_score'] = 10
        readiness['issues'].append("High percentage of missing target values")

    if task_type == 'classification':
        class_analysis = analysis['classification_analysis']

        # Balance score (40 points)
        balance_ratio = class_analysis['balance_ratio']
        if balance_ratio >= 0.8:
            readiness['balance_score'] = 40
            readiness['strengths'].append("Well-balanced classes")
        elif balance_ratio >= 0.5:
            readiness['balance_score'] = 35
        elif balance_ratio >= 0.2:
            readiness['balance_score'] = 25
            readiness['issues'].append("Moderate class imbalance")
        elif balance_ratio >= 0.1:
            readiness['balance_score'] = 15
            readiness['issues'].append("Severe class imbalance")
        else:
            readiness['balance_score'] = 5
            readiness['issues'].append("Extreme class imbalance")

        # Distribution score (30 points) - based on entropy and number of classes
        if class_analysis['num_classes'] >= 2:
            if class_analysis['entropy'] > 0.8:
                readiness['distribution_score'] = 30
                readiness['strengths'].append("Good class diversity")
            elif class_analysis['entropy'] > 0.5:
                readiness['distribution_score'] = 25
            else:
                readiness['distribution_score'] = 15
                readiness['issues'].append("Low class diversity")
        else:
            readiness['distribution_score'] = 0
            readiness['issues'].append("Only one class present")

    else:  # regression
        reg_analysis = analysis['regression_analysis']

        # Distribution score (40 points)
        skewness = abs(reg_analysis['skewness'])
        if skewness < 0.5:
            readiness['distribution_score'] = 40
            readiness['strengths'].append("Normal-like distribution")
        elif skewness < 1:
            readiness['distribution_score'] = 35
        elif skewness < 2:
            readiness['distribution_score'] = 25
            readiness['issues'].append("Moderately skewed distribution")
        else:
            readiness['distribution_score'] = 15
            readiness['issues'].append("Highly skewed distribution")

        # Balance score (30 points) - based on outliers and variance
        outlier_pct = reg_analysis['outliers']['percentage']
        if outlier_pct < 5:
            readiness['balance_score'] = 30
            readiness['strengths'].append("Few outliers detected")
        elif outlier_pct < 10:
            readiness['balance_score'] = 25
        elif outlier_pct < 20:
            readiness['balance_score'] = 20
            readiness['issues'].append("Some outliers present")
        else:
            readiness['balance_score'] = 10
            readiness['issues'].append("Many outliers detected")

        # Check for zero variance
        if reg_analysis['std'] == 0:
            readiness['balance_score'] = 0
            readiness['issues'].append("Target has no variation")

    # Calculate overall score
    readiness['overall_score'] = (
        readiness['missing_data_score'] +
        readiness['distribution_score'] +
        readiness['balance_score']
    )

    # Interpretation
    score = readiness['overall_score']
    if score >= 90:
        readiness['interpretation'] = "Excellent"
        readiness['color'] = "green"
    elif score >= 75:
        readiness['interpretation'] = "Good"
        readiness['color'] = "lightgreen"
    elif score >= 60:
        readiness['interpretation'] = "Fair"
        readiness['color'] = "yellow"
    elif score >= 40:
        readiness['interpretation'] = "Poor"
        readiness['color'] = "orange"
    else:
        readiness['interpretation'] = "Critical Issues"
        readiness['color'] = "red"

    return readiness

def auto_detect_target_candidates(df: pd.DataFrame) -> List[str]:
    """
    Automatically detect potential target columns.

    Args:
        df: Input DataFrame

    Returns:
        List of potential target column names
    """
    candidates = []

    # Common target column names
    target_keywords = [
        'target', 'label', 'y', 'class', 'outcome', 'result',
        'prediction', 'response', 'dependent', 'survived',
        'price', 'value', 'amount', 'score', 'rating', 'output',
        'status', 'category', 'type', 'flag', 'churn', 'fraud',
        'default', 'risk', 'success', 'failure', 'diagnosis'
    ]

    for col in df.columns:
        col_lower = col.lower()

        # Skip ID-like columns
        if any(id_term in col_lower for id_term in ['id', 'index', 'key', 'identifier']):
            continue

        # Check for keyword matches (high priority)
        keyword_match = False
        for keyword in target_keywords:
            if keyword in col_lower:
                candidates.append(col)
                keyword_match = True
                break

        if keyword_match:
            continue

        # Check data characteristics
        unique_count = df[col].nunique()
        unique_ratio = unique_count / len(df)

        # Skip columns that are likely identifiers (very high cardinality)
        if unique_ratio > 0.95:
            continue

        # Potential binary target (2 unique values)
        if unique_count == 2:
            candidates.append(col)

        # Potential classification target (3-50 unique values with reasonable ratio)
        elif 3 <= unique_count <= 50 and unique_ratio < 0.5:
            candidates.append(col)

        # Potential regression target (numeric with reasonable variation)
        elif pd.api.types.is_numeric_dtype(df[col]) and unique_count > 50:
            # Check if it's not a constant column
            if df[col].std() > 0:
                # Additional checks for numeric targets
                # Exclude if it looks like a year or timestamp
                if not (df[col].min() > 1900 and df[col].max() < 2100):
                    candidates.append(col)

    # Remove duplicates and return
    return list(set(candidates))