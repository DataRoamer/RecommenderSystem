import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

def calculate_comprehensive_readiness_score(
    quality_report: Dict[str, Any],
    target_analysis: Optional[Dict[str, Any]] = None,
    feature_engineering_report: Optional[Dict[str, Any]] = None,
    df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive model readiness score.

    Args:
        quality_report: Data quality analysis results
        target_analysis: Target variable analysis results
        feature_engineering_report: Feature engineering analysis results
        df: Original DataFrame

    Returns:
        Dict containing comprehensive readiness assessment
    """
    readiness = {
        'overall_score': 0,
        'category_scores': {
            'data_quality': 0,
            'target_quality': 0,
            'feature_quality': 0,
            'data_leakage': 0,
            'engineering_readiness': 0
        },
        'detailed_assessment': {},
        'priority_actions': [],
        'strengths': [],
        'critical_issues': [],
        'interpretation': '',
        'color': '',
        'next_steps': []
    }

    # Data Quality Score (25 points)
    data_quality_score = calculate_data_quality_score(quality_report)
    readiness['category_scores']['data_quality'] = data_quality_score['score']
    readiness['detailed_assessment']['data_quality'] = data_quality_score

    # Target Quality Score (25 points)
    if target_analysis:
        target_quality_score = calculate_target_quality_score(target_analysis)
        readiness['category_scores']['target_quality'] = target_quality_score['score']
        readiness['detailed_assessment']['target_quality'] = target_quality_score
    else:
        readiness['category_scores']['target_quality'] = 15  # Neutral score if no target
        readiness['detailed_assessment']['target_quality'] = {
            'score': 15,
            'issues': ['No target variable specified'],
            'recommendations': ['Select and analyze target variable']
        }

    # Feature Quality Score (25 points)
    feature_quality_score = calculate_feature_quality_score(quality_report, feature_engineering_report)
    readiness['category_scores']['feature_quality'] = feature_quality_score['score']
    readiness['detailed_assessment']['feature_quality'] = feature_quality_score

    # Data Leakage Score (15 points)
    leakage_score = calculate_leakage_score(df, quality_report, target_analysis.get('target_column') if target_analysis else None)
    readiness['category_scores']['data_leakage'] = leakage_score['score']
    readiness['detailed_assessment']['data_leakage'] = leakage_score

    # Engineering Readiness Score (10 points)
    if feature_engineering_report:
        engineering_score = calculate_engineering_readiness_score(feature_engineering_report)
        readiness['category_scores']['engineering_readiness'] = engineering_score['score']
        readiness['detailed_assessment']['engineering_readiness'] = engineering_score
    else:
        readiness['category_scores']['engineering_readiness'] = 5  # Neutral score
        readiness['detailed_assessment']['engineering_readiness'] = {
            'score': 5,
            'issues': ['Feature engineering not analyzed'],
            'recommendations': ['Analyze feature engineering opportunities']
        }

    # Calculate overall score
    readiness['overall_score'] = sum(readiness['category_scores'].values())

    # Generate interpretation
    readiness.update(interpret_readiness_score(readiness['overall_score']))

    # Compile issues and recommendations
    readiness = compile_issues_and_recommendations(readiness)

    return readiness

def calculate_data_quality_score(quality_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate data quality component score.

    Args:
        quality_report: Data quality analysis results

    Returns:
        Dict containing data quality assessment
    """
    assessment = {
        'score': 0,
        'max_score': 25,
        'components': {},
        'issues': [],
        'recommendations': []
    }

    # Missing values (8 points)
    missing_pct = quality_report['missing_values']['overall_missing_percentage']
    if missing_pct == 0:
        missing_score = 8
    elif missing_pct < 5:
        missing_score = 7
        assessment['recommendations'].append("Address missing values in key features")
    elif missing_pct < 15:
        missing_score = 5
        assessment['issues'].append(f"Moderate missing data ({missing_pct:.1f}%)")
    else:
        missing_score = 2
        assessment['issues'].append(f"High missing data ({missing_pct:.1f}%)")

    assessment['components']['missing_values'] = missing_score

    # Duplicates (6 points)
    duplicate_pct = quality_report['duplicates']['duplicate_percentage']
    if duplicate_pct == 0:
        duplicate_score = 6
    elif duplicate_pct < 1:
        duplicate_score = 5
        assessment['recommendations'].append("Remove duplicate rows")
    elif duplicate_pct < 5:
        duplicate_score = 3
        assessment['issues'].append(f"Some duplicates present ({duplicate_pct:.1f}%)")
    else:
        duplicate_score = 1
        assessment['issues'].append(f"Many duplicates ({duplicate_pct:.1f}%)")

    assessment['components']['duplicates'] = duplicate_score

    # Constant features (6 points)
    constant_features = len(quality_report['constant_features']['constant_features'])
    total_features = len(quality_report['dataset_info']['columns'])
    constant_ratio = constant_features / total_features if total_features > 0 else 0

    if constant_ratio == 0:
        constant_score = 6
    elif constant_ratio < 0.1:
        constant_score = 5
        assessment['recommendations'].append("Remove constant features")
    elif constant_ratio < 0.2:
        constant_score = 3
        assessment['issues'].append("Some constant features detected")
    else:
        constant_score = 1
        assessment['issues'].append("Many constant features")

    assessment['components']['constant_features'] = constant_score

    # Outliers (5 points)
    outlier_count = quality_report['outliers_iqr']['summary']['total_outliers']
    total_data_points = quality_report['dataset_info']['shape'][0] * len(quality_report['statistics']['numeric_stats'])
    outlier_ratio = outlier_count / total_data_points if total_data_points > 0 else 0

    if outlier_ratio < 0.01:
        outlier_score = 5
    elif outlier_ratio < 0.05:
        outlier_score = 4
        assessment['recommendations'].append("Review outliers in key features")
    elif outlier_ratio < 0.1:
        outlier_score = 2
        assessment['issues'].append("Some outliers present")
    else:
        outlier_score = 1
        assessment['issues'].append("Many outliers detected")

    assessment['components']['outliers'] = outlier_score

    assessment['score'] = sum(assessment['components'].values())

    return assessment

def calculate_target_quality_score(target_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate target quality component score.

    Args:
        target_analysis: Target analysis results

    Returns:
        Dict containing target quality assessment
    """
    assessment = {
        'score': 0,
        'max_score': 25,
        'components': {},
        'issues': [],
        'recommendations': []
    }

    basic_info = target_analysis['basic_info']
    task_type = basic_info['suggested_task_type']

    # Missing target values (10 points)
    missing_pct = basic_info['missing_percentage']
    if missing_pct == 0:
        missing_score = 10
    elif missing_pct < 5:
        missing_score = 8
        assessment['recommendations'].append("Handle missing target values")
    elif missing_pct < 15:
        missing_score = 5
        assessment['issues'].append(f"Missing target values ({missing_pct:.1f}%)")
    else:
        missing_score = 2
        assessment['issues'].append(f"High missing target values ({missing_pct:.1f}%)")

    assessment['components']['target_missing'] = missing_score

    # Task-specific scoring
    if task_type == 'classification':
        class_analysis = target_analysis['classification_analysis']

        # Class balance (15 points)
        balance_ratio = class_analysis['balance_ratio']
        imbalance_severity = class_analysis['imbalance_severity']

        if imbalance_severity == "Balanced":
            balance_score = 15
        elif imbalance_severity == "Slightly Imbalanced":
            balance_score = 12
            assessment['recommendations'].append("Consider class weighting")
        elif imbalance_severity == "Moderately Imbalanced":
            balance_score = 8
            assessment['issues'].append("Moderate class imbalance")
        elif imbalance_severity == "Severely Imbalanced":
            balance_score = 4
            assessment['issues'].append("Severe class imbalance")
        else:
            balance_score = 2
            assessment['issues'].append("Extreme class imbalance")

        assessment['components']['class_balance'] = balance_score

    else:  # regression
        reg_analysis = target_analysis['regression_analysis']

        # Distribution quality (15 points)
        skewness = abs(reg_analysis['skewness'])
        outlier_pct = reg_analysis['outliers']['percentage']

        if skewness < 0.5 and outlier_pct < 5:
            distribution_score = 15
        elif skewness < 1 and outlier_pct < 10:
            distribution_score = 12
            assessment['recommendations'].append("Consider target transformation")
        elif skewness < 2 and outlier_pct < 20:
            distribution_score = 8
            assessment['issues'].append("Skewed target distribution")
        else:
            distribution_score = 4
            assessment['issues'].append("Highly skewed target with outliers")

        assessment['components']['target_distribution'] = distribution_score

        # Check for zero variance
        if reg_analysis['std'] == 0:
            assessment['components']['target_distribution'] = 0
            assessment['issues'].append("Target has no variation")

    assessment['score'] = sum(assessment['components'].values())

    return assessment

def calculate_feature_quality_score(
    quality_report: Dict[str, Any],
    feature_engineering_report: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calculate feature quality component score.

    Args:
        quality_report: Data quality analysis results
        feature_engineering_report: Feature engineering analysis results

    Returns:
        Dict containing feature quality assessment
    """
    assessment = {
        'score': 0,
        'max_score': 25,
        'components': {},
        'issues': [],
        'recommendations': []
    }

    # Multicollinearity (10 points)
    if 'correlations' in quality_report:
        high_corr_pairs = len(quality_report['correlations'].get('high_correlations', []))
        total_features = len(quality_report['dataset_info']['columns'])

        if high_corr_pairs == 0:
            correlation_score = 10
        elif high_corr_pairs < total_features * 0.1:
            correlation_score = 8
            assessment['recommendations'].append("Consider removing highly correlated features")
        elif high_corr_pairs < total_features * 0.2:
            correlation_score = 5
            assessment['issues'].append("Some multicollinearity present")
        else:
            correlation_score = 2
            assessment['issues'].append("High multicollinearity")

        assessment['components']['multicollinearity'] = correlation_score
    else:
        assessment['components']['multicollinearity'] = 5  # Neutral score

    # Feature variance and information content (10 points)
    constant_features = len(quality_report['constant_features']['constant_features'])
    near_constant_features = len(quality_report['constant_features']['near_constant_features'])
    total_features = len(quality_report['dataset_info']['columns'])

    low_info_ratio = (constant_features + near_constant_features) / total_features

    if low_info_ratio == 0:
        variance_score = 10
    elif low_info_ratio < 0.1:
        variance_score = 8
        assessment['recommendations'].append("Remove low-variance features")
    elif low_info_ratio < 0.2:
        variance_score = 5
        assessment['issues'].append("Some low-information features")
    else:
        variance_score = 2
        assessment['issues'].append("Many low-information features")

    assessment['components']['feature_information'] = variance_score

    # Feature engineering readiness (5 points)
    if feature_engineering_report:
        id_features = len(feature_engineering_report['features_to_drop'])
        encoding_needed = len(feature_engineering_report['encoding_recommendations']['categorical_encoding'])
        total_features = feature_engineering_report['summary']['total_features']

        # Score based on how ready features are for ML
        if id_features == 0 and encoding_needed < total_features * 0.3:
            engineering_score = 5
        elif id_features < total_features * 0.1 and encoding_needed < total_features * 0.5:
            engineering_score = 4
            assessment['recommendations'].append("Apply recommended feature engineering")
        else:
            engineering_score = 2
            assessment['issues'].append("Significant feature engineering required")

        assessment['components']['engineering_readiness'] = engineering_score
    else:
        assessment['components']['engineering_readiness'] = 3  # Neutral score

    assessment['score'] = sum(assessment['components'].values())

    return assessment

def calculate_leakage_score(df: Optional[pd.DataFrame], quality_report: Dict[str, Any], target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate data leakage risk score using advanced leakage detection.

    Args:
        df: Original DataFrame
        quality_report: Data quality analysis results
        target_col: Target column name

    Returns:
        Dict containing leakage assessment
    """
    assessment = {
        'score': 15,  # Default full score if no leakage detected
        'max_score': 15,
        'components': {},
        'issues': [],
        'recommendations': []
    }

    if df is not None:
        try:
            # Use advanced leakage detection if available
            from modules.leakage_detection import comprehensive_leakage_detection

            # Run comprehensive leakage analysis
            leakage_results = comprehensive_leakage_detection(
                df=df,
                target_col=target_col,
                date_cols=None,  # Auto-detect
                id_cols=None     # Auto-detect
            )

            # Convert risk score (0-100) to component score (0-15)
            risk_score = leakage_results.get('risk_score', 0)
            component_score = max(0, 15 - int((risk_score / 100) * 15))
            assessment['score'] = component_score

            # Extract key information
            assessment['components'] = {
                'overall_risk_level': leakage_results.get('overall_leakage_risk', 'Low'),
                'risk_score': risk_score,
                'suspicious_features_count': len(leakage_results.get('suspicious_features', [])),
                'total_issues': len(leakage_results.get('detailed_findings', []))
            }

            # Include main issues (limit to top 5 for brevity)
            detailed_findings = leakage_results.get('detailed_findings', [])
            assessment['issues'] = detailed_findings[:5]
            if len(detailed_findings) > 5:
                assessment['issues'].append(f"... and {len(detailed_findings) - 5} more issues")

            # Include key recommendations (limit to top 3)
            recommendations = leakage_results.get('recommendations', [])
            assessment['recommendations'] = recommendations[:3]
            if len(recommendations) > 3:
                assessment['recommendations'].append("See detailed leakage analysis for more recommendations")

        except ImportError:
            # Fallback to basic checks if advanced module not available
            assessment['issues'].append("Advanced leakage detection not available - using basic checks")
            basic_issues = []

            # Basic perfect correlation check
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                try:
                    corr_matrix = df[numeric_cols].corr()
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if not pd.isna(corr_val) and abs(corr_val) > 0.99:
                                basic_issues.append(f"Perfect correlation: {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}")
                                assessment['score'] -= 3
                except:
                    pass

            # Basic suspicious names check
            suspicious_keywords = ['target', 'label', 'result', 'outcome', 'prediction']
            for col in df.columns:
                if any(keyword in col.lower() for keyword in suspicious_keywords):
                    basic_issues.append(f"Suspicious column name: {col}")
                    assessment['score'] -= 2

            assessment['issues'].extend(basic_issues)
            assessment['score'] = max(0, assessment['score'])

        except Exception as e:
            # Handle any other errors gracefully
            assessment['issues'].append(f"Error in leakage detection: {str(e)}")
            assessment['score'] = 10  # Moderate score when uncertain

    return assessment

def calculate_engineering_readiness_score(feature_engineering_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate feature engineering readiness score.

    Args:
        feature_engineering_report: Feature engineering analysis results

    Returns:
        Dict containing engineering readiness assessment
    """
    assessment = {
        'score': 0,
        'max_score': 10,
        'components': {},
        'issues': [],
        'recommendations': []
    }

    summary = feature_engineering_report['summary']
    total_features = summary['total_features']

    # Categorical encoding readiness (4 points)
    encoding_needed = summary['recommendations_count']['encoding']
    if encoding_needed == 0:
        encoding_score = 4
    elif encoding_needed < total_features * 0.3:
        encoding_score = 3
        assessment['recommendations'].append("Apply categorical encoding")
    elif encoding_needed < total_features * 0.5:
        encoding_score = 2
        assessment['issues'].append("Many features need encoding")
    else:
        encoding_score = 1
        assessment['issues'].append("Most features need encoding")

    assessment['components']['encoding_readiness'] = encoding_score

    # Scaling readiness (3 points)
    scaling_needed = summary['recommendations_count']['scaling']
    numeric_features = summary['feature_types']['numeric']

    if scaling_needed == 0 or numeric_features == 0:
        scaling_score = 3
    elif scaling_needed < numeric_features * 0.5:
        scaling_score = 2
        assessment['recommendations'].append("Apply feature scaling")
    else:
        scaling_score = 1
        assessment['issues'].append("Most numeric features need scaling")

    assessment['components']['scaling_readiness'] = scaling_score

    # Feature selection readiness (3 points)
    to_drop = summary['recommendations_count']['to_drop']
    if to_drop == 0:
        selection_score = 3
    elif to_drop < total_features * 0.1:
        selection_score = 2
        assessment['recommendations'].append("Remove recommended features")
    else:
        selection_score = 1
        assessment['issues'].append("Many features should be removed")

    assessment['components']['feature_selection'] = selection_score

    assessment['score'] = sum(assessment['components'].values())

    return assessment

def interpret_readiness_score(overall_score: int) -> Dict[str, str]:
    """
    Interpret the overall readiness score.

    Args:
        overall_score: Overall readiness score

    Returns:
        Dict containing interpretation and color
    """
    if overall_score >= 90:
        return {
            'interpretation': 'Excellent - Ready for Modeling',
            'color': 'green',
            'description': 'Dataset is in excellent condition for machine learning'
        }
    elif overall_score >= 75:
        return {
            'interpretation': 'Good - Minor Issues',
            'color': 'lightgreen',
            'description': 'Dataset is in good condition with minor issues to address'
        }
    elif overall_score >= 60:
        return {
            'interpretation': 'Fair - Needs Improvement',
            'color': 'yellow',
            'description': 'Dataset needs some work before optimal modeling'
        }
    elif overall_score >= 40:
        return {
            'interpretation': 'Poor - Significant Issues',
            'color': 'orange',
            'description': 'Dataset has significant issues that must be addressed'
        }
    else:
        return {
            'interpretation': 'Critical - Not Ready',
            'color': 'red',
            'description': 'Dataset has critical issues and is not ready for modeling'
        }

def compile_issues_and_recommendations(readiness: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compile all issues and recommendations into priority lists.

    Args:
        readiness: Readiness assessment results

    Returns:
        Updated readiness assessment with compiled issues
    """
    all_issues = []
    all_recommendations = []
    all_strengths = []

    # Compile from all categories
    for category, assessment in readiness['detailed_assessment'].items():
        for issue in assessment.get('issues', []):
            all_issues.append(f"{category.title()}: {issue}")

        for rec in assessment.get('recommendations', []):
            all_recommendations.append(f"{category.title()}: {rec}")

        # Identify strengths (high scoring components)
        if assessment['score'] >= assessment['max_score'] * 0.8:
            all_strengths.append(f"{category.title()}: High quality")

    # Prioritize critical issues
    critical_keywords = ['high', 'many', 'severe', 'extreme', 'critical', 'no variation']
    critical_issues = [issue for issue in all_issues
                      if any(keyword in issue.lower() for keyword in critical_keywords)]

    # Top priority actions (limit to 5)
    priority_actions = all_recommendations[:5]

    readiness['critical_issues'] = critical_issues
    readiness['priority_actions'] = priority_actions
    readiness['strengths'] = all_strengths

    # Generate next steps
    readiness['next_steps'] = generate_next_steps(readiness)

    return readiness

def generate_next_steps(readiness: Dict[str, Any]) -> List[str]:
    """
    Generate actionable next steps based on readiness assessment.

    Args:
        readiness: Readiness assessment results

    Returns:
        List of actionable next steps
    """
    next_steps = []
    overall_score = readiness['overall_score']

    if overall_score >= 75:
        next_steps = [
            "🚀 Proceed with model development",
            "🔧 Address minor quality issues",
            "📊 Start with baseline model",
            "🎯 Set up model evaluation framework"
        ]
    elif overall_score >= 60:
        next_steps = [
            "🔧 Address identified data quality issues",
            "🎯 Complete feature engineering tasks",
            "📊 Re-evaluate after improvements",
            "🚀 Proceed with simple models first"
        ]
    elif overall_score >= 40:
        next_steps = [
            "🚨 Address critical data issues first",
            "🧹 Complete data cleaning pipeline",
            "🔧 Implement feature engineering",
            "📊 Re-run readiness assessment"
        ]
    else:
        next_steps = [
            "🛑 Stop - critical issues must be resolved",
            "📋 Review data collection process",
            "🧹 Implement comprehensive data cleaning",
            "🔍 Consider data quality at source"
        ]

    return next_steps