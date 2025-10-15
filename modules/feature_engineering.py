import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import re
from datetime import datetime

def classify_feature_types(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Classify features into different types for engineering recommendations.

    Args:
        df: Input DataFrame

    Returns:
        Dict containing feature classifications
    """
    classification = {
        'numeric_features': [],
        'categorical_features': [],
        'datetime_features': [],
        'text_features': [],
        'id_features': [],
        'binary_features': [],
        'ordinal_candidates': [],
        'high_cardinality_features': [],
        'low_cardinality_numeric': [],
        'feature_details': {}
    }

    for col in df.columns:
        col_data = df[col].dropna()
        unique_count = col_data.nunique()
        unique_ratio = unique_count / len(col_data) if len(col_data) > 0 else 0

        feature_info = {
            'data_type': str(df[col].dtype),
            'unique_count': unique_count,
            'unique_ratio': round(unique_ratio, 4),
            'missing_count': df[col].isnull().sum(),
            'missing_ratio': round(df[col].isnull().sum() / len(df), 4)
        }

        # Datetime detection
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            classification['datetime_features'].append(col)
            feature_info['feature_type'] = 'datetime'

        # Numeric features
        elif pd.api.types.is_numeric_dtype(df[col]):
            classification['numeric_features'].append(col)
            feature_info['feature_type'] = 'numeric'

            # Check for low cardinality numeric (might be categorical)
            if unique_count <= 10:
                classification['low_cardinality_numeric'].append(col)
                feature_info['note'] = 'Low cardinality - consider as categorical'

            # Check for binary numeric
            if unique_count == 2:
                classification['binary_features'].append(col)
                feature_info['note'] = 'Binary feature'

        # Categorical/Object features
        else:
            classification['categorical_features'].append(col)
            feature_info['feature_type'] = 'categorical'

            # High cardinality check
            if unique_count > 50:
                classification['high_cardinality_features'].append(col)
                feature_info['note'] = 'High cardinality'

            # ID column detection
            if unique_ratio > 0.95 or 'id' in col.lower():
                classification['id_features'].append(col)
                feature_info['note'] = 'Likely ID column'

            # Text feature detection
            if col_data.dtype == 'object' and len(col_data) > 0:
                avg_length = col_data.astype(str).str.len().mean()
                if avg_length > 50:
                    classification['text_features'].append(col)
                    feature_info['note'] = 'Text feature'

            # Binary categorical
            if unique_count == 2:
                classification['binary_features'].append(col)
                feature_info['note'] = 'Binary categorical'

            # Ordinal candidates (look for patterns)
            if is_potential_ordinal(col_data):
                classification['ordinal_candidates'].append(col)
                feature_info['note'] = 'Potential ordinal feature'

        classification['feature_details'][col] = feature_info

    return classification

def recommend_encoding_strategies(df: pd.DataFrame, feature_classification: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recommend encoding strategies for different feature types.

    Args:
        df: Input DataFrame
        feature_classification: Feature classification results

    Returns:
        Dict containing encoding recommendations
    """
    recommendations = {
        'categorical_encoding': {},
        'scaling_recommendations': {},
        'feature_transformations': {},
        'features_to_drop': [],
        'summary': {}
    }

    # Categorical encoding recommendations
    for col in feature_classification['categorical_features']:
        unique_count = df[col].nunique()

        if col in feature_classification['id_features']:
            recommendations['features_to_drop'].append({
                'feature': col,
                'reason': 'ID column - high cardinality, not predictive'
            })
        elif col in feature_classification['binary_features']:
            recommendations['categorical_encoding'][col] = {
                'method': 'Label Encoding',
                'reason': 'Binary categorical - simple label encoding sufficient',
                'complexity': 'Low'
            }
        elif unique_count <= 10:
            recommendations['categorical_encoding'][col] = {
                'method': 'One-Hot Encoding',
                'reason': 'Low cardinality - one-hot encoding creates interpretable features',
                'complexity': 'Low'
            }
        elif unique_count <= 50:
            recommendations['categorical_encoding'][col] = {
                'method': 'Target Encoding or Frequency Encoding',
                'reason': 'Medium cardinality - target encoding preserves information',
                'complexity': 'Medium'
            }
        else:
            recommendations['categorical_encoding'][col] = {
                'method': 'Hash Encoding or Target Encoding',
                'reason': 'High cardinality - hash encoding reduces dimensionality',
                'complexity': 'High'
            }

    # Ordinal encoding for ordinal candidates
    for col in feature_classification['ordinal_candidates']:
        recommendations['categorical_encoding'][col] = {
            'method': 'Ordinal Encoding',
            'reason': 'Detected ordinal pattern - preserve order information',
            'complexity': 'Low'
        }

    # Scaling recommendations for numeric features
    for col in feature_classification['numeric_features']:
        col_data = df[col].dropna()

        if len(col_data) > 1:
            # Calculate outlier percentage using IQR method
            Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = col_data[(col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)]
            outlier_pct = len(outliers) / len(col_data) * 100

            # Calculate coefficient of variation
            cv = col_data.std() / col_data.mean() if col_data.mean() != 0 else 0

            if outlier_pct > 10:
                recommendations['scaling_recommendations'][col] = {
                    'method': 'Robust Scaler',
                    'reason': f'High outlier percentage ({outlier_pct:.1f}%) - robust scaling recommended',
                    'complexity': 'Low'
                }
            elif cv > 2:
                recommendations['scaling_recommendations'][col] = {
                    'method': 'Log Transform + Standard Scaler',
                    'reason': 'High variance - log transform then scale',
                    'complexity': 'Medium'
                }
            else:
                recommendations['scaling_recommendations'][col] = {
                    'method': 'Standard Scaler',
                    'reason': 'Normal distribution - standard scaling appropriate',
                    'complexity': 'Low'
                }

    # Feature transformations
    for col in feature_classification['datetime_features']:
        recommendations['feature_transformations'][col] = {
            'transformations': [
                'Extract year, month, day, hour',
                'Extract day of week, is_weekend',
                'Calculate time differences',
                'Create cyclical features (sin/cos) for periodic patterns'
            ],
            'reason': 'Datetime features need extraction for ML algorithms',
            'complexity': 'Medium'
        }

    for col in feature_classification['text_features']:
        recommendations['feature_transformations'][col] = {
            'transformations': [
                'Text length',
                'Word count',
                'TF-IDF vectorization',
                'Sentiment analysis',
                'Named entity extraction'
            ],
            'reason': 'Text features need NLP preprocessing',
            'complexity': 'High'
        }

    # Summary statistics
    recommendations['summary'] = {
        'total_features': len(df.columns),
        'features_needing_encoding': len(recommendations['categorical_encoding']),
        'features_needing_scaling': len(recommendations['scaling_recommendations']),
        'features_to_drop': len(recommendations['features_to_drop']),
        'complex_transformations': len([k for k, v in recommendations['feature_transformations'].items()
                                       if v.get('complexity') == 'High'])
    }

    return recommendations

def suggest_feature_engineering(df: pd.DataFrame, feature_classification: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suggest new features to create from existing ones.

    Args:
        df: Input DataFrame
        feature_classification: Feature classification results

    Returns:
        Dict containing feature engineering suggestions
    """
    suggestions = {
        'interaction_features': [],
        'polynomial_features': [],
        'binning_suggestions': [],
        'aggregation_features': [],
        'domain_specific_features': [],
        'feature_selection_hints': []
    }

    numeric_features = feature_classification['numeric_features']
    categorical_features = feature_classification['categorical_features']

    # Interaction features
    if len(numeric_features) >= 2:
        # Suggest interactions between correlated numeric features
        for i, feat1 in enumerate(numeric_features[:5]):  # Limit to avoid explosion
            for feat2 in numeric_features[i+1:6]:
                if feat1 != feat2:
                    correlation = abs(df[feat1].corr(df[feat2]))
                    if 0.3 < correlation < 0.9:  # Moderate correlation
                        suggestions['interaction_features'].append({
                            'feature1': feat1,
                            'feature2': feat2,
                            'suggested_operations': ['multiply', 'add', 'ratio'],
                            'reason': f'Moderate correlation ({correlation:.3f}) suggests potential interaction'
                        })

    # Polynomial features for numeric variables
    for col in numeric_features[:3]:  # Limit to avoid too many features
        col_data = df[col].dropna()
        if len(col_data) > 0:
            skewness = abs(col_data.skew())
            if skewness < 2:  # Not too skewed
                suggestions['polynomial_features'].append({
                    'feature': col,
                    'degrees': [2, 3],
                    'reason': 'Low skewness suggests polynomial features might capture non-linear patterns'
                })

    # Binning suggestions for continuous variables
    for col in numeric_features:
        col_data = df[col].dropna()
        if col_data.nunique() > 50:  # Continuous variable
            suggestions['binning_suggestions'].append({
                'feature': col,
                'methods': ['quantile-based', 'equal-width', 'kmeans-based'],
                'suggested_bins': [5, 10],
                'reason': 'High cardinality numeric - binning might capture non-linear relationships'
            })

    # Aggregation features if we have grouping variables
    potential_groups = [col for col in categorical_features
                       if df[col].nunique() < 50 and col not in feature_classification.get('id_features', [])]

    if potential_groups and numeric_features:
        for group_col in potential_groups[:3]:  # Limit number of grouping variables
            for num_col in numeric_features[:3]:  # Limit number of numeric features
                suggestions['aggregation_features'].append({
                    'group_by': group_col,
                    'aggregate': num_col,
                    'operations': ['mean', 'std', 'count', 'median'],
                    'reason': f'Group statistics of {num_col} by {group_col} might be informative'
                })

    # Domain-specific suggestions based on column names
    suggestions['domain_specific_features'] = suggest_domain_features(df)

    # Feature selection hints
    suggestions['feature_selection_hints'] = generate_feature_selection_hints(df, feature_classification)

    return suggestions

def identify_features_to_drop(df: pd.DataFrame, feature_classification: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Identify features that should be dropped.

    Args:
        df: Input DataFrame
        feature_classification: Feature classification results

    Returns:
        List of features to drop with reasons
    """
    to_drop = []

    # ID features
    for col in feature_classification.get('id_features', []):
        to_drop.append({
            'feature': col,
            'reason': 'ID column - unique identifier, not predictive',
            'severity': 'High'
        })

    # Constant features
    for col in df.columns:
        if df[col].nunique() <= 1:
            to_drop.append({
                'feature': col,
                'reason': 'Constant feature - no variation',
                'severity': 'High'
            })

    # Features with too many missing values
    for col in df.columns:
        missing_pct = df[col].isnull().sum() / len(df) * 100
        if missing_pct > 50:
            to_drop.append({
                'feature': col,
                'reason': f'Too many missing values ({missing_pct:.1f}%)',
                'severity': 'Medium'
            })

    # High cardinality categorical features (potential noise)
    for col in feature_classification.get('high_cardinality_features', []):
        if df[col].nunique() > 0.8 * len(df):  # Almost unique
            to_drop.append({
                'feature': col,
                'reason': 'Extremely high cardinality - likely noise',
                'severity': 'Medium'
            })

    return to_drop

# Helper functions

def is_potential_ordinal(series: pd.Series) -> bool:
    """
    Detect if a categorical series might be ordinal.

    Args:
        series: Pandas series to analyze

    Returns:
        Boolean indicating if potentially ordinal
    """
    unique_values = series.dropna().unique()

    if len(unique_values) < 3:
        return False

    # Convert to strings for pattern matching
    str_values = [str(val).lower() for val in unique_values]

    # Check for common ordinal patterns
    ordinal_patterns = [
        # Size patterns
        ['small', 'medium', 'large'],
        ['s', 'm', 'l', 'xl'],
        ['low', 'medium', 'high'],
        ['poor', 'fair', 'good', 'excellent'],
        # Grade patterns
        ['a', 'b', 'c', 'd', 'f'],
        ['first', 'second', 'third'],
        # Satisfaction patterns
        ['very dissatisfied', 'dissatisfied', 'neutral', 'satisfied', 'very satisfied'],
        # Frequency patterns
        ['never', 'rarely', 'sometimes', 'often', 'always']
    ]

    for pattern in ordinal_patterns:
        if all(val in str_values for val in pattern[:len(str_values)]):
            return True

    # Check for numeric-like ordering (1st, 2nd, 3rd, etc.)
    if all(re.match(r'^\d+(st|nd|rd|th)?$', val) for val in str_values):
        return True

    return False

def suggest_domain_features(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Suggest domain-specific feature engineering based on column names.

    Args:
        df: Input DataFrame

    Returns:
        List of domain-specific suggestions
    """
    suggestions = []
    columns = [col.lower() for col in df.columns]

    # Date/time related features
    date_cols = [col for col in df.columns if any(keyword in col.lower()
                for keyword in ['date', 'time', 'created', 'updated', 'birth', 'year', 'month'])]

    if date_cols:
        suggestions.append({
            'category': 'Temporal Features',
            'features': date_cols,
            'suggestions': [
                'Calculate age from birth dates',
                'Extract seasonality features',
                'Create time-since features (days since last activity)',
                'Business day vs weekend indicators'
            ]
        })

    # Geographic features
    geo_cols = [col for col in df.columns if any(keyword in col.lower()
                for keyword in ['lat', 'lon', 'zip', 'city', 'state', 'country', 'address'])]

    if geo_cols:
        suggestions.append({
            'category': 'Geographic Features',
            'features': geo_cols,
            'suggestions': [
                'Calculate distances from city centers',
                'Create region/zone groupings',
                'Population density from zip codes',
                'Climate data from coordinates'
            ]
        })

    # Financial features
    financial_cols = [col for col in df.columns if any(keyword in col.lower()
                     for keyword in ['price', 'cost', 'salary', 'income', 'revenue', 'amount', 'fee'])]

    if financial_cols:
        suggestions.append({
            'category': 'Financial Features',
            'features': financial_cols,
            'suggestions': [
                'Calculate ratios between financial metrics',
                'Create price per unit features',
                'Log transform for skewed distributions',
                'Inflation adjustment for historical data'
            ]
        })

    # Text/Name features
    text_cols = [col for col in df.columns if any(keyword in col.lower()
                for keyword in ['name', 'title', 'description', 'comment', 'text', 'review'])]

    if text_cols:
        suggestions.append({
            'category': 'Text Features',
            'features': text_cols,
            'suggestions': [
                'Extract name length/word count',
                'Sentiment analysis for reviews',
                'Extract keywords/topics',
                'Language detection for multilingual text'
            ]
        })

    return suggestions

def generate_feature_selection_hints(df: pd.DataFrame, feature_classification: Dict[str, Any]) -> List[str]:
    """
    Generate hints for feature selection.

    Args:
        df: Input DataFrame
        feature_classification: Feature classification results

    Returns:
        List of feature selection hints
    """
    hints = []

    total_features = len(df.columns)

    if total_features > 100:
        hints.append("📊 High-dimensional dataset - consider dimensionality reduction (PCA, feature selection)")

    if len(feature_classification.get('high_cardinality_features', [])) > 0:
        hints.append("⚠️ High cardinality features present - consider embedding or target encoding")

    if len(feature_classification.get('id_features', [])) > 0:
        hints.append("🗑️ ID columns detected - remove before modeling")

    numeric_ratio = len(feature_classification.get('numeric_features', [])) / total_features
    if numeric_ratio < 0.3:
        hints.append("📋 Few numeric features - focus on categorical encoding strategies")

    if len(feature_classification.get('text_features', [])) > 0:
        hints.append("📝 Text features detected - consider NLP preprocessing pipeline")

    return hints

def comprehensive_feature_engineering_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive feature engineering report.

    Args:
        df: Input DataFrame

    Returns:
        Dict containing complete feature engineering analysis
    """
    # Classify features
    feature_classification = classify_feature_types(df)

    # Get encoding recommendations
    encoding_recommendations = recommend_encoding_strategies(df, feature_classification)

    # Get feature engineering suggestions
    engineering_suggestions = suggest_feature_engineering(df, feature_classification)

    # Identify features to drop
    features_to_drop = identify_features_to_drop(df, feature_classification)

    report = {
        'feature_classification': feature_classification,
        'encoding_recommendations': encoding_recommendations,
        'engineering_suggestions': engineering_suggestions,
        'features_to_drop': features_to_drop,
        'summary': {
            'total_features': len(df.columns),
            'feature_types': {
                'numeric': len(feature_classification['numeric_features']),
                'categorical': len(feature_classification['categorical_features']),
                'datetime': len(feature_classification['datetime_features']),
                'text': len(feature_classification['text_features']),
                'id': len(feature_classification['id_features']),
                'binary': len(feature_classification['binary_features'])
            },
            'recommendations_count': {
                'encoding': len(encoding_recommendations['categorical_encoding']),
                'scaling': len(encoding_recommendations['scaling_recommendations']),
                'transformations': len(encoding_recommendations['feature_transformations']),
                'to_drop': len(features_to_drop),
                'new_features': len(engineering_suggestions['interaction_features']) +
                              len(engineering_suggestions['polynomial_features'])
            }
        }
    }

    return report