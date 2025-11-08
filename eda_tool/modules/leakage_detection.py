#!/usr/bin/env python3
"""
Advanced Data Leakage Detection Module

This module provides comprehensive algorithms to detect various types of data leakage
that could lead to overly optimistic model performance in machine learning projects.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

warnings.filterwarnings('ignore')

def comprehensive_leakage_detection(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    date_cols: Optional[List[str]] = None,
    id_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Perform comprehensive data leakage detection analysis.

    Args:
        df: Input DataFrame
        target_col: Target variable column name
        date_cols: List of date/time columns
        id_cols: List of identifier columns

    Returns:
        Dict containing comprehensive leakage analysis
    """
    results = {
        'overall_leakage_risk': 'Low',
        'risk_score': 0,  # 0-100 scale (higher = more risk)
        'leakage_types': {
            'target_leakage': {},
            'temporal_leakage': {},
            'data_duplication': {},
            'feature_leakage': {},
            'identifier_leakage': {},
            'statistical_leakage': {}
        },
        'suspicious_features': [],
        'recommendations': [],
        'detailed_findings': []
    }

    # Auto-detect columns if not provided
    if date_cols is None:
        date_cols = auto_detect_date_columns(df)
    if id_cols is None:
        id_cols = auto_detect_id_columns(df)

    risk_points = 0
    max_risk_points = 600  # Total possible risk points

    # 1. Target Leakage Detection (150 points max)
    if target_col and target_col in df.columns:
        target_analysis = detect_target_leakage(df, target_col)
        results['leakage_types']['target_leakage'] = target_analysis
        risk_points += target_analysis['risk_points']

        if target_analysis['issues']:
            results['suspicious_features'].extend(target_analysis['suspicious_columns'])
            results['detailed_findings'].extend(target_analysis['issues'])

    # 2. Temporal Leakage Detection (100 points max)
    temporal_analysis = detect_temporal_leakage(df, target_col, date_cols)
    results['leakage_types']['temporal_leakage'] = temporal_analysis
    risk_points += temporal_analysis['risk_points']

    if temporal_analysis['issues']:
        results['suspicious_features'].extend(temporal_analysis['suspicious_columns'])
        results['detailed_findings'].extend(temporal_analysis['issues'])

    # 3. Data Duplication Leakage (100 points max)
    duplication_analysis = detect_data_duplication_leakage(df)
    results['leakage_types']['data_duplication'] = duplication_analysis
    risk_points += duplication_analysis['risk_points']

    if duplication_analysis['issues']:
        results['detailed_findings'].extend(duplication_analysis['issues'])

    # 4. Feature Leakage Detection (150 points max)
    feature_analysis = detect_feature_leakage(df, target_col)
    results['leakage_types']['feature_leakage'] = feature_analysis
    risk_points += feature_analysis['risk_points']

    if feature_analysis['issues']:
        results['suspicious_features'].extend(feature_analysis['suspicious_columns'])
        results['detailed_findings'].extend(feature_analysis['issues'])

    # 5. Identifier Leakage Detection (50 points max)
    id_analysis = detect_identifier_leakage(df, id_cols, target_col)
    results['leakage_types']['identifier_leakage'] = id_analysis
    risk_points += id_analysis['risk_points']

    if id_analysis['issues']:
        results['suspicious_features'].extend(id_analysis['suspicious_columns'])
        results['detailed_findings'].extend(id_analysis['issues'])

    # 6. Statistical Leakage Detection (50 points max)
    statistical_analysis = detect_statistical_leakage(df, target_col)
    results['leakage_types']['statistical_leakage'] = statistical_analysis
    risk_points += statistical_analysis['risk_points']

    if statistical_analysis['issues']:
        results['suspicious_features'].extend(statistical_analysis['suspicious_columns'])
        results['detailed_findings'].extend(statistical_analysis['issues'])

    # Calculate overall risk score and level
    results['risk_score'] = min(100, (risk_points / max_risk_points) * 100)
    results['overall_leakage_risk'] = classify_risk_level(results['risk_score'])

    # Generate recommendations
    results['recommendations'] = generate_leakage_recommendations(results)

    # Remove duplicates from suspicious features
    results['suspicious_features'] = list(set(results['suspicious_features']))

    return results

def detect_target_leakage(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Detect features that are perfect predictors of the target (target leakage).
    """
    analysis = {
        'risk_points': 0,
        'max_risk_points': 150,
        'issues': [],
        'suspicious_columns': [],
        'perfect_predictors': [],
        'near_perfect_predictors': [],
        'suspicious_names': []
    }

    if target_col not in df.columns:
        return analysis

    target_data = df[target_col].dropna()
    feature_cols = [col for col in df.columns if col != target_col]

    # Check for perfect predictors (deterministic relationship)
    for col in feature_cols:
        try:
            # Skip non-informative columns
            if df[col].nunique() <= 1:
                continue

            # Align indices for comparison
            common_idx = df[col].dropna().index.intersection(target_data.index)
            if len(common_idx) < 10:  # Need sufficient data
                continue

            col_subset = df.loc[common_idx, col]
            target_subset = df.loc[common_idx, target_col]

            # Check for perfect prediction ability
            grouped = pd.DataFrame({'feature': col_subset, 'target': target_subset}).groupby('feature')['target']

            # For each unique feature value, check if target is constant
            perfect_prediction = True
            for name, group in grouped:
                if group.nunique() > 1:
                    perfect_prediction = False
                    break

            if perfect_prediction and col_subset.nunique() > 1:
                analysis['perfect_predictors'].append(col)
                analysis['suspicious_columns'].append(col)
                analysis['issues'].append(f"🚨 Perfect predictor detected: '{col}' - each value maps to single target value")
                analysis['risk_points'] += 50

            # Check for near-perfect predictors using mutual information
            elif len(common_idx) > 20:
                try:
                    if pd.api.types.is_numeric_dtype(df[target_col]):
                        # Regression target
                        if pd.api.types.is_numeric_dtype(df[col]):
                            mi_score = mutual_info_regression(col_subset.values.reshape(-1, 1), target_subset)[0]
                        else:
                            # Encode categorical for MI calculation
                            le = LabelEncoder()
                            encoded_col = le.fit_transform(col_subset.astype(str))
                            mi_score = mutual_info_regression(encoded_col.reshape(-1, 1), target_subset)[0]
                    else:
                        # Classification target
                        if pd.api.types.is_numeric_dtype(df[col]):
                            mi_score = mutual_info_classif(col_subset.values.reshape(-1, 1), target_subset)[0]
                        else:
                            le = LabelEncoder()
                            encoded_col = le.fit_transform(col_subset.astype(str))
                            mi_score = mutual_info_classif(encoded_col.reshape(-1, 1), target_subset)[0]

                    # High mutual information suggests potential leakage
                    if mi_score > 0.8:  # Threshold for suspicion
                        analysis['near_perfect_predictors'].append({'column': col, 'mi_score': mi_score})
                        analysis['suspicious_columns'].append(col)
                        analysis['issues'].append(f"⚠️ Near-perfect predictor: '{col}' (MI score: {mi_score:.3f})")
                        analysis['risk_points'] += 30

                except Exception:
                    pass  # Skip if MI calculation fails

        except Exception:
            continue  # Skip problematic columns

    # Check for suspicious column names that might indicate target leakage
    suspicious_keywords = [
        'target', 'label', 'prediction', 'predicted', 'result', 'outcome',
        'response', 'dependent', 'y', 'class', 'category', 'decision',
        'approved', 'rejected', 'success', 'failure', 'winner', 'loser'
    ]

    target_lower = target_col.lower()
    for col in feature_cols:
        col_lower = col.lower()

        # Check for variations of target name
        if target_lower in col_lower or col_lower in target_lower:
            if col != target_col:  # Exclude exact target column
                analysis['suspicious_names'].append(col)
                analysis['suspicious_columns'].append(col)
                analysis['issues'].append(f"🔍 Suspicious name similarity: '{col}' similar to target '{target_col}'")
                analysis['risk_points'] += 20

        # Check for suspicious keywords
        for keyword in suspicious_keywords:
            if keyword in col_lower and keyword not in target_lower:
                analysis['suspicious_names'].append(col)
                analysis['suspicious_columns'].append(col)
                analysis['issues'].append(f"🔍 Suspicious column name: '{col}' contains keyword '{keyword}'")
                analysis['risk_points'] += 15
                break

    return analysis

def detect_temporal_leakage(df: pd.DataFrame, target_col: Optional[str], date_cols: List[str]) -> Dict[str, Any]:
    """
    Detect temporal leakage where future information is used to predict past events.
    """
    analysis = {
        'risk_points': 0,
        'max_risk_points': 100,
        'issues': [],
        'suspicious_columns': [],
        'future_info_columns': [],
        'timestamp_analysis': {}
    }

    if not date_cols:
        analysis['issues'].append("ℹ️ No date columns detected - temporal leakage analysis limited")
        return analysis

    # Analyze each date column
    for date_col in date_cols:
        if date_col not in df.columns:
            continue

        try:
            # Convert to datetime if not already
            date_series = pd.to_datetime(df[date_col], errors='coerce')

            # Check for future dates
            current_time = datetime.now()
            future_dates = date_series > current_time

            if future_dates.any():
                future_count = future_dates.sum()
                analysis['future_info_columns'].append({
                    'column': date_col,
                    'future_dates_count': int(future_count),
                    'percentage': round((future_count / len(df)) * 100, 2)
                })
                analysis['suspicious_columns'].append(date_col)
                analysis['issues'].append(f"🔮 Future dates detected in '{date_col}': {future_count} rows ({future_count/len(df)*100:.1f}%)")
                analysis['risk_points'] += 30

            # Check for suspicious date patterns
            if target_col and target_col in df.columns:
                # Look for columns with dates that are suspiciously close to the target timeframe
                analysis['timestamp_analysis'][date_col] = {
                    'min_date': date_series.min(),
                    'max_date': date_series.max(),
                    'date_range_days': (date_series.max() - date_series.min()).days if date_series.notna().any() else 0
                }

        except Exception:
            analysis['issues'].append(f"⚠️ Could not parse date column: '{date_col}'")

    # Look for columns that might contain temporal information
    temporal_keywords = ['created', 'updated', 'modified', 'processed', 'completed', 'finished', 'ended']

    for col in df.columns:
        if col == target_col or col in date_cols:
            continue

        col_lower = col.lower()
        for keyword in temporal_keywords:
            if keyword in col_lower:
                analysis['suspicious_columns'].append(col)
                analysis['issues'].append(f"🕐 Potential temporal feature: '{col}' - review for temporal leakage")
                analysis['risk_points'] += 10
                break

    return analysis

def detect_data_duplication_leakage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect data duplication that could lead to leakage between train/test sets.
    """
    analysis = {
        'risk_points': 0,
        'max_risk_points': 100,
        'issues': [],
        'duplicate_analysis': {
            'exact_duplicates': 0,
            'near_duplicates': 0,
            'duplicate_percentage': 0
        }
    }

    # Check for exact duplicates
    exact_duplicates = df.duplicated().sum()
    duplicate_percentage = (exact_duplicates / len(df)) * 100

    analysis['duplicate_analysis']['exact_duplicates'] = int(exact_duplicates)
    analysis['duplicate_analysis']['duplicate_percentage'] = round(duplicate_percentage, 2)

    if exact_duplicates > 0:
        analysis['issues'].append(f"📋 Exact duplicates found: {exact_duplicates} rows ({duplicate_percentage:.1f}%)")
        if duplicate_percentage > 10:
            analysis['risk_points'] += 40
        elif duplicate_percentage > 5:
            analysis['risk_points'] += 25
        else:
            analysis['risk_points'] += 10

    # Check for near-duplicates (same values in most columns)
    if len(df.columns) > 1:
        # Sample for performance if dataset is large
        sample_size = min(1000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df

        # Check for rows that are similar in most columns
        numeric_cols = df_sample.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            try:
                # Calculate pairwise similarity
                near_duplicate_threshold = 0.95
                similarity_count = 0

                for i in range(min(100, len(df_sample))):  # Limit to prevent long computation
                    for j in range(i+1, min(100, len(df_sample))):
                        row1 = df_sample.iloc[i][numeric_cols]
                        row2 = df_sample.iloc[j][numeric_cols]

                        # Calculate similarity (ignore NaN)
                        mask = ~(row1.isna() | row2.isna())
                        if mask.sum() > 0:
                            similarity = np.corrcoef(row1[mask], row2[mask])[0,1]
                            if not np.isnan(similarity) and similarity > near_duplicate_threshold:
                                similarity_count += 1

                if similarity_count > 5:  # Arbitrary threshold
                    analysis['duplicate_analysis']['near_duplicates'] = similarity_count
                    analysis['issues'].append(f"🔍 Near-duplicate patterns detected: {similarity_count} similar row pairs")
                    analysis['risk_points'] += 20

            except Exception:
                pass  # Skip if similarity calculation fails

    return analysis

def detect_feature_leakage(df: pd.DataFrame, target_col: Optional[str]) -> Dict[str, Any]:
    """
    Detect feature leakage including perfect correlations and suspicious feature relationships.
    """
    analysis = {
        'risk_points': 0,
        'max_risk_points': 150,
        'issues': [],
        'suspicious_columns': [],
        'perfect_correlations': [],
        'high_correlations': [],
        'constant_features': []
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    if len(numeric_cols) < 2:
        return analysis

    # Calculate correlation matrix
    try:
        corr_matrix = df[numeric_cols].corr()

        # Find perfect correlations (excluding diagonal)
        perfect_corr_pairs = []
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]

                if not np.isnan(corr_val):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]

                    # Perfect correlation (likely duplicate features)
                    if abs(corr_val) > 0.99:
                        perfect_corr_pairs.append({
                            'feature1': col1,
                            'feature2': col2,
                            'correlation': round(corr_val, 4)
                        })
                        analysis['suspicious_columns'].extend([col1, col2])
                        analysis['issues'].append(f"🔗 Perfect correlation: '{col1}' ↔ '{col2}' (r={corr_val:.3f})")
                        analysis['risk_points'] += 25

                    # High correlation (potential redundancy)
                    elif abs(corr_val) > 0.95:
                        high_corr_pairs.append({
                            'feature1': col1,
                            'feature2': col2,
                            'correlation': round(corr_val, 4)
                        })
                        analysis['suspicious_columns'].extend([col1, col2])
                        analysis['issues'].append(f"⚠️ High correlation: '{col1}' ↔ '{col2}' (r={corr_val:.3f})")
                        analysis['risk_points'] += 15

        analysis['perfect_correlations'] = perfect_corr_pairs
        analysis['high_correlations'] = high_corr_pairs

    except Exception:
        analysis['issues'].append("⚠️ Could not calculate correlation matrix")

    # Check for constant features
    for col in df.columns:
        if col == target_col:
            continue

        if df[col].nunique() <= 1:
            analysis['constant_features'].append(col)
            analysis['suspicious_columns'].append(col)
            analysis['issues'].append(f"📊 Constant feature: '{col}' has no variation")
            analysis['risk_points'] += 10

    # Check for features with impossible values or suspicious distributions
    for col in numeric_cols:
        try:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue

            # Check for suspicious patterns
            if col_data.std() == 0:
                continue  # Already caught as constant

            # Check for features that are too good to be true (very low variance relative to mean)
            if col_data.mean() != 0:
                cv = col_data.std() / abs(col_data.mean())
                if cv < 0.01 and col_data.nunique() > 10:  # Very low coefficient of variation
                    analysis['suspicious_columns'].append(col)
                    analysis['issues'].append(f"📈 Suspiciously low variance: '{col}' (CV={cv:.4f})")
                    analysis['risk_points'] += 15

        except Exception:
            continue

    return analysis

def detect_identifier_leakage(df: pd.DataFrame, id_cols: List[str], target_col: Optional[str]) -> Dict[str, Any]:
    """
    Detect leakage from identifier columns that shouldn't be predictive.
    """
    analysis = {
        'risk_points': 0,
        'max_risk_points': 50,
        'issues': [],
        'suspicious_columns': [],
        'high_cardinality_features': [],
        'id_leakage_features': []
    }

    # Check provided ID columns
    for col in id_cols:
        if col not in df.columns or col == target_col:
            continue

        unique_ratio = df[col].nunique() / len(df)

        if unique_ratio > 0.95:
            analysis['high_cardinality_features'].append({
                'column': col,
                'unique_ratio': round(unique_ratio, 3),
                'unique_count': df[col].nunique()
            })
            analysis['suspicious_columns'].append(col)
            analysis['issues'].append(f"🆔 High-cardinality ID column: '{col}' ({unique_ratio:.1%} unique)")
            analysis['risk_points'] += 15

    # Auto-detect potential ID columns
    id_keywords = ['id', 'key', 'index', 'uuid', 'guid', 'identifier']

    for col in df.columns:
        if col == target_col or col in id_cols:
            continue

        col_lower = col.lower()
        unique_ratio = df[col].nunique() / len(df)

        # Check for ID-like column names with high cardinality
        is_id_like = any(keyword in col_lower for keyword in id_keywords)

        if is_id_like and unique_ratio > 0.8:
            analysis['id_leakage_features'].append({
                'column': col,
                'unique_ratio': round(unique_ratio, 3),
                'reason': 'ID-like name with high cardinality'
            })
            analysis['suspicious_columns'].append(col)
            analysis['issues'].append(f"🔍 Potential ID leakage: '{col}' - ID-like name with {unique_ratio:.1%} unique values")
            analysis['risk_points'] += 10

        # Check for extremely high cardinality in any column
        elif unique_ratio > 0.98 and df[col].nunique() > 100:
            analysis['high_cardinality_features'].append({
                'column': col,
                'unique_ratio': round(unique_ratio, 3),
                'unique_count': df[col].nunique()
            })
            analysis['suspicious_columns'].append(col)
            analysis['issues'].append(f"📊 Extremely high cardinality: '{col}' ({df[col].nunique()} unique values)")
            analysis['risk_points'] += 5

    return analysis

def detect_statistical_leakage(df: pd.DataFrame, target_col: Optional[str]) -> Dict[str, Any]:
    """
    Detect statistical anomalies that might indicate leakage.
    """
    analysis = {
        'risk_points': 0,
        'max_risk_points': 50,
        'issues': [],
        'suspicious_columns': [],
        'distribution_anomalies': [],
        'correlation_anomalies': []
    }

    if not target_col or target_col not in df.columns:
        return analysis

    target_data = df[target_col].dropna()
    feature_cols = [col for col in df.columns if col != target_col]

    # Check for features with distributions that are too perfect
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        col_data = df[col].dropna()
        if len(col_data) < 10:
            continue

        try:
            # Check for perfect normal distribution (suspicious)
            _, p_value = stats.normaltest(col_data)
            if p_value > 0.99:  # Too perfectly normal
                analysis['distribution_anomalies'].append({
                    'column': col,
                    'anomaly': 'perfectly_normal',
                    'p_value': p_value
                })
                analysis['suspicious_columns'].append(col)
                analysis['issues'].append(f"📊 Suspiciously perfect distribution: '{col}' (p={p_value:.3f})")
                analysis['risk_points'] += 10

            # Check for impossible statistical properties
            if col_data.std() > 0:
                skewness = stats.skew(col_data)
                kurtosis = stats.kurtosis(col_data)

                # Extremely high kurtosis might indicate artificial data
                if abs(kurtosis) > 20:
                    analysis['distribution_anomalies'].append({
                        'column': col,
                        'anomaly': 'extreme_kurtosis',
                        'kurtosis': kurtosis
                    })
                    analysis['suspicious_columns'].append(col)
                    analysis['issues'].append(f"📈 Extreme kurtosis: '{col}' (κ={kurtosis:.1f})")
                    analysis['risk_points'] += 10

        except Exception:
            continue

    # Check for categorical features with suspicious relationships to target
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    for col in categorical_cols:
        try:
            # Create contingency table
            contingency_table = pd.crosstab(df[col], df[target_col])

            # Chi-square test for independence
            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                chi2, p_value, _, _ = chi2_contingency(contingency_table)

                # Extremely significant association might indicate leakage
                if p_value < 1e-10 and chi2 > 100:
                    analysis['correlation_anomalies'].append({
                        'column': col,
                        'chi2': chi2,
                        'p_value': p_value
                    })
                    analysis['suspicious_columns'].append(col)
                    analysis['issues'].append(f"🔗 Extreme categorical association: '{col}' with target (χ²={chi2:.1f}, p={p_value:.2e})")
                    analysis['risk_points'] += 15

        except Exception:
            continue

    return analysis

def auto_detect_date_columns(df: pd.DataFrame) -> List[str]:
    """
    Automatically detect date/time columns in the DataFrame.
    """
    date_cols = []

    # Check existing datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64', 'datetimetz']).columns.tolist()
    date_cols.extend(datetime_cols)

    # Check for columns that might be dates based on name
    date_keywords = ['date', 'time', 'timestamp', 'created', 'updated', 'modified', 'year', 'month', 'day']

    for col in df.columns:
        if col in date_cols:
            continue

        col_lower = col.lower()

        # Check for date keywords in column name
        if any(keyword in col_lower for keyword in date_keywords):
            try:
                # Try to parse as datetime
                parsed = pd.to_datetime(df[col].dropna().head(100), errors='coerce')
                if parsed.notna().sum() > len(parsed) * 0.8:  # 80% parseable
                    date_cols.append(col)
            except:
                pass

    return date_cols

def auto_detect_id_columns(df: pd.DataFrame) -> List[str]:
    """
    Automatically detect identifier columns in the DataFrame.
    """
    id_cols = []
    id_keywords = ['id', 'key', 'index', 'uuid', 'guid', 'identifier', 'number', 'code']

    for col in df.columns:
        col_lower = col.lower()
        unique_ratio = df[col].nunique() / len(df)

        # Check for ID keywords and high uniqueness
        if any(keyword in col_lower for keyword in id_keywords) and unique_ratio > 0.8:
            id_cols.append(col)

        # Check for columns that are likely IDs based on patterns
        elif unique_ratio > 0.95 and df[col].nunique() > 100:
            # Additional check: see if values look like IDs
            sample_values = df[col].dropna().astype(str).head(10).tolist()
            id_like_patterns = 0

            for val in sample_values:
                # Check for typical ID patterns
                if (val.isdigit() or
                    '-' in val or
                    '_' in val or
                    len(val) > 10):
                    id_like_patterns += 1

            if id_like_patterns > len(sample_values) * 0.7:
                id_cols.append(col)

    return id_cols

def classify_risk_level(risk_score: float) -> str:
    """
    Classify the overall leakage risk level based on risk score.
    """
    if risk_score >= 80:
        return "Critical"
    elif risk_score >= 60:
        return "High"
    elif risk_score >= 40:
        return "Medium"
    elif risk_score >= 20:
        return "Low"
    else:
        return "Minimal"

def generate_leakage_recommendations(results: Dict[str, Any]) -> List[str]:
    """
    Generate specific recommendations based on leakage analysis results.
    """
    recommendations = []
    risk_score = results['risk_score']

    # General recommendations based on risk level
    if risk_score >= 80:
        recommendations.append("🚨 CRITICAL: Immediately review and remove identified leakage sources before model training")
        recommendations.append("🔍 Conduct thorough feature engineering review with domain experts")

    elif risk_score >= 60:
        recommendations.append("⚠️ HIGH RISK: Carefully investigate all flagged features before proceeding")
        recommendations.append("🧹 Consider removing or transforming highly suspicious features")

    elif risk_score >= 40:
        recommendations.append("⚠️ MEDIUM RISK: Review flagged features and validate with domain knowledge")
        recommendations.append("📊 Monitor model performance for signs of overfitting")

    elif risk_score >= 20:
        recommendations.append("✅ LOW RISK: Minor issues detected - review flagged items as precaution")

    else:
        recommendations.append("✅ MINIMAL RISK: No significant leakage detected")

    # Specific recommendations based on detected issues
    leakage_types = results['leakage_types']

    if leakage_types['target_leakage']['risk_points'] > 0:
        recommendations.append("🎯 Remove or investigate perfect/near-perfect target predictors")
        recommendations.append("📝 Review feature names for target-related terminology")

    if leakage_types['temporal_leakage']['risk_points'] > 0:
        recommendations.append("🕐 Ensure temporal ordering: features must be available before prediction time")
        recommendations.append("📅 Remove or adjust features with future information")

    if leakage_types['data_duplication']['risk_points'] > 0:
        recommendations.append("📋 Remove duplicate rows to prevent train/test contamination")
        recommendations.append("🔍 Investigate data collection process for duplication sources")

    if leakage_types['feature_leakage']['risk_points'] > 0:
        recommendations.append("🔗 Remove perfectly correlated features (keep one from each pair)")
        recommendations.append("📊 Consider dimensionality reduction for highly correlated features")

    if leakage_types['identifier_leakage']['risk_points'] > 0:
        recommendations.append("🆔 Remove or exclude high-cardinality ID columns from modeling")
        recommendations.append("🔒 Ensure identifiers don't encode target information")

    if leakage_types['statistical_leakage']['risk_points'] > 0:
        recommendations.append("📈 Investigate features with suspicious statistical properties")
        recommendations.append("🧮 Validate data generation process for artificial patterns")

    # Add cross-validation recommendations
    if risk_score > 20:
        recommendations.extend([
            "✅ Use time-series aware cross-validation if temporal data is involved",
            "🎲 Implement stratified sampling to maintain data integrity",
            "📏 Monitor for significant performance drops between train and validation"
        ])

    return recommendations

def get_leakage_summary_stats(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary statistics for the leakage analysis.
    """
    stats = {
        'total_suspicious_features': len(results['suspicious_features']),
        'total_issues': len(results['detailed_findings']),
        'risk_breakdown': {},
        'most_problematic_type': '',
        'action_priority': 'Low'
    }

    # Calculate risk breakdown by type
    leakage_types = results['leakage_types']
    max_risk = 0
    max_type = ''

    for leak_type, analysis in leakage_types.items():
        if isinstance(analysis, dict) and 'risk_points' in analysis:
            risk_pct = (analysis['risk_points'] / analysis.get('max_risk_points', 1)) * 100
            stats['risk_breakdown'][leak_type] = {
                'risk_points': analysis['risk_points'],
                'max_points': analysis.get('max_risk_points', 0),
                'risk_percentage': round(risk_pct, 1)
            }

            if analysis['risk_points'] > max_risk:
                max_risk = analysis['risk_points']
                max_type = leak_type

    stats['most_problematic_type'] = max_type

    # Determine action priority
    risk_score = results['risk_score']
    if risk_score >= 60:
        stats['action_priority'] = 'Critical'
    elif risk_score >= 40:
        stats['action_priority'] = 'High'
    elif risk_score >= 20:
        stats['action_priority'] = 'Medium'
    else:
        stats['action_priority'] = 'Low'

    return stats