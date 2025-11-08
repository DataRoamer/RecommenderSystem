#!/usr/bin/env python3
"""
Test script to verify all EDA tool sections work properly
"""

import pandas as pd
import numpy as np
from modules.target_analysis import analyze_target_variable
from modules.feature_engineering import comprehensive_feature_engineering_report
from modules.model_readiness import calculate_comprehensive_readiness_score
from modules.eda_analysis import comprehensive_eda_report
from modules.data_quality import comprehensive_data_quality_report

def create_test_dataset():
    """Create a synthetic test dataset for testing"""
    np.random.seed(42)

    # Create synthetic data similar to Titanic dataset
    n_samples = 1000

    data = {
        'age': np.random.normal(30, 15, n_samples),
        'fare': np.random.exponential(50, n_samples),
        'pclass': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5]),
        'sex': np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4]),
        'embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1]),
        'survived': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    }

    # Add some missing values
    missing_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    data['age'] = pd.Series(data['age'])
    data['age'].iloc[missing_indices[:50]] = np.nan

    df = pd.DataFrame(data)

    # Add some outliers
    df.loc[df.sample(20).index, 'fare'] = np.random.uniform(300, 500, 20)

    return df

def test_target_analysis():
    """Test target analysis functionality"""
    print("Testing Target Analysis...")
    df = create_test_dataset()

    # Test with classification target
    result = analyze_target_variable(df, 'survived')

    assert 'basic_info' in result
    assert 'classification_analysis' in result
    assert 'recommendations' in result
    assert 'ml_readiness' in result

    print("[PASS] Target Analysis: Classification test passed")

    # Test with regression target
    result = analyze_target_variable(df, 'fare')

    assert 'basic_info' in result
    assert 'regression_analysis' in result
    assert 'recommendations' in result
    assert 'ml_readiness' in result

    print("[PASS] Target Analysis: Regression test passed")

    return True

def test_feature_engineering():
    """Test feature engineering functionality"""
    print("Testing Feature Engineering...")
    df = create_test_dataset()

    result = comprehensive_feature_engineering_report(df)

    assert 'feature_classification' in result
    assert 'encoding_recommendations' in result
    assert 'engineering_suggestions' in result
    assert 'features_to_drop' in result
    assert 'summary' in result

    print("[PASS] Feature Engineering test passed")

    return True

def test_model_readiness():
    """Test model readiness functionality"""
    print("Testing Model Readiness...")
    df = create_test_dataset()

    # First get required reports
    from modules.data_quality import comprehensive_data_quality_report
    from modules.target_analysis import analyze_target_variable
    from modules.feature_engineering import comprehensive_feature_engineering_report

    quality_report = comprehensive_data_quality_report(df)
    target_analysis = analyze_target_variable(df, 'survived')
    feature_report = comprehensive_feature_engineering_report(df)

    # Now test model readiness
    result = calculate_comprehensive_readiness_score(
        quality_report=quality_report,
        target_analysis=target_analysis,
        feature_engineering_report=feature_report,
        df=df
    )

    assert 'overall_score' in result
    assert 'category_scores' in result
    assert 'detailed_assessment' in result
    assert 'priority_actions' in result
    assert 'strengths' in result

    print("[PASS] Model Readiness test passed")

    return True

def test_eda_analysis():
    """Test EDA analysis functionality"""
    print("Testing EDA Analysis...")
    df = create_test_dataset()

    result = comprehensive_eda_report(df)

    assert 'summary_statistics' in result
    assert 'correlations' in result
    assert 'distributions' in result

    print("[PASS] EDA Analysis test passed")

    return True

def test_data_quality():
    """Test data quality functionality"""
    print("Testing Data Quality...")
    df = create_test_dataset()

    result = comprehensive_data_quality_report(df)

    assert 'quality_score' in result
    assert 'missing_values' in result
    assert 'duplicates' in result
    assert 'outliers_zscore' in result

    print("[PASS] Data Quality test passed")

    return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("TESTING EDA TOOL FUNCTIONALITY")
    print("=" * 50)

    try:
        # Test all sections
        test_data_quality()
        test_eda_analysis()
        test_target_analysis()
        test_feature_engineering()
        test_model_readiness()

        print("\n" + "=" * 50)
        print("SUCCESS! ALL TESTS PASSED!")
        print("All sections are working properly.")
        print("=" * 50)

    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()