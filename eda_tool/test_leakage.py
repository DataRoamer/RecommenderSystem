#!/usr/bin/env python3
"""
Test script for advanced leakage detection functionality
"""

import pandas as pd
import numpy as np
from modules.leakage_detection import comprehensive_leakage_detection, get_leakage_summary_stats

def create_leakage_test_dataset():
    """Create a test dataset with intentional leakage"""
    np.random.seed(42)
    n_samples = 500

    # Create base features
    data = {
        'age': np.random.normal(30, 15, n_samples),
        'income': np.random.exponential(50000, n_samples),
        'education_years': np.random.choice(range(8, 21), n_samples),
        'experience': np.random.normal(5, 8, n_samples),
    }

    # Create target
    # Make target dependent on some features
    target_prob = 0.3 + 0.0001 * data['income'] + 0.05 * data['education_years'] / 20
    target_prob = np.clip(target_prob, 0, 1)
    data['approved'] = np.random.binomial(1, target_prob)

    # Add leakage features
    # 1. Perfect predictor
    data['approval_result'] = data['approved']  # Perfect leakage

    # 2. Near-perfect predictor
    data['credit_decision'] = data['approved'].copy()
    # Add some noise to 5% of entries
    noise_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    data['credit_decision'][noise_indices] = 1 - data['credit_decision'][noise_indices]

    # 3. High cardinality ID
    data['application_id'] = [f"APP_{i:06d}" for i in range(n_samples)]

    # 4. Suspicious correlations
    data['income_duplicate'] = data['income'] * 1.001  # Near-perfect correlation

    # 5. Future information
    from datetime import datetime, timedelta
    base_date = datetime.now()
    data['processing_date'] = [
        base_date + timedelta(days=np.random.randint(-30, 60))  # Some future dates
        for _ in range(n_samples)
    ]

    df = pd.DataFrame(data)
    return df

def test_leakage_detection():
    """Test the advanced leakage detection"""
    print("Creating test dataset with intentional leakage...")
    df = create_leakage_test_dataset()

    print("Running comprehensive leakage detection...")
    results = comprehensive_leakage_detection(
        df=df,
        target_col='approved',
        date_cols=['processing_date'],
        id_cols=['application_id']
    )

    print("\n" + "="*60)
    print("LEAKAGE DETECTION RESULTS")
    print("="*60)

    print(f"Overall Risk Level: {results['overall_leakage_risk']}")
    print(f"Risk Score: {results['risk_score']:.1f}/100")
    print(f"Suspicious Features: {len(results['suspicious_features'])}")
    print(f"Total Issues: {len(results['detailed_findings'])}")

    print(f"\nSuspicious Features Found:")
    for feature in results['suspicious_features']:
        print(f"  - {feature}")

    print(f"\nDetailed Findings:")
    for finding in results['detailed_findings']:
        # Remove emojis from findings for display
        clean_finding = finding.encode('ascii', errors='ignore').decode('ascii')
        print(f"  - {clean_finding}")

    print(f"\nRecommendations:")
    for i, rec in enumerate(results['recommendations'], 1):
        # Remove emojis from recommendations for display
        clean_rec = rec.encode('ascii', errors='ignore').decode('ascii')
        print(f"  {i}. {clean_rec}")

    # Test summary stats
    summary_stats = get_leakage_summary_stats(results)
    print(f"\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Action Priority: {summary_stats['action_priority']}")
    print(f"Most Problematic Type: {summary_stats['most_problematic_type']}")

    print("\nRisk Breakdown:")
    for leak_type, breakdown in summary_stats['risk_breakdown'].items():
        print(f"  {leak_type}: {breakdown['risk_points']}/{breakdown['max_points']} points ({breakdown['risk_percentage']:.1f}%)")

    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*60)

    return results

if __name__ == "__main__":
    test_leakage_detection()