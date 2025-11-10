"""
Generate Test Files with Data Quality Issues

This script creates sample data files with various data quality issues
to test the Smart Data Cleaning and Anomaly Explanation features.
"""

import pandas as pd
import numpy as np
import json
import os

def create_dirty_data():
    """Create sample dataset with multiple data quality issues"""
    np.random.seed(42)

    # Start with 150 rows
    n_rows = 150

    data = {
        'customer_id': range(1, n_rows + 1),
        'name': [f'Customer_{i}' for i in range(1, n_rows + 1)],
        'age': np.random.randint(18, 80, n_rows),
        'email': [f'customer{i}@example.com' for i in range(1, n_rows + 1)],
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_rows),
        'salary': np.random.randint(30000, 120000, n_rows),
        'purchase_amount': np.random.uniform(10, 1000, n_rows).round(2),
        'department': np.random.choice(['Sales', 'Marketing', 'Engineering', 'HR', 'Finance'], n_rows),
        'years_experience': np.random.randint(0, 30, n_rows),
        'performance_score': np.random.uniform(60, 100, n_rows).round(2),
        'join_date': pd.date_range('2020-01-01', periods=n_rows, freq='2D')
    }

    df = pd.DataFrame(data)

    print("Creating data quality issues...")
    print(f"Starting with {len(df)} rows, {len(df.columns)} columns")
    print()

    # 1. Add Missing Values (20% of certain columns)
    print("1. Adding missing values...")
    missing_indices_age = np.random.choice(df.index, size=int(0.15 * len(df)), replace=False)
    df.loc[missing_indices_age, 'age'] = np.nan

    missing_indices_email = np.random.choice(df.index, size=int(0.10 * len(df)), replace=False)
    df.loc[missing_indices_email, 'email'] = np.nan

    missing_indices_salary = np.random.choice(df.index, size=int(0.08 * len(df)), replace=False)
    df.loc[missing_indices_salary, 'salary'] = np.nan

    print(f"   - age: {df['age'].isna().sum()} missing values ({df['age'].isna().sum()/len(df)*100:.1f}%)")
    print(f"   - email: {df['email'].isna().sum()} missing values ({df['email'].isna().sum()/len(df)*100:.1f}%)")
    print(f"   - salary: {df['salary'].isna().sum()} missing values ({df['salary'].isna().sum()/len(df)*100:.1f}%)")
    print()

    # 2. Add Duplicates (10 exact duplicates)
    print("2. Adding duplicate rows...")
    duplicate_rows = df.sample(10).copy()
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    print(f"   - Added 10 duplicate rows")
    print(f"   - Total rows now: {len(df)}")
    print()

    # 3. Add Outliers
    print("3. Adding outliers...")

    # Age outliers (negative ages and very high ages)
    outlier_indices_age = np.random.choice(df.index, size=5, replace=False)
    df.loc[outlier_indices_age[0:2], 'age'] = [-5, -10]  # Invalid negative ages
    df.loc[outlier_indices_age[2:5], 'age'] = [150, 200, 999]  # Unrealistic high ages
    print(f"   - Added age outliers: negative ages and ages > 100")

    # Salary outliers
    outlier_indices_salary = np.random.choice(df.index, size=8, replace=False)
    df.loc[outlier_indices_salary[0:3], 'salary'] = [500000, 750000, 1000000]  # Very high salaries
    df.loc[outlier_indices_salary[3:5], 'salary'] = [5000, 8000]  # Very low salaries
    df.loc[outlier_indices_salary[5:8], 'salary'] = [-1000, -5000, 0]  # Invalid salaries
    print(f"   - Added salary outliers: very high, very low, and negative values")

    # Purchase amount outliers
    outlier_indices_purchase = np.random.choice(df.index, size=5, replace=False)
    df.loc[outlier_indices_purchase, 'purchase_amount'] = [10000, 15000, -500, -100, 25000]
    print(f"   - Added purchase_amount outliers: very high and negative values")

    # Performance score outliers
    outlier_indices_perf = np.random.choice(df.index, size=4, replace=False)
    df.loc[outlier_indices_perf, 'performance_score'] = [150, 200, -10, 0]
    print(f"   - Added performance_score outliers: > 100 and negative values")
    print()

    # 4. Add Format Issues
    print("4. Adding format inconsistencies...")

    # Email format issues
    format_indices_email = np.random.choice(df[df['email'].notna()].index, size=10, replace=False)
    df.loc[format_indices_email[0:3], 'email'] = ['invalid.email', 'no@domain', 'bad-format']
    df.loc[format_indices_email[3:6], 'email'] = ['UPPERCASE@EXAMPLE.COM', 'MixedCase@Example.Com', 'lower@example.com']
    df.loc[format_indices_email[6:10], 'email'] = ['  spaces@example.com  ', 'extra..dots@example.com', 'no-at-symbol.com', '@nodomain.com']
    print(f"   - Added email format issues: invalid formats, case inconsistencies, spaces")

    # Name format issues
    format_indices_name = np.random.choice(df.index, size=8, replace=False)
    df.loc[format_indices_name[0:3], 'name'] = ['', '   ', 'N/A']
    df.loc[format_indices_name[3:5], 'name'] = ['CUSTOMER_UPPERCASE', 'customer_lowercase']
    df.loc[format_indices_name[5:8], 'name'] = ['Customer With Spaces   ', '  Leading Spaces', 'Trailing Spaces  ']
    print(f"   - Added name format issues: empty values, case inconsistencies, extra spaces")

    # City format issues
    format_indices_city = np.random.choice(df.index, size=6, replace=False)
    df.loc[format_indices_city[0:3], 'city'] = ['new york', 'LOS ANGELES', 'ChIcAgO']
    df.loc[format_indices_city[3:6], 'city'] = ['Unknown', 'N/A', '']
    print(f"   - Added city format issues: case inconsistencies, unknown values")
    print()

    # 5. Add Data Type Issues
    print("5. Adding data type inconsistencies...")

    # Years experience negative values
    type_indices_exp = np.random.choice(df.index, size=5, replace=False)
    df.loc[type_indices_exp, 'years_experience'] = [-1, -5, -10, 100, 150]
    print(f"   - Added invalid years_experience: negative values and unrealistic high values")
    print()

    # Summary
    print("="*60)
    print("Data Quality Issues Summary")
    print("="*60)
    print(f"Total Rows: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print()
    print(f"Missing Values: {df.isnull().sum().sum()} total")
    print(f"Duplicate Rows: {df.duplicated().sum()}")
    print(f"Outliers (approximate):")
    print(f"  - Age: {len(df[(df['age'] < 0) | (df['age'] > 100)])}")
    print(f"  - Salary: {len(df[(df['salary'] < 10000) | (df['salary'] > 200000) | (df['salary'] < 0)])}")
    print(f"  - Purchase Amount: {len(df[(df['purchase_amount'] < 0) | (df['purchase_amount'] > 5000)])}")
    print(f"  - Performance Score: {len(df[(df['performance_score'] < 50) | (df['performance_score'] > 100)])}")
    print()

    return df

def save_dirty_data():
    """Generate and save dirty data in multiple formats"""

    df = create_dirty_data()

    output_dir = 'test_files_dirty'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print("\nSaving files...")

    # Save in multiple formats
    print("  [+] Saving dirty_data.csv")
    df.to_csv(os.path.join(output_dir, 'dirty_data.csv'), index=False)

    print("  [+] Saving dirty_data.xlsx")
    df.to_excel(os.path.join(output_dir, 'dirty_data.xlsx'), index=False)

    print("  [+] Saving dirty_data.json")
    df.to_json(os.path.join(output_dir, 'dirty_data.json'), orient='records', indent=2, date_format='iso')

    print("  [+] Saving dirty_data.parquet")
    df.to_parquet(os.path.join(output_dir, 'dirty_data.parquet'), index=False)

    print("\n[SUCCESS] Dirty data files created successfully!")
    print(f"\nFiles saved in: {output_dir}/")
    print("  - dirty_data.csv")
    print("  - dirty_data.xlsx")
    print("  - dirty_data.json")
    print("  - dirty_data.parquet")
    print()
    print("="*60)
    print("Testing Instructions")
    print("="*60)
    print("1. Upload any of these files to the EDA Tool")
    print("2. Navigate to 'Data Quality' to see detected issues")
    print("3. Go to 'Smart Data Cleaning' to get AI recommendations")
    print("4. Use 'Anomaly Explanation' to understand outliers")
    print("="*60)

if __name__ == "__main__":
    print("="*60)
    print("Dirty Data Generator - For Testing Data Quality Features")
    print("="*60)
    print()

    try:
        save_dirty_data()
    except Exception as e:
        print(f"\n[ERROR] Error generating dirty data: {str(e)}")
        import traceback
        traceback.print_exc()
