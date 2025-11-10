"""
Generate Test Files for Data Preprocessing Feature

This script creates sample data files in various formats to test
the preprocessing capabilities of the EDA Tool.
"""

import pandas as pd
import numpy as np
import json
import sqlite3
import os

def create_sample_data():
    """Create sample dataset for testing"""
    np.random.seed(42)

    data = {
        'id': range(1, 101),
        'name': [f'Person_{i}' for i in range(1, 101)],
        'age': np.random.randint(20, 70, 100),
        'city': np.random.choice(['New York', 'San Francisco', 'Chicago', 'Boston', 'Seattle'], 100),
        'salary': np.random.randint(40000, 150000, 100),
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], 100),
        'years_experience': np.random.randint(0, 30, 100),
        'performance_score': np.random.uniform(60, 100, 100).round(2)
    }

    df = pd.DataFrame(data)
    return df

def generate_all_formats():
    """Generate test files in all supported formats"""

    print("Generating sample data...")
    df = create_sample_data()

    output_dir = 'test_files'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print("\nGenerating test files...")

    # 1. CSV
    print("  [+] Generating test.csv")
    df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    # 2. Excel
    print("  [+] Generating test.xlsx")
    df.to_excel(os.path.join(output_dir, 'test.xlsx'), index=False)

    # 3. JSON - Array of objects
    print("  [+] Generating test.json")
    df.to_json(os.path.join(output_dir, 'test.json'), orient='records', indent=2)

    # 4. JSON Lines
    print("  [+] Generating test_lines.json")
    df.to_json(os.path.join(output_dir, 'test_lines.json'), orient='records', lines=True)

    # 5. Parquet
    print("  [+] Generating test.parquet")
    try:
        df.to_parquet(os.path.join(output_dir, 'test.parquet'), index=False)
    except ImportError:
        print("    [!] Skipped (pyarrow not installed)")

    # 6. TSV
    print("  [+] Generating test.tsv")
    df.to_csv(os.path.join(output_dir, 'test.tsv'), sep='\t', index=False)

    # 7. TXT (pipe-delimited)
    print("  [+] Generating test.txt")
    df.to_csv(os.path.join(output_dir, 'test.txt'), sep='|', index=False)

    # 8. Feather
    print("  [+] Generating test.feather")
    try:
        df.to_feather(os.path.join(output_dir, 'test.feather'))
    except ImportError:
        print("    [!] Skipped (pyarrow not installed)")

    # 9. HDF5
    print("  [+] Generating test.h5")
    try:
        df.to_hdf(os.path.join(output_dir, 'test.h5'), key='data', mode='w')
    except ImportError:
        print("    [!] Skipped (tables not installed)")

    # 10. Pickle
    print("  [+] Generating test.pkl")
    df.to_pickle(os.path.join(output_dir, 'test.pkl'))

    # 11. SQLite
    print("  [+] Generating test.db")
    conn = sqlite3.connect(os.path.join(output_dir, 'test.db'))
    df.to_sql('employees', conn, index=False, if_exists='replace')

    # Create additional table for testing
    df_summary = df.groupby('department').agg({
        'salary': 'mean',
        'age': 'mean',
        'years_experience': 'mean'
    }).reset_index()
    df_summary.to_sql('department_summary', conn, index=False, if_exists='replace')
    conn.close()

    # 12. HTML
    print("  [+] Generating test.html")
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Employee Data</title>
        <style>
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid black; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
        </style>
    </head>
    <body>
        <h1>Employee Data</h1>
        {df.to_html(index=False)}
    </body>
    </html>
    """
    with open(os.path.join(output_dir, 'test.html'), 'w') as f:
        f.write(html_content)

    # 13. XML
    print("  [+] Generating test.xml")
    xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n<employees>\n'
    for _, row in df.head(20).iterrows():  # Only first 20 rows for XML
        xml_content += '  <employee>\n'
        for col in df.columns:
            xml_content += f'    <{col}>{row[col]}</{col}>\n'
        xml_content += '  </employee>\n'
    xml_content += '</employees>'

    with open(os.path.join(output_dir, 'test.xml'), 'w') as f:
        f.write(xml_content)

    print(f"\n[SUCCESS] All test files generated successfully in '{output_dir}/' directory!")
    print(f"\nGenerated files:")
    print(f"  - test.csv")
    print(f"  - test.xlsx")
    print(f"  - test.json")
    print(f"  - test_lines.json")
    print(f"  - test.parquet")
    print(f"  - test.tsv")
    print(f"  - test.txt")
    print(f"  - test.feather")
    print(f"  - test.h5")
    print(f"  - test.pkl")
    print(f"  - test.db (contains 'employees' and 'department_summary' tables)")
    print(f"  - test.html")
    print(f"  - test.xml")

    print(f"\n[INFO] Test dataset contains:")
    print(f"  - {len(df)} rows")
    print(f"  - {len(df.columns)} columns")
    print(f"  - Columns: {', '.join(df.columns)}")

    print(f"\n[READY] You can now test the EDA Tool by uploading any of these files!")

if __name__ == "__main__":
    print("=" * 70)
    print("EDA Tool - Test File Generator")
    print("=" * 70)
    print()

    try:
        generate_all_formats()
    except Exception as e:
        print(f"\n[ERROR] Error generating test files: {str(e)}")
        print("\nMake sure all required packages are installed:")
        print("  pip install pandas openpyxl pyarrow tables")
