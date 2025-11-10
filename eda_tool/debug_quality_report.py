"""
Debug script to test quality report generation with preprocessed data
"""

import pandas as pd
from modules.data_loader import load_data
from modules.data_quality import comprehensive_data_quality_report

# Load the JSON test file
with open('test_files/test.json', 'rb') as f:
    df, metadata = load_data(f, 'json', filename='test.json')

print("Data loaded successfully:")
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Dtypes: {df.dtypes.to_dict()}")
print()

# Generate quality report
print("Generating quality report...")
try:
    quality_report = comprehensive_data_quality_report(df)
    print("Quality report generated successfully!")
    print()

    # Check if missing_values exists
    if 'missing_values' in quality_report:
        print(f"missing_values type: {type(quality_report['missing_values'])}")
        if quality_report['missing_values'] is not None:
            print(f"missing_values keys: {quality_report['missing_values'].keys()}")
            print(f"overall_missing_percentage: {quality_report['missing_values'].get('overall_missing_percentage', 'NOT FOUND')}")
        else:
            print("ERROR: missing_values is None!")
    else:
        print("ERROR: missing_values key not in quality_report!")

    print()
    print("Quality report keys:")
    for key in quality_report.keys():
        value_type = type(quality_report[key])
        is_none = quality_report[key] is None
        print(f"  {key}: {value_type} (None: {is_none})")

except Exception as e:
    print(f"ERROR generating quality report: {e}")
    import traceback
    traceback.print_exc()
